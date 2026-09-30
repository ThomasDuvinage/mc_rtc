/*
 * Copyright 2015-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_rbdyn/RobotModule.h>
#include <mc_rtc/config.h>
#include <mc_rtc/path.h>

#include <mc_rbdyn/Robot.h>

#include <mesh_sampling/mesh_sampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace
{

// A single mesh file can be reused by several bodies with a mirrored URDF
// <mesh scale="..."> (a negative component on one axis), e.g. to share one
// mesh between a robot's left/right sides. mesh_sampling only supports a
// uniform scale factor (see meshScale() in generate_convexes()), so a
// mirrored reuse needs its own, separately-mirrored convex hull file --
// otherwise every body sharing that mesh file ends up bound to the same
// (arbitrarily-mirrored-or-not) hull, regardless of which of them is
// actually mirrored.
using MirrorAxes = std::array<bool, 3>;

MirrorAxes meshMirrorAxes(const rbd::parsers::Geometry::Mesh & mesh)
{
  return {mesh.scaleV.x() < 0, mesh.scaleV.y() < 0, mesh.scaleV.z() < 0};
}

std::string mirrorSuffix(const MirrorAxes & mirror)
{
  std::string suffix;
  if(mirror[0]) { suffix += "x"; }
  if(mirror[1]) { suffix += "y"; }
  if(mirror[2]) { suffix += "z"; }
  return suffix.empty() ? std::string{} : "-mirror" + suffix;
}

// Convex hull files are qhull's "o f" (OFF-like) output: a dimension line, a
// "<numVertices> <numFacets> <numRidges>" header, that many "x y z" vertex
// lines, then facet/ridge lines that reference vertices purely by index.
// Reflecting the vertices while leaving those indices untouched still
// yields a valid convex polytope description, so mirroring only requires
// rewriting the vertex block.
void writeMirroredConvexFile(const fs::path & src, const fs::path & dst, const MirrorAxes & mirror)
{
  std::ifstream ifs(src);
  if(!ifs.is_open())
  {
    mc_rtc::log::error("[RobotModule] Could not open convex file {} for mirroring", src.string());
    return;
  }
  std::ofstream ofs(dst);
  if(!ofs.is_open())
  {
    mc_rtc::log::error("[RobotModule] Could not create mirrored convex file {}", dst.string());
    return;
  }
  std::string dim_line;
  std::getline(ifs, dim_line);
  ofs << dim_line << "\n";
  std::string header_line;
  std::getline(ifs, header_line);
  ofs << header_line << "\n";
  size_t numVertices = 0;
  std::istringstream(header_line) >> numVertices;
  for(size_t i = 0; i < numVertices && ifs; ++i)
  {
    double x, y, z;
    ifs >> x >> y >> z;
    ofs << (mirror[0] ? -x : x) << " " << (mirror[1] ? -y : y) << " " << (mirror[2] ? -z : z) << "\n";
  }
  ifs >> std::ws; // consume trailing whitespace/newline left after the last "x y z" extraction
  ofs << ifs.rdbuf();
}

} // namespace

namespace mc_rbdyn
{

// Repeat static constexpr declarations
// See also https://stackoverflow.com/q/8016780
constexpr double RobotModule::Gripper::Safety::DEFAULT_PERCENT_VMAX;
constexpr double RobotModule::Gripper::Safety::DEFAULT_ACTUAL_COMMAND_DIFF_TRIGGER;
constexpr double RobotModule::Gripper::Safety::DEFAULT_RELEASE_OFFSET;
constexpr unsigned int RobotModule::Gripper::Safety::DEFAULT_OVER_COMMAND_LIMIT_ITER_N;

DevicePtrVector::DevicePtrVector(const DevicePtrVector & v) : std::vector<DevicePtr>()
{
  reserve(v.size());
  for(const auto & s : v) { push_back(s->clone()); }
}

DevicePtrVector & DevicePtrVector::operator=(const DevicePtrVector & v)
{
  if(&v == this) { return *this; }
  resize(v.size());
  for(size_t i = 0; i < v.size(); ++i) { (*this)[i] = v[i]->clone(); }
  return *this;
}

RobotModule::RobotModule(const std::string & name, const rbd::parsers::ParserResult & res)
: RobotModule("/CREATED/BY/MC/RTC/", name)
{
  rsdf_dir = "";
  convex_dir = "";
  init(res);
}

void RobotModule::init(const rbd::parsers::ParserResult & res)
{
  mb = res.mb;
  mbc = res.mbc;
  mbg = res.mbg;
  for(const auto & col : res.collision)
  {
    const auto & body = col.first;
    const auto & cols = col.second;
    if(cols.size()) { _collisionTransforms[body] = cols[0].origin; }
  }
  boundsFromURDF(res.limits);
  _visual = res.visual;
  _collision = res.collision;
  if(_ref_joint_order.size() == 0) { make_default_ref_joint_order(); }
  expand_stance();
  // FIXME revert this once https://github.com/jrl-umi3218/mesh_sampling/pull/6 has been resolved.
  const char * disableConvexGen = std::getenv("MC_RTC_DISABLE_CONVEX_GENERATION_PATCH");
  if(!(disableConvexGen && std::string(disableConvexGen) == "ON"))
  {
    generate_convexes();
    bind_convexes();
  }
}

void RobotModule::generate_convexes(bool regenerate, unsigned int sampling_point)
{
  // A mesh's URDF <collision><geometry><mesh scale="x y z"> is parsed into
  // Geometry::Mesh::scaleV, but mesh_sampling (via Assimp's
  // AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY) only supports a single uniform scale
  // factor -- it cannot represent an anisotropic scale or an axis mirror
  // (negative component, commonly used to mirror a single mesh file between
  // a robot's left/right sides). We take the scale's average magnitude
  // (correct for the common case of a uniform scale, e.g. a mm->m
  // conversion, with or without a mirrored axis) and warn when that
  // assumption doesn't hold, since silently ignoring scale entirely (as
  // before) produces convex hulls that can be orders of magnitude larger or
  // smaller than the actual collision mesh.
  auto meshScale = [&](const rbd::parsers::Geometry::Mesh & mesh) -> float
  {
    const auto & s = mesh.scaleV;
    double sx = std::abs(s.x());
    double sy = std::abs(s.y());
    double sz = std::abs(s.z());
    double avg = (sx + sy + sz) / 3.0;
    double maxDev = std::max({std::abs(sx - avg), std::abs(sy - avg), std::abs(sz - avg)});
    if(avg > 0 && maxDev / avg > 1e-3)
    {
      mc_rtc::log::warning("[RobotModule] Mesh {} has a non-uniform scale ({}, {}, {}), convex hull generation only "
                           "supports a uniform scale factor and will use their average magnitude ({}); the "
                           "generated convex hull may not exactly match the collision mesh",
                           mesh.filename, s.x(), s.y(), s.z(), avg);
    }
    return static_cast<float>(avg);
  };

  auto urdfCRC = [&]()
  {
    uint32_t crc = 0;
    for(const auto & col : _collision)
    {
      for(const auto & c : col.second)
      {
        if(c.geometry.type == rbd::parsers::Geometry::Type::MESH)
        {
          const auto & mesh = boost::get<rbd::parsers::Geometry::Mesh>(c.geometry.data);
          const auto & mesh_path = fs::path(mesh.filename);
          crc += fs::file_size(mc_rtc::convertURI(mesh_path));
          // Fold the scale into the cache key so that a scale change (in
          // particular, this fix starting to honor a scale that was
          // previously silently ignored) invalidates stale cached convexes.
          crc += static_cast<uint32_t>(meshScale(mesh) * 1e6);
          // Same for mirroring support: a mesh that needs a mirrored convex
          // (see writeMirroredConvexFile()) bumps the cache key so it gets
          // (re)generated once this support is added.
          if(!mirrorSuffix(meshMirrorAxes(mesh)).empty()) { crc += 7919; }
        }
      }
    }
    return crc;
  };

  auto generate =
      [&](const fs::path & mesh_file, const fs::path & output_convex_dir, float scale, const MirrorAxes & mirror)
  {
    if(!fs::exists(mesh_file))
    {
      mc_rtc::log::error_and_throw("Couldn't find meshes at {} for {}", mesh_file.string(), name);
    }

    mesh_sampling::MeshSampling sampler(mesh_file, scale);
    const auto & meshes = sampler.create_clouds(sampling_point);
    // Note: create_convexes()'s return value maps each cloud name to the
    // generated file's *content* (qhull's text output), not its path -- the
    // path it wrote to is reconstructed below instead, following the same
    // naming rule create_convexes() uses internally.
    sampler.create_convexes(meshes, output_convex_dir, false);

    auto suffix = mirrorSuffix(mirror);
    if(!suffix.empty())
    {
      for(const auto & cloud : meshes)
      {
        std::string stem = fs::path(cloud.first).filename().stem().string();
        fs::path base_path = output_convex_dir / (stem + "-ch.txt");
        if(!fs::exists(base_path))
        {
          mc_rtc::log::error("[RobotModule] Expected convex file {} was not generated, cannot create its mirrored "
                             "variant",
                             base_path.string());
          continue;
        }
        fs::path mirrored_path = output_convex_dir / (stem + suffix + "-ch.txt");
        writeMirroredConvexFile(base_path, mirrored_path, mirror);
      }
    }
  };

  if(!name.empty())
  {
    auto cache_root = fs::path(mc_rtc::local_share_directory(name));
    fs::create_directories(cache_root);

    std::string variant_name = fs::path(urdf_path).filename().stem().string();
    std::string target_folder_name = variant_name + "-" + std::to_string(urdfCRC());
    convex_dir = cache_root / target_folder_name;

    if(!fs::exists(convex_dir) || regenerate)
    {
      // Clean up any old folders for the same variant
      for(const auto & cache_entry : fs::directory_iterator(cache_root))
      {
        if(!fs::is_directory(cache_entry)) continue;

        std::string folder_name = mc_rtc::filename(cache_entry, true);
        if(folder_name.rfind(variant_name + "-", 0) == 0 && folder_name != target_folder_name)
        {
          fs::remove_all(cache_entry);
        }
      }

      fs::create_directories(convex_dir);

      for(const auto & col : _collision)
      {
        for(const auto & c : col.second)
        {
          if(c.geometry.type == rbd::parsers::Geometry::Type::MESH)
          {
            const auto & mesh = boost::get<rbd::parsers::Geometry::Mesh>(c.geometry.data);
            const auto & mesh_file = fs::path(mesh.filename);
            generate(mc_rtc::convertURI(mesh_file), convex_dir, meshScale(mesh), meshMirrorAxes(mesh));
          }
        }
      }
    }
  }
  else
  {
    mc_rtc::log::error("[RobotModule] name is empty, cannot generate convexes");
  }
}

void RobotModule::bind_convexes()
{
  for(const auto & b : mb.bodies())
  {
    const auto & collisions = _collision[b.name()];
    if(collisions.size() == 1)
    {
      if(collisions[0].geometry.type == rbd::parsers::Geometry::Type::MESH)
      {
        const auto & mesh = boost::get<rbd::parsers::Geometry::Mesh>(collisions[0].geometry.data);
        fs::path convex_path =
            fs::path(convex_dir) / (mc_rtc::filename(mesh.filename) + mirrorSuffix(meshMirrorAxes(mesh)) + "-ch.txt");

        if(!fs::exists(convex_path))
        {
          mc_rtc::log::error("Convex hull file does not exist: {}", convex_path.string());
          continue;
        }

        _convexHull[b.name()] = {b.name(), convex_path};
      }
      continue;
    }

    size_t added = 0;
    for(const auto & col : collisions)
    {
      if(col.geometry.type == rbd::parsers::Geometry::Type::MESH)
      {
        const auto & mesh = boost::get<rbd::parsers::Geometry::Mesh>(col.geometry.data);
        fs::path convex_path =
            fs::path(convex_dir) / (mc_rtc::filename(mesh.filename) + mirrorSuffix(meshMirrorAxes(mesh)) + "-ch.txt");

        if(!fs::exists(convex_path))
        {
          mc_rtc::log::error("Convex hull file does not exist: {}", convex_path.string());
          continue;
        }

        _convexHull[b.name() + "_" + std::to_string(added)] = {b.name(), convex_path};
        added++;
      }
    }
  }
}

RobotModule::Gripper::Gripper(const std::string & name, const std::vector<std::string> & joints, bool reverse_limits)
: Gripper(name, joints, reverse_limits, nullptr, nullptr)
{
}

RobotModule::Gripper::Gripper(const std::string & name,
                              const std::vector<std::string> & joints,
                              bool reverse_limits,
                              const Safety & safety)
: Gripper(name, joints, reverse_limits, &safety, nullptr)
{
}

RobotModule::Gripper::Gripper(const std::string & name,
                              const std::vector<std::string> & joints,
                              bool reverse_limits,
                              const Safety & safety,
                              const std::vector<Mimic> & mimics)
: Gripper(name, joints, reverse_limits, &safety, &mimics)
{
}

RobotModule::Gripper::Gripper(const std::string & name,
                              const std::vector<std::string> & joints,
                              bool reverse_limits,
                              const Safety * safety,
                              const std::vector<Mimic> * mimics)
: name(name), joints(joints), reverse_limits(reverse_limits), hasSafety_(safety != nullptr),
  hasMimics_(mimics != nullptr)
{
  if(mimics) { mimics_ = *mimics; }
  if(safety) { safety_ = *safety; }
}

void RobotModule::Gripper::Safety::load(const mc_rtc::Configuration & config)
{
  if(config.has("actualCommandDiffTrigger")) { actualCommandDiffTrigger = config("actualCommandDiffTrigger"); }
  if(config.has("threshold"))
  {
    mc_rtc::log::warning("[MC_RTC_DEPRECATED] Gripper safety property \"threshold\" (expressed in degrees) is "
                         "deprecated, please use \"actualCommandDiffTrigger\" (expressed in radian) instead");
    actualCommandDiffTrigger = mc_rtc::constants::toRad(config("threshold"));
  }

  if(config.has("overCommandLimitIterN"))
  {
    overCommandLimitIterN =
        std::max<unsigned int>(1, config("overCommandLimitIterN", DEFAULT_OVER_COMMAND_LIMIT_ITER_N));
  }
  else if(config.has("iter"))
  {
    mc_rtc::log::warning("[MC_RTC_DEPRECATED] Gripper safety property \"iter\" is deprecated, please use "
                         "\"overCommandLimitIterN\" instead");
    overCommandLimitIterN = std::max<unsigned int>(1, config("iter", DEFAULT_OVER_COMMAND_LIMIT_ITER_N));
  }

  if(config.has("releaseSafetyOffset")) { releaseSafetyOffset = config("releaseSafetyOffset"); }
  else if(config.has("release"))
  {
    mc_rtc::log::warning("[MC_RTC_DEPRECATED] Gripper safety property \"release\" (expressed in degrees) is "
                         "deprecated, please use \"releaseSafetyOffset\" instead (expressed in radian)");
    releaseSafetyOffset = mc_rtc::constants::toRad(config("release"));
  }

  if(config.has("percentVMax"))
  {
    percentVMax = mc_filter::utils::clamp(static_cast<double>(config("percentVMax")), 0, 1);
  }
  else if(config.has("percentVMAX"))
  {
    mc_rtc::log::warning("[MC_RTC_DEPRECATED] Gripper safety property \"percentVMAX\" is deprecated, please use "
                         "\"percentVMax\" instead");
    percentVMax = mc_filter::utils::clamp(static_cast<double>(config("percentVMAX")), 0, 1);
  }
}

mc_rtc::Configuration RobotModule::Gripper::Safety::save() const
{
  mc_rtc::Configuration config;
  config.add("actualCommandDiffTrigger", actualCommandDiffTrigger);
  config.add("overCommandLimitIterN", overCommandLimitIterN);
  config.add("releaseSafetyOffset", releaseSafetyOffset);
  config.add("percentVMax", percentVMax);
  return config;
}

void RobotModule::boundsFromURDF(const rbd::parsers::Limits & limits)
{
  _bounds = urdf_limits_to_bounds(limits);
}

void RobotModule::expand_stance()
{
  for(const auto & j : mb.joints())
  {
    if(!_stance.count(j.name()) && j.name() != "Root") { _stance[j.name()] = j.zeroParam(); }
  }
}

void RobotModule::make_default_ref_joint_order()
{
  _ref_joint_order.resize(0);
  for(const auto & j : mb.joints())
  {
    if(j.dof() >= 1 && j.type() != rbd::Joint::Free) { _ref_joint_order.push_back(j.name()); }
  }
}

RobotModule::bounds_t urdf_limits_to_bounds(const rbd::parsers::Limits & limits)
{
  RobotModule::bounds_t ret = {};
  ret.reserve(6);
  ret.push_back(limits.lower);
  ret.push_back(limits.upper);
  auto convert = [](const std::map<std::string, std::vector<double>> & l)
  {
    auto ret = l;
    for(auto & el : ret)
    {
      for(auto & e : el.second) { e = -e; }
    }
    return ret;
  };
  ret.push_back(convert(limits.velocity));
  ret.push_back(limits.velocity);
  ret.push_back(convert(limits.torque));
  ret.push_back(limits.torque);
  return ret;
}

bool check_module_compatibility(const RobotModule & lhs, const RobotModule & rhs)
{
  bool is_ok = true;
  auto format_params = [](const std::vector<std::string> & params) -> std::string
  {
    if(params.size() == 0) { return "[]"; }
    std::stringstream ss;
    ss << "[";
    for(size_t i = 0; i < params.size() - 1; ++i) { ss << params[i] << ", "; }
    ss << params.back() << "]";
    return ss.str();
  };
  auto show_incompatiblity_issue = [&](std::string_view reason)
  {
    if(is_ok)
    {
      is_ok = false;
      mc_rtc::log::critical("{} (params: {}) and {} (params: {}) are incompatible.\nReasons:", lhs.name,
                            format_params(lhs.parameters()), rhs.name, format_params(rhs.parameters()));
    }
    mc_rtc::log::critical("- {}", reason);
  };
  if(lhs.ref_joint_order() != rhs.ref_joint_order()) { show_incompatiblity_issue("Different reference joint order"); }
  if(lhs.bodySensors() != rhs.bodySensors()) { show_incompatiblity_issue("Different body sensors"); }
  if(lhs.forceSensors() != rhs.forceSensors()) { show_incompatiblity_issue("Different force sensors"); }
  if(lhs.jointSensors() != rhs.jointSensors()) { show_incompatiblity_issue("Different joint sensors"); }
  if(lhs.grippers() != rhs.grippers()) { show_incompatiblity_issue("Different grippers"); }
  if(lhs.devices().size() != rhs.devices().size()) { show_incompatiblity_issue("Different devices"); }
  else
  {
    const auto & lhs_devices = lhs.devices();
    const auto & rhs_devices = rhs.devices();
    for(size_t i = 0; i < lhs_devices.size(); ++i)
    {
      const auto & lhs_d = *lhs_devices[i];
      const auto & rhs_d = *rhs_devices[i];
      if(lhs_d.name() != rhs_d.name() || lhs_d.type() != rhs_d.type() || lhs_d.parent() != rhs_d.parent())
      {
        show_incompatiblity_issue(fmt::format("Different device: {} != {}", lhs_d.name(), rhs_d.name()));
      }
    }
  }
  return is_ok;
}

} // namespace mc_rbdyn
