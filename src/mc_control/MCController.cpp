/*
 * Copyright 2015-2019 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_control/MCController.h>

#include <mc_rbdyn/RobotLoader.h>
#include <mc_rbdyn/RobotModule.h>
#include <mc_rtc/constants.h>

#include <mc_rtc/config.h>
#include <mc_rtc/gui/Schema.h>
#include <mc_rtc/io_utils.h>
#include <mc_rtc/logging.h>

#include <mc_tasks/MetaTaskLoader.h>

#include <RBDyn/FK.h>
#include <RBDyn/FV.h>

#include <array>
#include <fstream>

namespace mc_control
{

MCController::MCController(std::shared_ptr<mc_rbdyn::RobotModule> robot, double dt) : MCController(robot, dt, {}) {}

MCController::MCController(std::shared_ptr<mc_rbdyn::RobotModule> robot,
                           double dt,
                           const mc_rtc::Configuration & config)
: MCController({robot, mc_rbdyn::RobotLoader::get_robot_module("env",
                                                               std::string(mc_rtc::MC_ENV_DESCRIPTION_PATH),
                                                               std::string("ground"))},
               dt,
               config)
{
}

MCController::MCController(const std::vector<std::shared_ptr<mc_rbdyn::RobotModule>> & robots_modules, double dt)
: MCController(robots_modules, dt, {})
{
}

MCController::MCController(const std::vector<std::shared_ptr<mc_rbdyn::RobotModule>> & robots_modules,
                           double dt,
                           const mc_rtc::Configuration & config)
: qpsolver(std::make_shared<mc_solver::QPSolver>(dt)),
  logger_(std::make_shared<mc_rtc::Logger>(mc_rtc::Logger::Policy::NON_THREADED, "", "")),
  gui_(std::make_shared<mc_rtc::gui::StateBuilder>()), config_(config), timeStep(dt)
{
  /* Load robots */
  qpsolver->logger(logger_);
  qpsolver->gui(gui_);
  for(auto rm : robots_modules)
  {
    loadRobot(rm, rm->name, robots());
    loadRobot(rm, rm->name, realRobots());
  }

  if(gui_)
  {
    gui_->addElement({"Global", "Add task"},
                     mc_rtc::gui::Schema("Add MetaTask", "MetaTask", [this](const mc_rtc::Configuration & config) {
                       try
                       {
                         auto t = mc_tasks::MetaTaskLoader::load(this->solver(), config);
                         this->solver().addTask(t);
                       }
                       catch(...)
                       {
                         mc_rtc::log::error("Failed to load MetaTask from request\n{}", config.dump(true));
                       }
                     }));
  }
  /* Initialize constraints and tasks */
  std::array<double, 3> damper = {0.1, 0.01, 0.5};
  contactConstraint = mc_solver::ContactConstraint(timeStep, mc_solver::ContactConstraint::Velocity);
  dynamicsConstraint = mc_solver::DynamicsConstraint(robots(), 0, timeStep, damper, 0.5);
  kinematicsConstraint = mc_solver::KinematicsConstraint(robots(), 0, timeStep, damper, 0.5);
  selfCollisionConstraint = mc_solver::CollisionsConstraint(robots(), 0, 0, timeStep);
  selfCollisionConstraint.addCollisions(solver(), robots_modules[0]->minimalSelfCollisions());
  compoundJointConstraint.reset(new mc_solver::CompoundJointConstraint(robots(), 0, timeStep));
  postureTask = std::make_shared<mc_tasks::PostureTask>(solver(), 0, 10.0, 5.0);
  mc_rtc::log::info("MCController(base) ready");
}

MCController::~MCController() {}

mc_rbdyn::Robot & MCController::loadRobot(mc_rbdyn::RobotModulePtr rm, const std::string & name)
{
  return loadRobot(rm, name, robots());
}

mc_rbdyn::Robot & MCController::loadRobot(mc_rbdyn::RobotModulePtr rm,
                                          const std::string & name,
                                          mc_rbdyn::Robots & robots)
{
  assert(rm);
  auto & r = robots.load(*rm);
  r.name(name);
  r.mbc().gravity = mc_rtc::constants::gravity;
  r.forwardKinematics();
  r.forwardVelocity();
  if(gui_)
  {
    auto data = gui_->data();
    if(!data.has("robots"))
    {
      data.array("robots");
    }
    if(!data.has("bodies"))
    {
      data.add("bodies");
    }
    if(!data.has("surfaces"))
    {
      data.add("surfaces");
    }
    data("robots").push(r.name());
    auto bs = data("bodies").array(r.name());
    for(const auto & b : r.mb().bodies())
    {
      bs.push(b.name());
    }
    data("surfaces").add(r.name(), r.availableSurfaces());
  }
  solver().updateNrVars();
  return r;
}

void MCController::removeRobot(const std::string & name)
{
  robots().removeRobot(name);
  solver().updateNrVars();
}

bool MCController::resetObservers()
{
  auto pipelineDesc = std::string{};

  for(size_t i = 0; i < pipelineObservers_.size(); ++i)
  {
    const auto & observerPair = pipelineObservers_[i];
    auto observer = observerPair.first;
    bool updateRobots = observerPair.second;
    observer->reset(*this);

    if(updateRobots)
    {
      observer->updateRobots(*this, realRobots());
      pipelineDesc += observer->desc();
    }
    else
    {
      pipelineDesc += "[" + observer->desc() + "]";
    }

    if(i < pipelineObservers_.size() - 1)
    {
      pipelineDesc += " -> ";
    }

    observer->addToLogger(*this, logger());
    if(gui_)
    {
      observer->addToGUI(*this, *gui_);
    }
  }
  if(!pipelineObservers_.empty())
  {
    mc_rtc::log::success("Observers: {}", pipelineDesc);
  }
  return true;
}

bool MCController::runObservers()
{
  for(const auto & observerPair : pipelineObservers_)
  {
    auto observer = observerPair.first;
    bool updateRobots = observerPair.second;
    bool r = observer->run(*this);
    if(!r)
    {
      mc_rtc::log::error("Observer {} failed to run", observer->name());
      return false;
    }
    if(updateRobots)
    {
      observer->updateRobots(*this, realRobots());
    }
  }
  return true;
}

bool MCController::run()
{
  return run(mc_solver::FeedbackType::None);
}

bool MCController::run(mc_solver::FeedbackType fType)
{
  if(!qpsolver->run(fType))
  {
    mc_rtc::log::error("QP failed to run()");
    return false;
  }
  qpsolver->fillTorque(dynamicsConstraint);
  return true;
}

const mc_solver::QPResultMsg & MCController::send(const double & t)
{
  return qpsolver->send(t);
}

void MCController::reset(const ControllerResetData & reset_data)
{
  std::vector<std::string> supported;
  supported_robots(supported);
  if(supported.size() && std::find(supported.cbegin(), supported.cend(), robot().name()) == supported.end())
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[MCController] The main robot {} is not supported by this controller. Supported robots are: [{}]",
        robot().name(), mc_rtc::io::to_string(supported));
  }

  robot().mbc().zero(robot().mb());
  robot().mbc().q = reset_data.q;
  postureTask->posture(reset_data.q);
  robot().forwardKinematics();
  robot().forwardVelocity();
}

const mc_rbdyn::Robot & MCController::robot() const
{
  return qpsolver->robot();
}

const mc_rbdyn::Robot & MCController::env() const
{
  return qpsolver->env();
}

mc_rbdyn::Robot & MCController::robot()
{
  return qpsolver->robot();
}

mc_rbdyn::Robot & MCController::env()
{
  return qpsolver->env();
}

const mc_rbdyn::Robots & MCController::robots() const
{
  return qpsolver->robots();
}

mc_rbdyn::Robots & MCController::robots()
{
  return qpsolver->robots();
}

const mc_solver::QPSolver & MCController::solver() const
{
  assert(qpsolver);
  return *qpsolver;
}

mc_solver::QPSolver & MCController::solver()
{
  assert(qpsolver);
  return *qpsolver;
}

mc_rtc::Logger & MCController::logger()
{
  return *logger_;
}

void MCController::supported_robots(std::vector<std::string> & out) const
{
  out = {};
}

const mc_rbdyn::Robots & MCController::realRobots() const
{
  return solver().realRobots();
}

mc_rbdyn::Robots & MCController::realRobots()
{
  return solver().realRobots();
}

const mc_rbdyn::Robot & MCController::realRobot() const
{
  return realRobots().robot();
}

mc_rbdyn::Robot & MCController::realRobot()
{
  return realRobots().robot();
}

const sva::PTransformd & MCController::anchorFrame() const
{
  return anchorFrame_;
}

void MCController::anchorFrame(const sva::PTransformd & anchor)
{
  anchorFrame_ = anchor;
}

const sva::PTransformd & MCController::anchorFrameReal() const
{
  return anchorFrameReal_;
}

void MCController::anchorFrameReal(const sva::PTransformd & anchor)
{
  anchorFrameReal_ = anchor;
}

void MCController::stop() {}

Gripper & MCController::gripper(const std::string & robot, const std::string & gripper)
{
  return robots().robot(robot).gripper(gripper);
}

} // namespace mc_control
