#ifndef _H_STANCECONFIG_H_
#define _H_STANCECONFIG_H_

/* This struct holds the configuration for stances in a seq plan */

/* TODO Fill-it up */
/* TODO Build a tool to import from python */
/* TODO Build a tool to read from configuration file */

#include <Eigen/Core>
#include <SpaceVecAlg/SpaceVecAlg>
#include <functional>
#include <map>

namespace mc_rbdyn
{

std::function<Eigen::Vector3d (const sva::PTransformd &, const sva::PTransformd &, const Eigen::Vector3d &)>
percentWaypoint(double x, double y, double z, double nOff);

std::function<Eigen::Vector3d (const sva::PTransformd &, const sva::PTransformd &, const Eigen::Vector3d &)>
hardCodedPos(double x, double y, double z);

struct StanceConfig
{
public:
  struct CoMTask
  {
    double stiffness;
    double extraStiffness;
    double weight;
    double targetSpeed;
  };
  struct CoMObj
  {
    double posThresh;
    double velThresh;
    Eigen::Vector3d comOffset;
  };
  struct PostureTask
  {
    double stiffness;
    double weight;
  };
  struct Position
  {
    double stiffness;
    double extraStiffness;
    double weight;
    double targetSpeed;
  };
  struct Orientation
  {
    double stiffness;
    double weight;
    double finalWeight;
  };
  struct LinVel
  {
    double stiffness;
    double weight;
    double speed;
  };
  struct WaypointConf
  {
    double thresh;
    std::function<Eigen::Vector3d (const sva::PTransformd &, const sva::PTransformd &, const Eigen::Vector3d &)> pos;
  };
  struct CollisionConf
  {
    double iDist;
    double sDist;
    double damping;
  };
  struct ContactTask
  {
    Position position;
    Orientation orientation;
    LinVel linVel;
    WaypointConf waypointConf;
    CollisionConf collisionConf;
  };
  struct ContactObj
  {
    double posThresh;
    double velThresh;
    double adjustPosThresh;
    double adjustVelThresh;
    double adjustOriThresh;
    Eigen::Vector3d adjustOffset;
    Eigen::Vector3d adjustOriTBNWeight;
    double preContactDist;
  };
  struct BodiesCollisionConf
  {
    std::string body1;
    std::string body2;
    CollisionConf collisionConf;
  };
  struct Collisions
  {
    std::vector<BodiesCollisionConf> autoc;
    std::vector<BodiesCollisionConf> robotEnv;
    std::map< std::pair<std::string, std::string>, std::vector< std::pair<std::string, std::string> > > robotEnvContactFilter;
  };
public:
  StanceConfig();
public:
  CoMTask comTask;
  CoMObj comObj;
  PostureTask postureTask;
  ContactTask contactTask;
  ContactObj contactObj;
  Collisions collisions;
};

}

#endif