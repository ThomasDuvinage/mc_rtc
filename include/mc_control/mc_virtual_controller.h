#ifndef _H_MCVIRTUALCONTROLLER_H_
#define _H_MCVIRTUALCONTROLLER_H_

#include <mc_rbdyn/robot.h>
#include <mc_control/mc_solver/qpsolver.h>
#include <mc_control/generic_gripper.h>

#include <mc_robots/hrp2_drc.h>
#include <mc_robots/env.h>

#include <Tasks/QPTasks.h>
#include <mc_tasks/EndEffectorTask.h>

namespace mc_control
{

/* This structure will be filled in time with the necessary information */
/* Coming most likely from the previous controller */
struct ControllerResetData
{
  const std::vector< std::vector<double> > & q;
  const std::vector<mc_rbdyn::Contact> & contacts;
};

struct MCDRCGlobalController;

/*FIXME Get some data as parameters (e.g. timeStep, path to default env...) */
struct MCVirtualController
{
  friend struct MCDRCGlobalController;
public:
  MCVirtualController();

  virtual bool run() = 0;

  virtual const mc_control::QPResultMsg & send(const double & t) = 0;

  virtual void reset(const ControllerResetData & reset_data) = 0;

  virtual void setWrenches(const std::vector< std::pair<Eigen::Vector3d, Eigen::Vector3d> > & wrenches) = 0;
  /* Helper function to access robots, robot and env */
  virtual const mc_rbdyn::Robot & robot() const = 0;

  virtual const mc_rbdyn::Robot & env() const = 0;

  virtual const mc_rbdyn::Robots & robots() const = 0;

  virtual mc_rbdyn::Robots & robots() = 0;

  virtual mc_rbdyn::Robot & robot() = 0;

  virtual mc_rbdyn::Robot & env() = 0;
public:
  /* Common stuff */
  const double timeStep;
  /* Encoder values provided by the low-level controller */
  std::vector<double> encoderValues;
  /* FIXME A bit dirty but needed */
  std::shared_ptr<mc_control::Gripper> lgripper;
  std::shared_ptr<mc_control::Gripper> rgripper;
};

}

#endif