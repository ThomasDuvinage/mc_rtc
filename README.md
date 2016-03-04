Introduction
==

`mc_rtc` is a super-set of SpaceVecAlg/RBDyn/Tasks libraries which provides a friendlier/easier interface for the user. It was originally a C++ portage of the [mc_ros](https://gite.lirmm.fr/multi-contact/mc_ros) package aimed at running the controller on the embedded computer of the robot, as such, readers familiar with the Python interface proposed in `mc_ros` should easily find their way around `mc_rtc`.

This document assumes some familiarity with the [SpaceVecAlg](https://github.com/jorisv/SpaceVecAlg), [RBDyn](https://github.com/jorisv/RBDyn) and [Tasks](https://github.com/jorisv/Tasks) although the libraries in the `mc_rtc` package abstract most of these libraries inner-workings. Unfamiliar users may refer to the [SpaceVecAlg and RBDyn tutorials](https://github.com/jorisv/sva_rbdyn_tutorials) and the [SpaceVecAlg and RBDyn presentations](https://github.com/jorisv/sva_rbdyn_presentation) authored by Joris Vaillant.

Questions specific to the `mc_rtc` package can be adressed to <pierre.gergondet@gmail.com>. Some frequently asked questions are answered in the last section of this document.

Building
==

The following packages are required to build mc_rtc:
- `mc_non_ros` (see the [installation wiki](https://gite.lirmm.fr/multi-contact/installation_wiki/wikis/home) for more information)
- [mc_rbdyn_urdf_cpp](https://gite.lirmm.fr/multi-contact/mc_rbdyn_urdf_cpp)
- [jsoncpp](https://github.com/open-source-parsers/jsoncpp) (or `libjsoncpp-dev` on debian-like systems)
- [geos](https://github.com/libgeos/libgeos) (or `libgeos++-dev` on debian-like systems)o

The following packages are not required but bring additionnal features:
- [ROS](http://www.ros.org/)
> If `roscpp` is available during build then `tf2_ros` and `sensor_msgs` are also required. This will add some integration between the `ROS` framework and `mc_rtc` (e.g. robot's state publication) and allow controllers to use ROS functionnalities (provides a `ros::NodeHandle` instance that can be used anywhere in `mc_rtc`).
- [hrpsys-base](https://github.com/fkanehiro/hrpsys-base)
> If `hrpsys-base` is available during build then the RTC component `MCControl` will be built. This is necessary to use `mc_rtc` in the `choreonoid` simulator and on the robot.

Work in progress (last update 2015/12/08)
==

- python bindings using cython ([current progress](https://gite.lirmm.fr/multi-contact/mc_cython)) aimed at replacing most of `mc_ros`
- unify remote interactions with the controller(s)

Library by library overview
==

While `mc_rtc` is a single package, it consists of 7 independent libraries. We give an overview of each library and their role here.

mc_rbdyn
--

A super-set of `SpaceVecAlg` and `RBDyn`. Essentially turns collections of low-level types into `Robot`, `Surface`, `Contact`, etc...
 Refer to the headers/doxygen documentation for available class/methods.

### A note on the `Robot` class

`Tasks` is designed to operate with a multi-robot approach in mind (e.g. a door may be seen as a robot with 2 degrees of freedom) and as such most classes in `mc_rtc` will operate on the `Robots` class rather than the `Robot` class to "talk" with `Tasks`. In order to maintain a coherent state, this creates a tight coupling between the `Robot` and `Robots` classes. Therefore, the `Robot` class cannot exist by itself and only the `Robots` class should be stored.

mc_robots
--

Implementations of the `mc_rbdyn::RobotModule`, in particular:
- `EnvRobotModule` is able to load any non-interactive robot (i.e. a static environment) that follows the `mc_env_description` scheme
- `HRP2DRCRobotModule` and `HRP2DRCGripperRobotModule` provide a HRP2-kai robot (without/with grippers in the model respectively)

These implementations are provided through a library. Examples are provided in `src/mc_robots`

In `mc_rtc`, one can access those robots' modules by calling `mc_rbdyn::RobotLoader::get_robot_module("RobotName")`.

mc_solver
--

Provides a high-level interface above `tasks::qp::QPSolver` and `Tasks` constraints that can be used with `mc_rbdyn` types. See the headers/doxygen documentation for available class/methods.

mc_tasks
--

A mixture of:
- High-level interface above some `Tasks` tasks that can be used with `mc_rbdyn` types
- Higher-level tasks (e.g. `AddRemoveContactTask`)

See the headers/doxygen documentation for available class/methods.

mc_trajectory (experimental)
--

Set of functions/class to generate trajectories

mc_control
--

The basis for new controller. More details regarding the implementation of a new controller are provided below but all controllers should inherit from the `MCController` class.

Some controllers are provided in `mc_rtc` to serve as examples/basis for your own:
- `PostureController` controls a robot's posture
- `Body6dController` controls a robot's end-effector(s)
- `CoMController` controls a robot's CoM
- `SeqController` implements a FSM to play a sequence of stance/actions obtained from planning softwares

The `MCGlobalController` class is used to manage the controllers and transfer the information from the simulation/robot (e.g. force sensors' data or joints positions) to the current controller. It also decides what will be the "main" robot for the simulation/run.

mc_rtc_ros
--

If `mc_rtc` was compiled with `ROS` support then this library is used by `MCGlobalController` to communicate with `ROS`. It can also be invoked to retrieve a `ros::NodeHandle` instance in any class of `mc_rtc` that may require it for ROS functionnalities.


Implementing a new robot module
==

### A note on environments

To import a new non-interactive robot (i.e. an environment) it is not necessary to define a new robot inside or outisde mc_rtc as the `mc_robots::EnvRobotModule` should work out of the box given the proper environment path and name as long as the environment definition follows the `mc_env_description` scheme.

Inside mc_rtc
--

1. Write a new class that inherits from `mc_rbdyn::RobotModule` (e.g. `include/mc_robots/MyRobot.h` and `src/mc_robots/MyRobot.cpp`)
2. Implement the virtual functions your robot requires
3. In your header file, invoke the `ROBOT_MODULE_DEFAULT_CONSTRUCTOR` or `ROBOT_MODULE_CANONIC_CONSTRUCTOR` macro (e.g. `ROBOT_MODULE_DEFAULT_CONSTRUCTOR("MyRobot", mc_robot::MyRobot)`). The first is adapted for classes which can be instantiated by a default constructor. The second is adapted for classes which should be instantiated by two `std::string` values. For more options, check `include/mc_rbdyn/RobotModule.h` for the full definition of the macros
4. Add the entry `add_robot(MyRobot)` in `src/CMakeLists.txt`
5. Your robot can now be used in `mc_rtc` by calling `mc_rbdyn::RobotLoader::get_robot_module("MyRobot")`

Outisde mc_rtc
--

The steps are similar to the previous option but do not require to write code inside `mc_rtc`. No sample project is available at the moment.

Implementing a new controller
==

Inside mc_rtc
--

1. Inherit from `mc_control::MCController`. Note that this controller already defines:
2. Add the tasks/constraints you need, implement the functions you require, implement the services you provide
3. In your header file, invoke the `CONTROLLER_CONSTRUCTOR` or `SIMPLE_CONTROLLER_CONSTRUCTOR`. The first will construct your controller using the timestep `dt`, a pointer to the main robot module and the `Json::Value` obtained from the parsing of the `MCGlobalController` instance. The second macro will not pass you the `Json::Value`
4. Add the entry `add_controller(MyController ${SOURCE_FILES})` in `src/CMakeLists.txt`
5. Your controller can be enable by adding `MyController` to the `Enabled`/`Default` entry of `mc_rtc.conf`

### Functions to implement

A quick description of the `virtual` functions in `MCController` and what they are supposed to do.

#### `virtual void reset(const ControllerResetData & reset_data)`

Called when:
1. the control loop starts
2. `MCGlobalController` switches from a different controller

ATM, the `ControllerResetData` holds the following data: (depending on the moment of the invokation)
1. "True" state of the robot (encoders)
2. Controlled state of the robot forwarded by the previous controller

#### `virtual bool run()`

Called at every iteration of the control loop. Runs in the real-time context on the robot. Returns `false` to indicate a failure, `true` otherwise.

#### `virtual const QPResultMsg & send(const double & t)`

Returns the current control output. Does not run if `run()` invokation failed. The default implementation should be fine generally.


#### `bool read_msg(std::string & msg)` and `bool read_write_msg(std::string & msg, std::string & out)`

These functions are used to provide a generic service interface. In both functions, `msg` holds the content of the request made by the caller. In the `read_write_msg`, `out` may be filled with an answer for the caller. Both functions should return `false` to indicate that the controller was not able to handle the request `msg` and `true` otherwise.

Outisde mc_rtc
--

Similar to the previous method but steps 4 and 5 are obviously unnecessary. A sample project is available [here](https://gite.lirmm.fr/multi-contact/mc_sample_controller).

A note on real-time
--

When writting a controller that will ultimately run as part of the real-time loop on the robot one has to be careful about potential performance bottleneck as missing real-time iterations will result in non-smooth behaviours. In particular, **everything that happens in the `run` method is happenning in the real-time loop** and thus one should avoid:
- io-blocking function (in particular network-related, light-weight disk operation can be performed)
- threading (any thread should be launched prior to entering the real-time context)


Remote interaction with the controller(s)
==

Using MCControl RTC component
--

1. Implement a new function in the idl and its appropriate counter-part in `MCControllerService' that relies on `MCGlobalController` to call the appropriate function on the controller (as tedious as it sounds)
2. Implement `send_msg` or `send_recv_msg` depending on your need and perform the appropriate command parsing (e.g. `MCBCISelfInteract`)

Using mc_vrep
--

See mc_vrep for implementation details. mc_vrep already mimics the `send_msg` and `send_recv_msg` functionnality in the CLI.


FAQ
==

These are not the questions you're looking for.