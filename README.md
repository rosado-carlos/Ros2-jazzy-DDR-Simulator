
# ROS2 Jazzy DDR Simulator

This repository contains the necessary packages to simulate a Differential Drive Robot (DDR) using ROS 2 Jazzy and Gazebo.

The project is under active development and aims to provide a modular and structured simulation environment for learning and experimentation in mobile robotics.

## Overview

The simulator includes:

* A robot model described in URDF/Xacro
* A Gazebo simulation environment
* Integration of LiDAR, RGB camera, depth camera, and IMU sensors
* ROS 2 Control configuration
* Reactive navigation methods
* Twist multiplexing for velocity command management
* RViz configuration for visualization

## Repository Structure

```
.
├── src/
│   ├── echo_bringup/
│   ├── echo_control/
│   ├── echo_description/
│   ├── echo_gazebo/
│   ├── echo_wall_following/
│   └── echo_gap_following/
├── README.md
└── .gitignore
```

## Requirements

* Ubuntu 24.04
* ROS 2 Jazzy
* Gazebo (compatible with Jazzy)
* colcon build tools

Refer to the official ROS 2 documentation for installation and environment setup instructions.

## Build Instructions

Clone the repository inside a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <repository_url>
cd ..
colcon build --symlink-install
source install/setup.bash
```

## Running the Simulation

Launch the simulation:

```bash
ros2 launch echo_bringup gz_spawn.launch.py
```

Run reactive navigation methods:

Wall following:

```bash
ros2 launch echo_wall_following wall_following.launch.py
```

Gap following:

```bash
ros2 launch echo_gap_following gap_following.launch.py
```

## Development Status

The project currently provides a working simulation of a differential drive robot in ROS 2 Jazzy using Gazebo. The robot model includes multiple onboard sensors such as LiDAR, RGB camera, depth camera, and IMU.

Basic control functionality and reactive navigation behaviors have been implemented, including an emergency braking node (AEB) and velocity command multiplexing. The system is organized into separate packages to maintain a clear structure and allow future extensions.

Ongoing work focuses on improving sensor fusion, state estimation, and integrating SLAM capabilities.
