# Ros2-jazzy-DDR-Simulator
This is repository contains the necesary code for run a DDR simulatir. The Development is in progress
To run AMCL + pathfinding methods use:
-Gazebo Headless
```
ros2 launch echo_bringup gz_spawn.launch.py world:=src/echo_gazebo/worlds/walls_world2.sdf gz_mode:=False
```
-EKF 
```
ros2 launch echo_ekf ekf.launch.py
```

-AMCL
```
ros2 launch echo_bringup amcl_localization.launch.py map:=src/echo_gazebo/maps/map2.yaml
```
-Rviz2 with markers 
```
rviz2 -d src/echo_gazebo/worlds/bitstar.rviz 
```

-BIT* Pathfinding
```
ros2 run echo_path_planner bitstar_planner_node 
```
