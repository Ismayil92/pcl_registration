cmake_minimum_required(VERSION 2.8.3)
enable_language(CXX)

project(priority_control_project)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  stereo_msgs
  geometry_msgs
  genmsg
  tf tf2 tf2_bullet tf2_ros tf2_geometry_msgs tf2_kdl tf2_msgs
  sensor_msgs
  message_generation
  pcl_conversions
  pcl_ros
)
find_package(Eigen3 REQUIRED)
find_package(rostest REQUIRED)
find_package(PCL REQUIRED)

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES
   CATKIN_DEPENDS roscpp rospy std_msgs tf tf2 tf2_bullet tf2_ros tf2_geometry_msgs tf2_kdl tf2_msgs sensor_msgs
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


## Declare a C++ library
add_executable(pcl_saver src/pcl/pcl_io.cpp)
target_link_libraries(pcl_saver ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pcl_alignment src/pcl/pcl_register_template_alignment.cpp)
target_link_libraries(pcl_alignment ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pcl_visualizer src/pcl/pcl_visualizer.cpp)
target_link_libraries(pcl_visualizer ${catkin_LIBRARIES} ${PCL_LIBRARIES})
