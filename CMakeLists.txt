cmake_minimum_required(VERSION 2.8.3)
enable_language(CXX)

project(pcl_registration)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
#set (CMAKE_CXX_STANDARD 14)
#
######################################
# CUDA SETUP 
######################################
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/pcl_registration/cmake" ${CMAKE_MODULE_PATH})
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)
#list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE} -rdc=true)
set(CUDA_NVCC_FLAGS --gpu-architecture=compute_75 --gpu-code=sm_75 -rdc=true)
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC)
set(CUDA_SEPARABLE_COMPILATION ON)

set(GLFW_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
CUDA_ADD_LIBRARY(IcpCudaLib ${LIB_TYPE} 
src/cuda/octree.cu
src/cuda/kdtree.cu
src/cuda/pointcloud.cu
src/cuda/scanmatch.cu
src/cuda/glslUtility.cpp
src/cuda/utilityCore.cpp
include/cudaMat4.hpp
include/glslUtility.hpp 
include/octree.h 
include/kdtree.h
include/pointcloud.h 
include/scanmatch.h 
include/svd3.h 
include/utilityCore.hpp
include/pcl_register_CUDA.hpp)
set_target_properties(IcpCudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
## System dependencies are found with CMake's conventions


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  stereo_msgs
  geometry_msgs
  genmsg
  tf tf2 tf2_bullet tf2_ros tf2_geometry_msgs tf2_kdl tf2_msgs tf_conversions
  image_transport
  cv_bridge
  sensor_msgs
  message_generation
  pcl_conversions
  pcl_ros
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost 1.65.1 REQUIRED)




## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
#catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS include
   LIBRARIES
   CATKIN_DEPENDS roscpp rospy std_msgs tf tf2 tf2_bullet tf2_ros tf2_geometry_msgs tf2_kdl tf2_msgs cv_bridge sensor_msgs image_transport  message_runtime 
#  DEPENDS system_lib
)

###########
## Build ##
###########
#if(CATKIN_ENABLE_TESTING)

#endif()
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GLEW_INCLUDE_DIR}
  ${GLFW_INCLUDE_DIRS}
  
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
## Declare a C++ library

add_executable(pcl_register_CUDA src/pcl_register_CUDA.cpp)
add_dependencies(pcl_register_CUDA IcpCudaLib)
target_link_libraries(pcl_register_CUDA ${catkin_LIBRARIES} ${PCL_LIBRARIES} IcpCudaLib ${GLEW_LIBRARY} ${GLFW_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(pcl_register_CUDA_alternative src/pcl_register_CUDA_alternative.cpp)
add_dependencies(pcl_register_CUDA_alternative IcpCudaLib)
target_link_libraries(pcl_register_CUDA_alternative ${catkin_LIBRARIES} ${PCL_LIBRARIES} IcpCudaLib ${GLEW_LIBRARY} ${GLFW_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(pcl_register_CUDA_offline src/pcl_register_CUDA_offline.cpp)
add_dependencies(pcl_register_CUDA_offline IcpCudaLib)
target_link_libraries(pcl_register_CUDA_offline ${catkin_LIBRARIES} ${PCL_LIBRARIES} IcpCudaLib ${GLEW_LIBRARY} ${GLFW_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(pcl_register_CPU src/pcl_register_CPU.cpp)
target_link_libraries(pcl_register_CPU ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(pcl_register_SIFT src/pcl_register_SIFT.cpp)
target_link_libraries(pcl_register_SIFT ${catkin_LIBRARIES} ${PCL_LIBRARIES})


#add_executable(teaser_registration_cpp src/pcl/teaser_registration.cpp)
																						#target_link_libraries(teaser_registration_cpp ${catkin_LIBRARIES} ${TeaserPP_LIBS} ${PCL_LIBRARIES})

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/priority_control_project_node.cpp)




## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )


#############
## Install ##
#############
install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)
# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_priority_control_project.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
