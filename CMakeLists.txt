cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_benchmark)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  sensor_msgs
  pcl_ros
  pcl_conversions
  rosbag
)

find_package(CUDA REQUIRED)

catkin_python_setup()

find_package(PCL 1.7.1 REQUIRED)

### CUDA ###
set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
set (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC) 
cuda_add_library(PointCloudCuda ${LIB_TYPE} src/pointcloud_cuda.cu)
cuda_add_library(FrustumCuda ${LIB_TYPE} src/frustum_cuda.cu)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS message_runtime
)


# PCL Experiment
add_executable(experiment_pcl src/experiment_pcl.cpp)

target_link_libraries(experiment_pcl 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
)

add_dependencies(experiment_pcl ${catkin_EXPORTED_TARGETS})


# CUDA Experiment

add_executable(experiment_cuda src/experiment_cuda.cpp)
add_dependencies(experiment_cuda PointCloudCuda)

target_link_libraries(experiment_cuda 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
  PointCloudCuda
)

# Frustum Experiment

add_executable(experiment_frustum_cuda src/experiment_frustum.cpp)
add_dependencies(experiment_frustum_cuda FrustumCuda)

target_link_libraries(experiment_frustum_cuda 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
  FrustumCuda
)


# Open3D Experiment
catkin_install_python(PROGRAMS script/experiment_open3d
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Includes
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
