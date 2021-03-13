#include "ros/ros.h"

#include <sstream>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include "voxel_grid.cuh"

// System include
#include <iostream>
#include <string>
#include <cstdarg>
#include <iostream>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH
#include "lz4.h"     // This is all that is required to expose the prototypes for basic compression and decompression.


void benchmark_one(sensor_msgs::PointCloud2::ConstPtr cloud) {
    sensor_msgs::PointCloud2Ptr point_cloud(new sensor_msgs::PointCloud2);

    const size_t point_count = cloud->height * cloud->width;

    // Setup PointCloud
    point_cloud->height = 1;
    point_cloud->is_dense = false;
    point_cloud->is_bigendian = false;
    point_cloud->header.frame_id = "calib_board";
    point_cloud->header.stamp = ros::Time::now();

    // Setup PointCloud if not yet done
    sensor_msgs::PointCloud2Modifier pcd_modifier(*point_cloud);
    if(point_cloud->fields.size() == 0) {
        pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        pcd_modifier.resize(point_count);    
    }

    void* temp_pc = malloc(point_cloud->data.size());
    uint32_t voxelized_num = transformCropAndVoxelize(cloud, (float*) temp_pc);
    pcd_modifier.resize(voxelized_num);

    const int compressed_data_size = LZ4_compress_default((const char*) temp_pc, (char*) point_cloud->data.data(), sizeof(Point) * voxelized_num, point_cloud->data.size());
    point_cloud->data.resize(compressed_data_size);

    free(temp_pc);

    printf("Before lz4 %d After lz4 %d \n", voxelized_num * sizeof(Point), compressed_data_size); 

}

void benchmark(std::string bag_path, std::string topic) {
    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    
    foreach(rosbag::MessageInstance const m, view) {
        sensor_msgs::PointCloud2::ConstPtr s = m.instantiate<sensor_msgs::PointCloud2>();
        if (s == NULL) continue;

        ros::WallTime start_, end_;
        start_ = ros::WallTime::now();
        benchmark_one(s);
        end_ = ros::WallTime::now();
        double execution_time = (end_ - start_).toNSec() * 1e-6;

        ROS_INFO_STREAM("Exectution time (ms): " << execution_time);
    }

    bag.close();
}

int main(int argc, char** argv)
{ 
  // Start the ros node
  ros::init(argc, argv, "pc_merger");

  ros::NodeHandle n;
  ros::NodeHandle nh("~");

  std::string bag_path;
  std::string topic;
  nh.getParam("bag_path", bag_path);
  nh.getParam("topic", topic);

  benchmark(bag_path, topic);

  return 0;
}
