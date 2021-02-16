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


void benchmark_one(sensor_msgs::PointCloud2::ConstPtr cloud, ros::Publisher& pc_pub) {
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

    uint32_t voxelized_num = transformCropAndVoxelizeCenter(cloud, (float*) point_cloud->data.data());
    pcd_modifier.resize(voxelized_num);

    
    pc_pub.publish(point_cloud);

}

void benchmark(std::string bag_path, std::string topic) {
    ros::NodeHandle n;
    ros::Publisher pc_pub = n.advertise<sensor_msgs::PointCloud2>("pc", 5);

    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    
    // for(int i = 0; i < 1000; i++)
    foreach(rosbag::MessageInstance const m, view) {
        sensor_msgs::PointCloud2::ConstPtr s = m.instantiate<sensor_msgs::PointCloud2>();
        if (s == NULL) continue;

        ros::WallTime start_, end_;
        start_ = ros::WallTime::now();
        benchmark_one(s, pc_pub);
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
