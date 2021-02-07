#include "ros/ros.h"

#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_ros/transforms.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/PointCloud2.h>


// System include
#include <iostream>
#include <string>
#include <cstdarg>
#include <iostream>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

void benchmark_one(sensor_msgs::PointCloud2::ConstPtr cloud) {
    PointCloud::Ptr pc_start(new PointCloud);
    PointCloud::Ptr pc_transformed(new PointCloud());
    PointCloud::Ptr pc_boxed(new PointCloud());
    PointCloud::Ptr pc_voxeled(new PointCloud());

    // Start converting to PCL
    pcl::fromROSMsg(*cloud, *pc_start);

    Eigen::Matrix4f tf;
    tf << 0, 1, 0, 0,
          1, 0, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1;

    // Transforming a bit
    pcl::transformPointCloud(*pc_start, *pc_transformed, tf);

    // Cropping
    pcl::CropBox<pcl::PointXYZRGB> box_filter;
    box_filter.setMin(Eigen::Vector4f(-4, -4, -4, 1.0));
    box_filter.setMax(Eigen::Vector4f(4, 4, 4, 1.0));
    box_filter.setInputCloud(pc_transformed);
    box_filter.filter(*pc_boxed);

    // Voxel Filtering
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud (pc_boxed);
    voxel_filter.setLeafSize (0.005f, 0.005f, 0.005f);
    // voxel_filter.setFilterLimits(-5, 5);
    // voxel_filter.setMinimumPointsNumberPerVoxel(2);
    voxel_filter.filter (*pc_voxeled);

    sensor_msgs::PointCloud2 out_cloud;
    pcl::toROSMsg(*pc_voxeled.get(), out_cloud);
    
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
