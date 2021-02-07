#ifndef _POINTCLOUD_FRUSTUM_CUDA
#define _POINTCLOUD_FRUSTUM_CUDA
#include <Eigen/Dense>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <sensor_msgs/PointCloud2.h>


uint32_t removePointsInsideView(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out);

struct Point {
    float x;
    float y;
    float z;
    float _pad0;
    float rgb;
    float _pad_1;
    float _pad_2;
    float _pad_3;

    __host__ __device__
    inline Point(){};

    __host__ __device__
    inline Point(float x_, float y_, float z_, float rgb_) 
        : x(x_), y(y_), z(z_), rgb(rgb_) {};

};


#endif