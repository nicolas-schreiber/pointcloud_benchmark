#ifndef _POINTCLOUD_CUDA
#define _POINTCLOUD_CUDA
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



struct __attribute__((__packed__)) i16Point {
    int16_t x;
    int16_t y;
    int16_t z;
};

struct __attribute__((__packed__)) ui8RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

struct __attribute__((__packed__)) fXYZ {
    float x;
    float y;
    float z;

    __host__ __device__
    inline fXYZ(){};

    __host__ __device__
    inline fXYZ(float x_, float y_, float z_) 
        : x(x_), y(y_), z(z_) {};
};

struct __attribute__((__packed__)) sXYZ {
    size_t x;
    size_t y;
    size_t z;

    __host__ __device__
    inline sXYZ(){};

    __host__ __device__
    inline sXYZ(size_t x_, size_t y_, size_t z_) 
        : x(x_), y(y_), z(z_) {};
};


uint32_t transformCropAndVoxelize(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out);

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

struct Voxel {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;

    __host__ __device__
    inline Voxel() {};

    // Convert from Point
    __host__ __device__
    inline Voxel(Point p) {
        x = p.x; 
        y = p.y; 
        z = p.z; 
        
        uint8_t* rgb = (uint8_t*) &p.rgb;
        r = (float) pow(rgb[0], 2);
        g = (float) pow(rgb[1], 2);
        b = (float) pow(rgb[2], 2);
    };

    // From Values
    __host__ __device__
    inline Voxel(float x_, float y_, float z_, float r_, float g_, float b_) 
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {};

    __host__ __device__
    inline Voxel operator+(Voxel v) const {
        return Voxel(x + v.x, y + v.y, z + v.z, r + v.r, g + v.g, b + v.b);
    }
};

#endif