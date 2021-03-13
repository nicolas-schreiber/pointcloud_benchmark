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

struct __attribute__((__packed__)) ui64RGB {
    uint64_t r = 0;
    uint64_t g = 0;
    uint64_t b = 0;

    // Empty
    __host__ __device__
    inline ui64RGB() {};

        // From Values
    __host__ __device__
    inline ui64RGB(uint64_t _r, uint64_t _g, uint64_t _b) 
        : r(_r), g(_g), b(_b){};
    
    __host__ __device__
    inline ui64RGB operator+(ui64RGB v) const {
        return ui64RGB(r + v.r, g + v.g, b + v.b);
    }

    __host__ __device__
    inline ui64RGB operator*(ui64RGB v) const {
        return ui64RGB(r * v.r, g * v.g, b * v.b);
    }
};

struct __attribute__((__packed__)) ui8RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

struct __attribute__((__packed__)) ui8RGB {
    uint8_t r;
    uint8_t g;
    uint8_t b;
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

struct __attribute__((__packed__)) ui32XYZ {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    __host__ __device__
    inline ui32XYZ(){};

    __host__ __device__
    inline ui32XYZ(uint32_t x_, uint32_t y_, uint32_t z_) 
        : x(x_), y(y_), z(z_) {};

    __host__ __device__
    inline ui32XYZ(const ui32XYZ& other) 
        : x(other.x), y(other.y), z(other.z) {};

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

    __host__ __device__
    inline sXYZ(const sXYZ& other) 
        : x(other.x), y(other.y), z(other.z) {};

};

uint32_t transformCropAndVoxelize(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out);
uint32_t transformCropAndVoxelizeCenter(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out);

struct __attribute__((__packed__)) Point {
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
        r = powf(rgb[0], 2);
        g = powf(rgb[1], 2);
        b = powf(rgb[2], 2);
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