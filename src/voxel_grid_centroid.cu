#include <cassert>
#include <iostream>

#include "voxel_grid.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

using namespace std;


struct PointToKey {
    sXYZ voxel_nums;
    fXYZ voxel_lengths;
    fXYZ min_xyz;
    PointToKey(sXYZ _voxel_nums, fXYZ _voxel_lengths, fXYZ _min_xyz) : voxel_nums(_voxel_nums), voxel_lengths(_voxel_lengths), min_xyz(_min_xyz) {};

    __host__ __device__
    uint32_t operator()(const Voxel v) {
        if(isnan(v.x)) return UINT32_MAX;
        uint32_t idx_x = round((v.x / voxel_lengths.x) - min_xyz.x);
        uint32_t idx_y = round((v.y / voxel_lengths.y) - min_xyz.y);
        uint32_t idx_z = round((v.z / voxel_lengths.z) - min_xyz.z);

        return idx_x + idx_y * voxel_nums.x + idx_z * voxel_nums.x * voxel_nums.y;
    }
};


struct PointDiv {
    __host__ __device__
    Point operator()(const uint32_t& w, const Voxel& v) {
        if(w < 3) return Point(0, 0, 0, 0);

        Point p;
        p.x = v.x / w;
        p.y = v.y / w;
        p.z = v.z / w;

        uint8_t* rgb = (uint8_t*) &p.rgb;
        rgb[0] = round(sqrt(v.r / w));
        rgb[1] = round(sqrt(v.g / w));
        rgb[2] = round(sqrt(v.b / w));

        return p;
    }
};

struct is_voxel_invalid
{
  __host__ __device__
  bool operator()(const Voxel v)
  {
    return isnan(v.x);
  }
};


struct PointToVoxel
{   
    const Eigen::Matrix4f tf;
    fXYZ min_xyz;
    fXYZ max_xyz;

    PointToVoxel(Eigen::Matrix4f _tf, fXYZ _min_xyz, fXYZ _max_xyz) : tf(_tf), min_xyz(_min_xyz), max_xyz(_max_xyz) {};

    __host__ __device__ Voxel operator()(Point p) //difsq
    {
        if(p.x < min_xyz.x || p.x > max_xyz.x || p.y < min_xyz.y || p.y > max_xyz.y || p.z < min_xyz.z || p.z > max_xyz.z ) 
            return Voxel(nanf(""), nanf(""), nanf(""), 0, 0, 0);

        float tgt[3];
        tgt[0] = static_cast<float> (tf (0, 0) * p.x + tf (0, 1) * p.y + tf (0, 2) * p.z + tf (0, 3));
        tgt[1] = static_cast<float> (tf (1, 0) * p.x + tf (1, 1) * p.y + tf (1, 2) * p.z + tf (1, 3));
        tgt[2] = static_cast<float> (tf (2, 0) * p.x + tf (2, 1) * p.y + tf (2, 2) * p.z + tf (2, 3));

        uint8_t* rgb = (uint8_t*) &p.rgb;

        return Voxel(
          tgt[0], 
          tgt[1], 
          tgt[2], 
          pow(rgb[0], 2), 
          pow(rgb[1], 2), 
          pow(rgb[2], 2)
        );
    }
};


uint32_t voxel_grid(
    uint32_t num, thrust::device_vector<Voxel> d_voxel_cloud, float* out, sXYZ voxel_nums, fXYZ voxel_lengths, fXYZ min_xyz) {
    thrust::device_vector<uint32_t> voxel_idxs(num);

    // Step 1: Produce Indizes
    thrust::transform(d_voxel_cloud.begin(), d_voxel_cloud.end(), voxel_idxs.begin(), PointToKey(voxel_nums, voxel_lengths, min_xyz));
    
    // Step 2: Sort by Idxs
    thrust::device_vector<uint32_t> point_idxs(num);
    thrust::device_vector<Voxel> d_voxel_cloud_sorted(num);
    thrust::sequence(point_idxs.begin(), point_idxs.end());
    thrust::sort_by_key(voxel_idxs.begin(), voxel_idxs.end(), point_idxs.begin());
    thrust::copy(thrust::make_permutation_iterator(d_voxel_cloud.begin(), point_idxs.begin()), thrust::make_permutation_iterator(d_voxel_cloud.begin(), point_idxs.end()), d_voxel_cloud_sorted.begin());

    thrust::sort_by_key(voxel_idxs.begin(), voxel_idxs.end(), d_voxel_cloud.begin());

    // Step 3: Count Amount of Voxels
    // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
    uint32_t num_voxels = thrust::inner_product(voxel_idxs.begin(), voxel_idxs.end() - 1, voxel_idxs.begin() + 1, 1, thrust::plus<uint32_t>(), thrust::not_equal_to<uint32_t>());

    thrust::device_vector<uint32_t> d_weights(num_voxels);
    thrust::device_vector<uint32_t> d_idx_reduced(num_voxels);

    // Step 4: Produce "Histogram" for weights
    thrust::reduce_by_key(voxel_idxs.begin(), voxel_idxs.end(), thrust::constant_iterator<uint32_t>(1), d_idx_reduced.begin(), d_weights.begin());

    // Step 5: Merge all values with same idx
    thrust::device_vector<uint32_t> d_idx_after_vox(num_voxels);
    thrust::device_vector<Voxel> d_voxels_out(num_voxels);
    thrust::reduce_by_key(thrust::device, voxel_idxs.begin(), voxel_idxs.end(), d_voxel_cloud_sorted.begin(), d_idx_after_vox.begin(), d_voxels_out.begin(), thrust::equal_to<uint32_t>(), thrust::plus<Voxel>());

    // Step 6: Divide by weight
    thrust::device_vector<Point> d_point_cloud_out(num_voxels);
    thrust::transform(d_weights.begin(), d_weights.end(), d_voxels_out.begin(), d_point_cloud_out.begin(), PointDiv());

    thrust::copy(d_point_cloud_out.begin(), d_point_cloud_out.end(), (Point*) out);

    return num_voxels;

}



uint32_t transformCropAndVoxelize(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out) {
    // Count all Points
    size_t num_points = msg->height * msg->width;
    fXYZ min_xyz(-4, -4, -4);
    fXYZ max_xyz(4, 4, 4);
    fXYZ voxel_lengths(0.005, 0.005, 0.005);
    sXYZ voxel_nums(0, 0, 0);

    voxel_nums.x = ceil((max_xyz.x - min_xyz.x) / voxel_lengths.x);
    voxel_nums.y = ceil((max_xyz.y - min_xyz.y) / voxel_lengths.y);
    voxel_nums.z = ceil((max_xyz.z - min_xyz.z) / voxel_lengths.z);

    Eigen::Matrix4f tf;
    tf << 0, 1, 0, 0,
          1, 0, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1;
        
    thrust::device_vector<Point> d_points(num_points);
    thrust::device_vector<Voxel> d_voxels(num_points);
    thrust::copy(reinterpret_cast<const Point *>(&msg->data[0]), reinterpret_cast<const Point *>(&msg->data[msg->data.size()]), d_points.begin());
    thrust::transform(d_points.begin(), d_points.end(), d_voxels.begin(), PointToVoxel(tf, min_xyz, max_xyz));

    // Remove all invalid points or points outside the box
    size_t new_size = thrust::remove_if(d_voxels.begin(), d_voxels.end(), is_voxel_invalid()) - d_voxels.begin();
    d_voxels.resize(new_size);

    return voxel_grid(new_size, d_voxels, point_cloud_out, voxel_nums, voxel_lengths, min_xyz);
}