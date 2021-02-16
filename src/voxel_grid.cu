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
    ui32XYZ voxel_nums;
    fXYZ voxel_lengths;
    fXYZ min_xyz;
    PointToKey(ui32XYZ _voxel_nums, fXYZ _voxel_lengths, fXYZ _min_xyz) : voxel_nums(_voxel_nums), voxel_lengths(_voxel_lengths), min_xyz(_min_xyz) {};

    __host__ __device__
    uint32_t operator()(const Point v) {
        if(isnan(v.x)) return UINT32_MAX;
        uint32_t idx_x = round((v.x - min_xyz.x) / voxel_lengths.x);
        uint32_t idx_y = round((v.y - min_xyz.y) / voxel_lengths.y);
        uint32_t idx_z = round((v.z - min_xyz.z) / voxel_lengths.z);
        
        return idx_x + idx_y * voxel_nums.x + idx_z * voxel_nums.x * voxel_nums.y;
    }
};


struct PointToColor {
    __host__ __device__
    ui64RGB operator()(const Point p) {
        uint8_t* rgb = (uint8_t*) &p.rgb;
        uint16_t r = rgb[0];
        uint16_t g = rgb[1];
        uint16_t b = rgb[2];
        return ui64RGB(r * r, g * g, b * b);
    }
};


struct IdxColorWeightToPoint {
    ui32XYZ voxel_nums;
    fXYZ voxel_lengths;
    fXYZ min_xyz;
    IdxColorWeightToPoint(ui32XYZ _voxel_nums, fXYZ _voxel_lengths, fXYZ _min_xyz) : voxel_nums(_voxel_nums), voxel_lengths(_voxel_lengths), min_xyz(_min_xyz) {
        printf("%f %f %f %f %f %f\n", voxel_lengths.x, voxel_lengths.y, voxel_lengths.z, min_xyz.x, min_xyz.y, min_xyz.z);
    };


    __host__ __device__
    Point operator()(thrust::tuple<uint32_t, ui64RGB, uint32_t> t) {
        uint32_t idx = thrust::get<0>(t);
        ui64RGB color = thrust::get<1>(t);
        uint32_t weight = thrust::get<2>(t);

        if(weight < 3) return Point(0, 0, 0, 0);
        
        uint32_t idx_z = idx / (voxel_nums.x * voxel_nums.y);
        uint32_t idx_y = (idx - (idx_z * voxel_nums.x * voxel_nums.y)) / voxel_nums.x;
        uint32_t idx_x = idx % voxel_nums.x;

        // if(idx_x == idx_y) printf("Damnnit %d\n", idx);
        if(idx - (idx_x + idx_y * voxel_nums.x + idx_z * voxel_nums.x * voxel_nums.y))
            printf("AAAAAAAAA %d %d\n", idx, idx_x + idx_y * voxel_nums.x + idx_z * voxel_nums.x * voxel_nums.y);


        Point p;
        p.x = ((float) idx_x) * voxel_lengths.x + min_xyz.x;
        p.y = ((float) idx_y) * voxel_lengths.y + min_xyz.y;
        p.z = ((float) idx_z) * voxel_lengths.z + min_xyz.z;
        // printf("%f %f %f\n", p.x, p.y, p.z);

        uint8_t* rgb = (uint8_t*) &p.rgb;
        rgb[0] = (uint8_t) round(sqrt((float) (color.r / weight)));
        rgb[1] = (uint8_t) round(sqrt((float) (color.g / weight)));
        rgb[2] = (uint8_t) round(sqrt((float) (color.b / weight)));

        return p;
    }
};

struct is_point_invalid
{
  __host__ __device__
  bool operator()(const Point p) { return isnan(p.x); }
};


struct TFAndCropPoint {   
    const Eigen::Matrix4f tf;
    const fXYZ min_xyz;
    const fXYZ max_xyz;

    TFAndCropPoint(Eigen::Matrix4f _tf, fXYZ _min_xyz, fXYZ _max_xyz) : tf(_tf), min_xyz(_min_xyz), max_xyz(_max_xyz) {};

    __host__ __device__ 
    Point operator()(Point p) {   
        Point np;
        if(p.x < min_xyz.x || p.x > max_xyz.x || p.y < min_xyz.y || p.y > max_xyz.y || p.z < min_xyz.z || p.z > max_xyz.z ) 
            return Point(nanf(""), nanf(""), nanf(""), 0);

        np.x = static_cast<float> (tf (0, 0) * p.x + tf (0, 1) * p.y + tf (0, 2) * p.z + tf (0, 3));
        np.y = static_cast<float> (tf (1, 0) * p.x + tf (1, 1) * p.y + tf (1, 2) * p.z + tf (1, 3));
        np.z = static_cast<float> (tf (2, 0) * p.x + tf (2, 1) * p.y + tf (2, 2) * p.z + tf (2, 3));
        np.rgb = p.rgb;

        return np;
    }
};


uint32_t voxel_grid(thrust::device_vector<Point>& d_points, float* out, ui32XYZ voxel_nums, fXYZ voxel_lengths, fXYZ min_xyz) {
    uint32_t num = d_points.size();

    // Step 1: Produce Indizes and Colors
    thrust::device_vector<ui64RGB> d_colors(num);
    thrust::device_vector<uint32_t> d_voxel_idxs(num);
    thrust::transform(d_points.begin(), d_points.end(), d_colors.begin(), PointToColor());
    thrust::transform(d_points.begin(), d_points.end(), d_voxel_idxs.begin(), PointToKey(voxel_nums, voxel_lengths, min_xyz));

    // Step 2: Sort by Idxs
    thrust::device_vector<uint32_t> d_point_idxs(num);
    thrust::device_vector<ui64RGB> d_colors_sorted(num);
    thrust::sequence(d_point_idxs.begin(), d_point_idxs.end());
    thrust::sort_by_key(d_voxel_idxs.begin(), d_voxel_idxs.end(), d_point_idxs.begin());
    // thrust::copy(thrust::make_permutation_iterator(d_colors.begin(), point_idxs.begin()), thrust::make_permutation_iterator(d_colors.begin(), point_idxs.end()), d_colors_sorted.begin());

    // Step 2: Sort by Idxs
    // thrust::sort_by_key(d_voxel_idxs.begin(), d_voxel_idxs.end(), d_colors.begin()); // This sorts the keys (d_voxel_idxs) as well

    
    // Step 3: Count Amount of Voxels
    // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
    // uint32_t num_voxels = thrust::inner_product(d_voxel_idxs.begin(), d_voxel_idxs.end() - 1, d_voxel_idxs.begin() + 1, 1, thrust::plus<uint32_t>(), thrust::not_equal_to<uint32_t>());

    // Step 4: Produce "Histogram" for weights
    thrust::device_vector<uint32_t> d_weights(num);
    thrust::device_vector<uint32_t> d_idx_reduced(num);
    auto new_ends = thrust::reduce_by_key(d_voxel_idxs.begin(), d_voxel_idxs.end(), thrust::constant_iterator<uint32_t>(1), d_idx_reduced.begin(), d_weights.begin());
    uint32_t num_voxels = new_ends.first - d_idx_reduced.begin();
    d_weights.resize(num_voxels);
    d_idx_reduced.resize(num_voxels);

    // Step 5: Merge all values with same idx
    thrust::device_vector<uint32_t> d_idx_after_vox(num_voxels);
    thrust::device_vector<ui64RGB> d_colors_out(num_voxels);
    thrust::reduce_by_key(thrust::device, d_voxel_idxs.begin(), d_voxel_idxs.end(), thrust::make_permutation_iterator(d_colors.begin(), d_point_idxs.begin()), d_idx_after_vox.begin(), d_colors_out.begin(), thrust::equal_to<uint32_t>(), thrust::plus<ui64RGB>());

    // Step 6: Divide by weight
    thrust::device_vector<Point> d_point_cloud_out(num_voxels);
    auto dit_idx_color_weight_begin = thrust::make_zip_iterator(thrust::make_tuple(d_idx_after_vox.begin(), d_colors_out.begin(), d_weights.begin()));
    auto dit_idx_color_weight_end   = thrust::make_zip_iterator(thrust::make_tuple(d_idx_after_vox.end(),   d_colors_out.end(),   d_weights.end()));

    thrust::transform(dit_idx_color_weight_begin, dit_idx_color_weight_end, d_point_cloud_out.begin(), IdxColorWeightToPoint(voxel_nums, voxel_lengths, min_xyz));

    thrust::copy(d_point_cloud_out.begin(), d_point_cloud_out.end(), (Point*) out);

    return num_voxels;

}



uint32_t transformCropAndVoxelizeCenter(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out) {
    // Count all Points
    size_t num_points = msg->height * msg->width;
    fXYZ min_xyz(-2, -2, -2);
    fXYZ max_xyz(2, 2, 2);
    fXYZ voxel_lengths(0.005, 0.005, 0.005);
    ui32XYZ voxel_nums(0, 0, 0);

    voxel_nums.x = ceil((max_xyz.x - min_xyz.x) / voxel_lengths.x);
    voxel_nums.y = ceil((max_xyz.y - min_xyz.y) / voxel_lengths.y);
    voxel_nums.z = ceil((max_xyz.z - min_xyz.z) / voxel_lengths.z);
        
    Eigen::Matrix4f tf;
    tf << 0, 1, 0, 0,
          1, 0, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1;
        
    thrust::device_vector<Point> d_points(num_points);
    thrust::copy(reinterpret_cast<const Point *>(&msg->data[0]), reinterpret_cast<const Point *>(&msg->data[msg->data.size()]), d_points.begin());
    thrust::transform(d_points.begin(), d_points.end(), d_points.begin(), TFAndCropPoint(tf, min_xyz, max_xyz));

    // Remove all invalid points or points outside the box
    size_t new_size = thrust::remove_if(d_points.begin(), d_points.end(), is_point_invalid()) - d_points.begin();
    d_points.resize(new_size);

    return voxel_grid(d_points, point_cloud_out, voxel_nums, voxel_lengths, min_xyz);
}

