#include <cassert>
#include <iostream>

#include "pointcloud_cuda.cuh"
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

struct FrustumCulling {
    Eigen::Vector4f pl_n; // near plane 
    Eigen::Vector4f pl_f; // far plane
    Eigen::Vector4f pl_t; // top plane
    Eigen::Vector4f pl_b; // bottom plane
    Eigen::Vector4f pl_r; // right plane
    Eigen::Vector4f pl_l; // left plane
    
    FrustumCulling(const Eigen::Matrix4f& camera_pose, float vfov, float hfov, float np_dist, float fp_dist) {
        Eigen::Vector3f view  = camera_pose.block<3, 1> (0, 0);    // view vector for the camera  - first column of the rotation matrix
        Eigen::Vector3f up    = camera_pose.block<3, 1> (0, 1);      // up vector for the camera    - second column of the rotation matrix
        Eigen::Vector3f right = camera_pose.block<3, 1> (0, 2);   // right vector for the camera - third column of the rotation matrix
        Eigen::Vector3f T     = camera_pose.block<3, 1> (0, 3);       // The (X, Y, Z) position of the camera w.r.t origin
       
        float vfov_rad = float (vfov * M_PI / 180);  // degrees to radians
        float hfov_rad = float (hfov * M_PI / 180);  // degrees to radians
        
        float np_h = float (2 * tan (vfov_rad / 2) * np_dist);  // near plane height
        float np_w = float (2 * tan (hfov_rad / 2) * np_dist);  // near plane width
       
        float fp_h = float (2 * tan (vfov_rad / 2) * fp_dist);  // far plane height
        float fp_w = float (2 * tan (hfov_rad / 2) * fp_dist);  // far plane width
       
        Eigen::Vector3f fp_c (T + view * fp_dist);                           // far plane center
        Eigen::Vector3f fp_tl (fp_c + (up * fp_h / 2) - (right * fp_w / 2));  // Top left corner of the far plane
        Eigen::Vector3f fp_tr (fp_c + (up * fp_h / 2) + (right * fp_w / 2));  // Top right corner of the far plane
        Eigen::Vector3f fp_bl (fp_c - (up * fp_h / 2) - (right * fp_w / 2));  // Bottom left corner of the far plane
        Eigen::Vector3f fp_br (fp_c - (up * fp_h / 2) + (right * fp_w / 2));  // Bottom right corner of the far plane
       
        Eigen::Vector3f np_c (T + view * np_dist);                            // near plane center
        Eigen::Vector3f np_tl (np_c + (up * np_h/2) - (right * np_w/2));      // Top left corner of the near plane
        Eigen::Vector3f np_tr (np_c + (up * np_h / 2) + (right * np_w / 2));  // Top right corner of the near plane
        Eigen::Vector3f np_bl (np_c - (up * np_h / 2) - (right * np_w / 2));  // Bottom left corner of the near plane
        Eigen::Vector3f np_br (np_c - (up * np_h / 2) + (right * np_w / 2));  // Bottom right corner of the near plane
       
        pl_f.head<3> () = (fp_bl - fp_br).cross (fp_tr - fp_br);  // Far plane equation - cross product of the 
        pl_f (3) = -fp_c.dot (pl_f.head<3> ());                   // perpendicular edges of the far plane
       
        pl_n.head<3> () = (np_tr - np_br).cross (np_bl - np_br);  // Near plane equation - cross product of the 
        pl_n (3) = -np_c.dot (pl_n.head<3> ());                   // perpendicular edges of the far plane
       
        Eigen::Vector3f a (fp_bl - T);  // Vector connecting the camera and far plane bottom left
        Eigen::Vector3f b (fp_br - T);  // Vector connecting the camera and far plane bottom right
        Eigen::Vector3f c (fp_tr - T);  // Vector connecting the camera and far plane top right
        Eigen::Vector3f d (fp_tl - T);  // Vector connecting the camera and far plane top left
       
        //                   Frustum and the vectors a, b, c and d. T is the position of the camera
        //                             _________
        //                           /|       . |
        //                       d  / |   c .   |
        //                         /  | __._____| 
        //                        /  /  .      .
        //                 a <---/-/  .    .
        //                      / / .   .  b
        //                     /   .
        //                     . 
        //                   T
        //
       
        pl_r.head<3> () = b.cross (c);
        pl_l.head<3> () = d.cross (a);
        pl_t.head<3> () = c.cross (d);
        pl_b.head<3> () = a.cross (b);
       
        pl_r (3) = -T.dot (pl_r.head<3> ());
        pl_l (3) = -T.dot (pl_l.head<3> ());
        pl_t (3) = -T.dot (pl_t.head<3> ());
        pl_b (3) = -T.dot (pl_b.head<3> ());     
    }

    __host__ __device__
    bool operator()(const Point v) {
        Eigen::Vector4f pt (v.x,
                            v.y,
                            v.z,
                            1.0f);
        bool is_in_fov = (pt.dot (pl_l) <= 0) && 
                        (pt.dot (pl_r) <= 0) &&
                        (pt.dot (pl_t) <= 0) && 
                        (pt.dot (pl_b) <= 0) && 
                        (pt.dot (pl_f) <= 0) &&
                        (pt.dot (pl_n) <= 0);

        return is_in_fov;
    }
     
};

uint32_t removePointsInsideView(sensor_msgs::PointCloud2::ConstPtr msg, float* point_cloud_out) {
    float num_points = msg->width * msg->height;
    // Count all Points      
    thrust::device_vector<Point> d_points(num_points);

    Eigen::Matrix4f cam_pose;
    cam_pose << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;


    // Remove all invalid points or points outside the box
    FrustumCulling cull{cam_pose, 90, 90, 0.05, 3.00};
    size_t new_size = thrust::remove_if(d_points.begin(), d_points.end(), cull) - d_points.begin();
    d_points.resize(new_size);

    thrust::copy(d_points.begin(), d_points.end(), (Point*) point_cloud_out);


    return new_size;
}