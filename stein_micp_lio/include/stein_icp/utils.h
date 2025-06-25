//
// Created by haoming on 01.06.23.
//

#ifndef STEIN_ICP_UTILS_H
#define STEIN_ICP_UTILS_H

#include <string>
#include <iostream>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <torch/torch.h>
#include <gtsam/geometry/Pose3.h>
#include <tbb/parallel_for.h>

#include "types.h"


namespace stein_icp
{
/**
 * \brief transfer Cloud_t.points to a 1 dimension std::vector
 */
    std::vector<double> pointcloud2vector(const Cloud_t::Ptr &cloud);  //pcl to std::vector

/**
 *
 * \brief transfer std::vector to tensor
 */
    torch::Tensor vector2tensor(std::vector<double>, Device_type device);    //std::vector to torch::Tensor

/**
 * \brief    initialize pose particles with particle_count, in uniform distribution
 *
 * \return   return pose in size: 6*particle_count*1
 */
    torch::Tensor
    initialize_particles(int particle_count, Device_type device, const torch::Tensor &ub, const torch::Tensor &lb);

    torch::Tensor initialize_particles_gaussian(int particle_count, Device_type device, const gtsam::Vector6 &cov);

    gtsam::Vector6 tensor2Vector6(const torch::Tensor &this_point);
    /**
     *  \brief transform Tensor to Pose3
     */
    gtsam::Pose3 tensor2gtsamPose3(const torch::Tensor &this_point);

    std::tuple<Eigen::Matrix<double, 3, 3>, Eigen::Vector3d> tensor2gtsamtransform(const torch::Tensor &this_point);

    Eigen::Matrix<double, 4, 4> tensor2Matrix(const torch::Tensor &this_point);

    std::tuple<gtsam::Vector6, gtsam::Matrix6> left2right(std::vector<double> particles, const gtsam::Pose3 &initial_guess);
}

class Timer {
public:
    Timer(std::string name = "") : name_(name){};

    inline void reset(){
        start_ = std::chrono::steady_clock::now();
    }

    inline double duration(){
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start_;
        return duration.count();
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};



#endif //STEIN_ICP_UTILS_H
