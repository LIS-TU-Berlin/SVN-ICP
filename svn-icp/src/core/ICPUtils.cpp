/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 Shiping Ma and Haoming Zhang
    email: shiping.ma@tu-berlin.de and haoming.zhang@rwth-aachen.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    ICPUtils.h
 * @brief   ICPUtils
 * @author  Shiping Ma*
 * @author  Haoming Zhang
 * @date    June 22, 2025
 */

#include <string>
#include <chrono>
#include <pcl/visualization/pcl_visualizer.h>

#include "data/DataTypes.h"
#include "core/ICPUtils.h"

namespace svnicp {
  using namespace data_types;

  std::vector<double> pointcloud2vector(const Cloud_t::Ptr &cloud) {
    std::vector<double> cloud_vec;
    for (const auto &pt: cloud->points) {
      cloud_vec.push_back(pt.x);
      cloud_vec.push_back(pt.y);
      cloud_vec.push_back(pt.z);
    }
    return cloud_vec;
  }

  torch::Tensor vector2tensor(std::vector<double> v, Device_type device) {
    long size_v = v.size() / 3;
    torch::Tensor cloud_tensor = torch::from_blob(v.data(), {size_v, 3}
                                                  , torch::TensorOptions().dtype(torch::kFloat64)
    ).to(device);
    return cloud_tensor;
  }

  torch::Tensor initialize_particles(int particle_count, Device_type device, const torch::Tensor &ub,
                                     const torch::Tensor &lb) {
    torch::Tensor init_pose = torch::empty({6, particle_count, 1}, torch::TensorOptions().dtype(torch::kFloat64)).
        to(device);
    if (particle_count == 1)
      return torch::zeros({6, 1, 1}).to(device).to(torch::kFloat64);
    init_pose[0] = (ub[0] - lb[0]) * torch::rand({particle_count, 1}).to(device) + lb[0];
    init_pose[1] = (ub[1] - lb[1]) * torch::rand({particle_count, 1}).to(device) + lb[1];
    init_pose[2] = (ub[2] - lb[2]) * torch::rand({particle_count, 1}).to(device) + lb[2];
    init_pose[3] = (ub[3] - lb[3]) * torch::rand({particle_count, 1}).to(device) + lb[3];
    init_pose[4] = (ub[4] - lb[4]) * torch::rand({particle_count, 1}).to(device) + lb[4];
    init_pose[5] = (ub[5] - lb[5]) * torch::rand({particle_count, 1}).to(device) + lb[5];
    return init_pose;
  }

  torch::Tensor initialize_particles_gaussian(int particle_count, Device_type device, const gtsam::Vector6 &cov) {
    if (particle_count == 1)
      return torch::zeros({6, 1, 1}).to(device).to(torch::kFloat64);
    const auto std_dev = cov.cwiseSqrt();
    const torch::Tensor init_pose = torch::empty({6, particle_count, 1}, torch::TensorOptions().dtype(torch::kFloat64)).
        to(device);
    init_pose[0] = torch::normal(0, std_dev[0], {particle_count, 1}).to(torch::kCUDA);
    init_pose[1] = torch::normal(0, std_dev[1], {particle_count, 1}).to(torch::kCUDA);
    init_pose[2] = torch::normal(0, std_dev[2], {particle_count, 1}).to(torch::kCUDA);
    init_pose[3] = torch::normal(0, std_dev[3], {particle_count, 1}).to(torch::kCUDA);
    init_pose[4] = torch::normal(0, std_dev[4], {particle_count, 1}).to(torch::kCUDA);
    init_pose[5] = torch::normal(0, std_dev[5], {particle_count, 1}).to(torch::kCUDA);
    auto bound = 3 * torch::tensor({std_dev[0], std_dev[1], std_dev[2], std_dev[3], std_dev[4], std_dev[5]})
                 .to(device).reshape({6, 1}).expand({6, particle_count}).view({6, particle_count, 1});
    return init_pose.clamp(-bound, bound);
  }

  gtsam::Vector6 tensor2Vector6(const torch::Tensor &this_point) {
    const auto pose_tr_cpu = this_point.to(torch::kCPU);
    auto pose_vector = std::vector<double>(pose_tr_cpu.data_ptr<double>(),
                                           pose_tr_cpu.data_ptr<double>() + 6);
    return gtsam::Vector6(pose_vector.data());
  }

  gtsam::Pose3 tensor2gtsamPose3(const torch::Tensor &this_point) {
    const auto pose_tr_cpu = this_point.to(torch::kCPU);
    const auto pose_vector = std::vector<double>(pose_tr_cpu.data_ptr<double>(),
                                                 pose_tr_cpu.data_ptr<double>() + 6);
    const auto &x = pose_vector[0];
    const auto &y = pose_vector[1];
    const auto &z = pose_vector[2];
    const auto &roll = pose_vector[3];
    const auto &pitch = pose_vector[4];
    const auto &yaw = pose_vector[5];

    gtsam::Vector6 vector;
    vector << roll, pitch, yaw, x, y, z;
    return gtsam::Pose3(gtsam::Rot3::Expmap(vector(Eigen::seq(0, 2))), gtsam::Point3(x, y, z));
  }

  std::tuple<Eigen::Matrix<double, 3, 3>, Eigen::Vector3d> tensor2gtsamtransform(const torch::Tensor &this_point) {
    const auto x = this_point[0].item<double>();
    const auto y = this_point[1].item<double>();
    const auto z = this_point[2].item<double>();
    const auto roll = this_point[3].item<double>();
    const auto pitch = this_point[4].item<double>();
    const auto yaw = this_point[5].item<double>();


    const auto rollAngle(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
    const auto pitchAngle(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    const auto yawAngle(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    //        auto rotation_matrix = gtsam::Rot3::RzRyRx(double(roll), double(pitch), yaw).matrix();
    Eigen::Matrix<double, 3, 3> rotation_matrix = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
    auto translation_vector = Eigen::Vector3d(x, y, z);

    return {rotation_matrix, translation_vector};
  }

  Eigen::Matrix<double, 4, 4> tensor2Matrix(const torch::Tensor &this_point) {
    const auto [rot, tran] = tensor2gtsamtransform(this_point);
    //        auto rot_mat = rot.matrix();
    gtsam::Matrix4 T;
    T.block(0, 0, 3, 3) = rot;
    T.block(0, 3, 3, 1) = tran;
    T.block(3, 0, 1, 4) << 0, 0, 0, 1;
    return T;
  }

  std::tuple<gtsam::Vector6, gtsam::Matrix66> left2right(std::vector<double> particles,
                                                         const gtsam::Pose3 &initial_guess) {
    const int particle_count = particles.size() / 6;
    Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor> particle_matrix_tmp(6, particle_count);
    particle_matrix_tmp = Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor> >(
      particles.data(), 6, particle_count);

    // convert to se3
    Eigen::Matrix<double, 6, Eigen::Dynamic> particle_matrix_left(6, particle_count);
    tbb::parallel_for(0, particle_count, [&](int i) {
      auto particle = particle_matrix_tmp.col(i);
      const auto pose3 = gtsam::Pose3(gtsam::Rot3::Expmap(particle(Eigen::seq(3, 5))),
                                      gtsam::Point3(particle(Eigen::seq(0, 2))));
      particle_matrix_left.col(i) = gtsam::Pose3::Logmap(pose3);
    });

    const auto Adj_inv_init = initial_guess.inverse().AdjointMap();
    auto particle_matrix_right = Adj_inv_init * particle_matrix_left;
    ////This is the right hand side vector in se3!!!
    const gtsam::Vector6 v_right = particle_matrix_right.rowwise().mean();

    //remind that the vector required by my kalman filter is [t, phi]
    gtsam::Vector6 icp_correction;
    const gtsam::Pose3 correction_pose = gtsam::Pose3::Expmap(v_right);
    icp_correction.block<3, 1>(0, 0) = correction_pose.translation();
    icp_correction.block<3, 1>(3, 0) = gtsam::Rot3::Logmap(correction_pose.rotation());

    //We can use the particles of the right hand side to calculate the cov..
    //Note, we want to get the cov of translation and phi, but the particles are now on se3.
    // But translation is approximately equal to the trans part on se3, if we should do, use tbb::paraller_for
    const auto diff_matrix = particle_matrix_right - v_right;
    gtsam::Vector6 variance = diff_matrix.cwiseAbs2().rowwise().sum() / particle_count;
    gtsam::Vector6 variance_kf;
    variance_kf.block<3, 1>(0, 0) = variance.block<3, 1>(3, 0);
    variance_kf.block<3, 1>(3, 0) = variance.block<3, 1>(0, 0);

    return {icp_correction, variance_kf.asDiagonal()};
  }
}
