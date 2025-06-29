/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 XXX
    email: XXX

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    ESKF.h
 * @brief   Implementation of error-state Kalman filter
 * @author  UNKNOWN
 * @author  UNKNOWN
 * @date    June 22, 2025
 */

#ifndef SVNICP_ESKF_H
#define SVNICP_ESKF_H

#include "KalmanFilterBase.h"

namespace svnicp::estimator {

  class ErrorStateKalmanFilter final : public KalmanFilterBase {
  public:
    explicit ErrorStateKalmanFilter(const sensor::LIOParam &lio_param, const sensor::IMURandomWalk &imu_rw);

    void predict(svnicp::data_types::State &body_state,
                 const svnicp::data_types::IMUMeasurement &imu_previous,
                 svnicp::data_types::IMUMeasurement &imu_current) override;

    void update(svnicp::data_types::State &body_state,
                const gtsam::Matrix6 &SteinICP_Cov,
                const gtsam::Vector6 &icp_correction) override;

    gtsam::Pose3 get_initial_guess();

    gtsam::Pose3 get_state_matrix();

    Eigen::Matrix<double, 15, 6> get_KFGain();

    gtsam::Vector6 get_random_walk_variance() {
      return (gtsam::Vector6() << process_noise_.block<3, 3>(3, 3).diagonal(), process_noise_.block<3, 3>(6, 6).
              diagonal()).finished();
    };

    [[nodiscard]] Eigen::Matrix<double, 15, 15> get_cov() const{
      return cov_;
    }

  private:
    gtsam::Vector15 state_;
    Eigen::Matrix<double, 15, 15> cov_;
    gtsam::Pose3 error_state_pose_;
    Eigen::Matrix<double, 6, 15> observation_matrix_;
    gtsam::Pose3 initial_guess_;
    Eigen::Matrix<double, 15, 6> KF_gain_;
    Eigen::Matrix<double, 15, 15> process_noise_;
  };
}


#endif //SVNICP_ESKF_H
