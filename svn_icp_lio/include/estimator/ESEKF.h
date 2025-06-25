//
// Created by haoming on 21.09.24.
//

#ifndef STEIN_MICP_ESEKF_H
#define STEIN_MICP_ESEKF_H

#include "Estimator.h"

namespace estimator {
  class ErrorStateKalmanFilter : public Estimator {
  public:
    explicit ErrorStateKalmanFilter(const IMU::LioParam &lio_param, const IMU::ImuRandomWalk &imu_rw);

    void predict(fgo::data_types::State &body_state,
                 const fgo::data_types::IMUMeasurement &imu_previous,
                 fgo::data_types::IMUMeasurement &imu_current) override;

    void update(fgo::data_types::State &body_state,
                const gtsam::Matrix6 &SteinICP_Cov,
                const gtsam::Vector6 &icp_correction) override;

    gtsam::Pose3 get_initial_guess();

    gtsam::Pose3 get_state_matrix();

    Eigen::Matrix<double, 15, 6> get_KFGain();

    gtsam::Vector6 get_random_walk_variance() {
      return (gtsam::Vector6() << Process_Noise_.block<3, 3>(3, 3).diagonal(), Process_Noise_.block<3, 3>(6, 6).
              diagonal()).finished();
    };

    inline Eigen::Matrix<double, 15, 15> get_cov(){
        return Cov_;
    }

  private:
    gtsam::Vector15 state_;
    Eigen::Matrix<double, 15, 15> Cov_;
    gtsam::Pose3 error_state_pose_;
    Eigen::Matrix<double, 6, 15> Observation_Matrix_;
    gtsam::Pose3 initial_guess_;
    Eigen::Matrix<double, 15, 6> KF_Gain_;
    Eigen::Matrix<double, 15, 15> Process_Noise_;
  };
}


#endif //STEIN_MICP_ESEKF_H
