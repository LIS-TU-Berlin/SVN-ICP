//
// Created by haoming on 30.09.24.
//

#ifndef STEIN_MICP_LIO_IMU_H
#define STEIN_MICP_LIO_IMU_H

#endif //STEIN_MICP_LIO_IMU_H

#include "data/DataTypes.h"

namespace IMU{
    static constexpr double G = 9.782940329221166;

    struct LioParam{
        gtsam::Matrix3 R_lidar_imu;
        gtsam::Vector3 t_lidar_imu;
        gtsam::Vector3 init_pos_std;
        gtsam::Vector3 init_vel_std;
        gtsam::Vector3 init_rot_std;
        gtsam::Pose3 T_lidar_imu;
    };
    struct ImuRandomWalk{
        gtsam::Vector3 rot_rw;
        gtsam::Vector3 vel_rw;
        gtsam::Vector3 bg_std;
        gtsam::Vector3 ba_std;
    };

    fgo::data_types::State ImuPropagation(const fgo::data_types::State &previous_state,
                                          const fgo::data_types::IMUMeasurement &imu_start,
                                          const fgo::data_types::IMUMeasurement &imu_finish);

    void ImuCompensation(fgo::data_types::IMUMeasurement &imu_cur, const gtsam::imuBias::ConstantBias &error);

    fgo::data_types::IMUMeasurement ImuInterpolation(const rclcpp::Time &lidar_time,
                                                     fgo::data_types::IMUMeasurement &imu_pre,
                                                     fgo::data_types::IMUMeasurement &imu_current);


}