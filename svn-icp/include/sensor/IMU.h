/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 Shiping Ma and Haoming Zhang
    email: shiping.ma@tu-berlin.de and haoming.zhang@rwth-aachen.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    IMU.h
 * @brief   Definition of the IMU sensor interface
 * @author  Shiping Ma*
 * @author  Haoming Zhang
 * @date    June 22, 2025
 */

#ifndef LIO_IMU_H
#define LIO_IMU_H
#pragma once

#include "data/DataTypes.h"

namespace svnicp::sensor {
  static constexpr double G = 9.782940329221166;

  struct LIOParam {
    gtsam::Matrix3 R_lidar_imu;
    gtsam::Vector3 t_lidar_imu;
    gtsam::Vector3 init_pos_std;
    gtsam::Vector3 init_vel_std;
    gtsam::Vector3 init_rot_std;
    gtsam::Pose3 T_lidar_imu;
  };

  struct IMURandomWalk {
    gtsam::Vector3 rot_rw;
    gtsam::Vector3 vel_rw;
    gtsam::Vector3 bg_std;
    gtsam::Vector3 ba_std;
  };

  data_types::State IMUPropagation(const data_types::State &previous_state,
                                   const data_types::IMUMeasurement &imu_start,
                                   const data_types::IMUMeasurement &imu_finish);

  void IMUCompensation(data_types::IMUMeasurement &imu_cur, const gtsam::imuBias::ConstantBias &error);

  data_types::IMUMeasurement IMUInterpolation(const rclcpp::Time &lidar_time,
                                              data_types::IMUMeasurement &imu_pre,
                                              data_types::IMUMeasurement &imu_current);
}

#endif //LIO_IMU_H