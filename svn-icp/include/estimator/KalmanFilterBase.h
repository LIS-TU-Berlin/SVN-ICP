/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 XXX
    email: XXX

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    KalmanFilterBase.h
 * @brief   Definition of the Kalman filter basic functions
 * @author  UNKNOWN
 * @author  UNKNOWN
 * @date    June 22, 2025
 */
#ifndef KALMANFILTERBASE_H
#define KALMANFILTERBASE_H

#include "data/DataTypes.h"
#include "sensor/IMU.h"

namespace svnicp::estimator {
  class KalmanFilterBase {
  public:
    virtual ~KalmanFilterBase() = default;

    explicit KalmanFilterBase(const sensor::LIOParam &lio_param, const sensor::IMURandomWalk &imu_rw);

    KalmanFilterBase();

    virtual void predict(svnicp::data_types::State &body_state,
                         const svnicp::data_types::IMUMeasurement &imu_previous,
                         svnicp::data_types::IMUMeasurement &imu_current) = 0;

    virtual void update(svnicp::data_types::State &body_state,
                        const gtsam::Matrix6 &H,
                        const gtsam::Vector6 &b) = 0;

  protected:
    sensor::LIOParam lio_param_;
    sensor::IMURandomWalk imu_rw_;
  };
}


#endif //STEIN_MICP_KALMANFILTERBASE_H
