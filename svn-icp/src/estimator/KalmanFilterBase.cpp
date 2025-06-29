/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 XXX
    email: XXX

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    KalmanFilterBase.cpp
 * @brief   Kalman Filter base
 * @author  UNKNOWN
 * @author  UNKNOWN
 * @date    June 22, 2025
 */

#include "estimator/KalmanFilterBase.h"

namespace svnicp::estimator {
  KalmanFilterBase::KalmanFilterBase(const sensor::LIOParam &lio_param, const sensor::IMURandomWalk &imu_rw) {
    lio_param_ = lio_param;
    imu_rw_ = imu_rw;
  }
}
