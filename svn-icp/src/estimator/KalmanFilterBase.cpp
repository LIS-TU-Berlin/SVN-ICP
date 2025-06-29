/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 Shiping Ma and Haoming Zhang
    email: shiping.ma@tu-berlin.de and haoming.zhang@rwth-aachen.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    .cpp
 * @brief   Definies the odometry pipline that serves as the estimator and data interface for the ICP
 * @author  Shiping Ma
 * @author  Haoming Zhang*
 * @date    June 22, 2025
 */

#include "estimator/KalmanFilterBase.h"

namespace svnicp::estimator {
  KalmanFilterBase::KalmanFilterBase(const sensor::LIOParam &lio_param, const sensor::IMURandomWalk &imu_rw) {
    lio_param_ = lio_param;
    imu_rw_ = imu_rw;
  }
}
