//
// Created by haoming on 19.09.24.
//

#include <Estimator.h>

namespace estimator{

    Estimator::Estimator(const IMU::LioParam &lio_param, const IMU::ImuRandomWalk &imu_rw)
    {
        lio_param_ = lio_param;
        imu_rw_ = imu_rw;
    }

}