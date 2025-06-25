//
// Created by haoming on 19.09.24.
//

#ifndef STEIN_MICP_KALMANFILTERBASE_H
#define STEIN_MICP_KALMANFILTERBASE_H

#include "data/DataTypes.h"
#include "imu/imu.h"

namespace estimator {


    class Estimator
    {

    public:
        explicit Estimator(const IMU::LioParam &lio_param, const IMU::ImuRandomWalk &imu_rw);

        Estimator();

        virtual void predict(fgo::data_types::State &body_state,
                             const fgo::data_types::IMUMeasurement &imu_previous,
                             fgo::data_types::IMUMeasurement &imu_current) = 0;

        virtual void update(fgo::data_types::State &body_state,
                            const gtsam::Matrix6 &H,
                            const gtsam::Vector6 &b)  = 0;


    protected:
        IMU::LioParam lio_param_;
        IMU::ImuRandomWalk imu_rw_;



    };



}






#endif //STEIN_MICP_KALMANFILTERBASE_H
