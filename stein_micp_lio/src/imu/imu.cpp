//
// Created by haoming on 30.09.24.
//
#include "imu.h"

namespace IMU{

    fgo::data_types::State ImuPropagation(const fgo::data_types::State &previous_state,
                                          const fgo::data_types::IMUMeasurement &imu_start,
                                          const fgo::data_types::IMUMeasurement &imu_finish) {

        fgo::data_types::State propagated_state;
        gtsam::Vector3 d_vfn, d_vgn, d_vfb;
        gtsam::Vector3 imu1_dvel, imu1_dtheta, imu2_dvel, imu2_dtheta;

        imu2_dvel = imu_finish.accLin * imu_finish.dt;
        imu2_dtheta = imu_finish.velRot * imu_finish.dt;
        imu1_dvel = imu_start.accLin * imu_start.dt;
        imu1_dtheta = imu_start.velRot * imu_start.dt;

        gtsam::Vector3 d_vfb1 = imu2_dtheta.cross(imu2_dvel) / 2;
        gtsam::Vector3 d_vfb2 = imu1_dtheta.cross(imu2_dvel) / 12;
        gtsam::Vector3 d_vfb3 = imu1_dvel.cross(imu2_dtheta) / 12;

//        d_vfb = imu2_dvel + d_vfb1 + d_vfb2 + d_vfb3;
        d_vfb = imu1_dvel;
        d_vfn = previous_state.state.pose().rotation().matrix() * d_vfb;
        gtsam::Vector3 g;
        g << 0, 0, -G; // minus for z-up
        d_vgn =  previous_state.state.pose().rotation().matrix() * g * imu_finish.dt;

        // update velocity
        gtsam::Vector3 vel = previous_state.state.velocity() + d_vfn + d_vgn;
//        gtsam::Vector3 avg_vel = 0.5 * (vel + previous_state.state.velocity());

        //update translation
//        gtsam::Vector3 t_propagate = previous_state.state.position() + avg_vel*imu_finish.dt;
//        std::cout << "position\n " << previous_state.state.t() <<std::endl;
        gtsam::Vector3  t_propagate = previous_state.state.t() + previous_state.state.v()*imu_finish.dt +
                                      0.5 * g * imu_finish.dt*imu_finish.dt + 0.5 * d_vfn * imu_finish.dt;
//        gtsam::Vector3 rot_vector = imu2_dtheta + imu1_dtheta.cross(imu2_dtheta) / 12;
        gtsam::Vector3 rot_vector = imu_start.velRot * imu_finish.dt;
        //update rotation
        gtsam::Matrix3 R_propagate = previous_state.state.R() * gtsam::Rot3::RzRyRx(rot_vector).matrix();

        propagated_state.state = gtsam::NavState(gtsam::Rot3(R_propagate), t_propagate, vel);

        return propagated_state;

    }

    void ImuCompensation(fgo::data_types::IMUMeasurement &imu_cur, const gtsam::imuBias::ConstantBias &error)
    {
        imu_cur.accLin -= error.accelerometer();
        imu_cur.velRot -= error.gyroscope();
    }

    fgo::data_types::IMUMeasurement ImuInterpolation(const rclcpp::Time &lidar_time,
                                                     fgo::data_types::IMUMeasurement &imu_pre,
                                                     fgo::data_types::IMUMeasurement &imu_current)
    {
        double time_ratio = (lidar_time.seconds() - imu_pre.timestamp.seconds()) /
                            (imu_current.timestamp.seconds() - imu_pre.timestamp.seconds());
        fgo::data_types::IMUMeasurement mid_imu;
        mid_imu.timestamp = lidar_time;
        mid_imu.velRot = (1-time_ratio) * imu_pre.velRot + time_ratio * imu_current.velRot;
        mid_imu.accLin = (1-time_ratio) * imu_pre.accLin + time_ratio * imu_current.accLin;

        imu_pre.dt = lidar_time.seconds() - imu_pre.timestamp.seconds();
        imu_current.dt = imu_current.timestamp.seconds() - lidar_time.seconds();

        return mid_imu;
    }


}