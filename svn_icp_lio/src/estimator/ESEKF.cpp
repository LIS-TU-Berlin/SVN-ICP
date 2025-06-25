//
// Created by haoming on 24.09.24.
//
#include <ESEKF.h>

namespace estimator {

    ErrorStateKalmanFilter::ErrorStateKalmanFilter(const IMU::LioParam &lio_param,
                                                   const IMU::ImuRandomWalk &imu_rw)
            : Estimator(lio_param, imu_rw) {
        state_ = gtsam::Vector15::Zero();
        Cov_.block<3, 3>(0, 0) = lio_param_.init_pos_std.cwiseAbs2().asDiagonal();
        Cov_.block<3, 3>(3, 3) = lio_param_.init_vel_std.cwiseAbs2().asDiagonal();
        Cov_.block<3, 3>(6, 6) = lio_param_.init_rot_std.cwiseAbs2().asDiagonal();
        Cov_.block<3, 3>(9, 9) = imu_rw_.bg_std.cwiseAbs2().asDiagonal();
        Cov_.block<3, 3>(12, 12) = imu_rw_.ba_std.cwiseAbs2().asDiagonal();
        Observation_Matrix_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        Observation_Matrix_.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
        Process_Noise_ = Eigen::Matrix<double, 15, 15>::Zero();
    }

    void ErrorStateKalmanFilter::predict(fgo::data_types::State &body_state,
                                         const fgo::data_types::IMUMeasurement &imu_previous,
                                         fgo::data_types::IMUMeasurement &imu_current) {
        fgo::data_types::State body_state_previous = body_state;
        IMU::ImuCompensation(imu_current, body_state.imuBias);
        body_state = IMU::ImuPropagation(body_state, imu_previous, imu_current);

        // State Vector [dp, dv, dlog(R), dbg, dba]
        Eigen::Matrix<double, 15, 15> State_Transition;

        // State transition
        double dt = imu_current.dt;
        State_Transition = Eigen::Matrix<double, 15, 15>::Identity();
        State_Transition.block<3, 3>(0, 3) = gtsam::Matrix33::Identity() * dt;
        State_Transition.block<3, 3>(3, 6) =
                gtsam::Rot3::Rodrigues(body_state_previous.state.R() * imu_current.accLin).matrix() * dt;
        State_Transition.block<3, 3>(3, 12) = body_state_previous.state.R() * dt;
        State_Transition.block<3, 3>(6, 9) = -body_state_previous.state.R() * dt;

        // Process Noise
        Process_Noise_.setZero();
        Eigen::Matrix<double, 3, 3> R = body_state_previous.state.R();
        Process_Noise_.block<3, 3>(3, 3) = R * imu_rw_.vel_rw.cwiseAbs2().asDiagonal() * R.transpose() * dt;
        Process_Noise_.block<3, 3>(6, 6) = R * imu_rw_.rot_rw.cwiseAbs2().asDiagonal() * R.transpose() * dt;
        Process_Noise_.block<3, 3>(9, 9) = 2 * imu_rw_.bg_std.cwiseAbs2().asDiagonal() * dt;
        Process_Noise_.block<3, 3>(12, 12) = 2 * imu_rw_.ba_std.cwiseAbs2().asDiagonal() * dt;

        //State prediction
//        state_ = State_Transition * state_;
//        std::cout << "error state \n" << state_ << std::endl;
        Cov_ = State_Transition * Cov_ * State_Transition.transpose() + Process_Noise_;

//        Eigen::Vector3d initial_t = body_state.state.t() + state_(Eigen::seq(0,2));
//        auto initial_R = body_state.state.pose().rotation() * gtsam::Rot3::RzRyRx(state_(Eigen::seq(6,8)));
//        initial_guess_ = gtsam::Pose3(initial_R, initial_t);
        initial_guess_ = gtsam::Pose3(body_state.state.pose().rotation(), body_state.state.t());

    }

    void ErrorStateKalmanFilter::update(fgo::data_types::State &body_state,
                                        const gtsam::Matrix6 &SteinICP_Cov,
                                        const gtsam::Vector6 &icp_correction) {
        // Kalman gain
        Eigen::Matrix<double, 15, 6> KF_Gain_ = Cov_ * Observation_Matrix_.transpose() *
                                               (Observation_Matrix_ * Cov_ * Observation_Matrix_.transpose()
                                               +SteinICP_Cov).inverse();

        // Innovation
        Eigen::Matrix<double, 15, 1> KF_correction = KF_Gain_ * icp_correction;
        Cov_ = Cov_ - KF_Gain_ * Observation_Matrix_ * Cov_;

        gtsam::Vector3 t_error = KF_correction(Eigen::seq(0, 2));
        gtsam::Vector3 phi_error = KF_correction(Eigen::seq(6, 8));
        gtsam::Rot3 Rot_error = gtsam::Rot3::Expmap(phi_error);
        error_state_pose_ = gtsam::Pose3(Rot_error, t_error);

        auto last_pose = body_state.state.pose();
        // T*T_correction
        gtsam::Pose3 updated_pose = gtsam::Pose3(body_state.state.pose().matrix() * error_state_pose_.matrix());
        // v + R*v_correction
        gtsam::Vector3 updated_velocity = body_state.state.v() + updated_pose.rotation().matrix() * KF_correction(Eigen::seq(3,5));
        gtsam::Vector3 AccBias = body_state.imuBias.accelerometer() + KF_correction(Eigen::seq(13, 15));
        gtsam::Vector3 GyrBias = body_state.imuBias.gyroscope() + KF_correction(Eigen::seq(10, 12));

        body_state.imuBias = gtsam::imuBias::ConstantBias(AccBias, GyrBias);
        body_state.state = gtsam::NavState(updated_pose, updated_velocity);
    }

    gtsam::Pose3 ErrorStateKalmanFilter::get_initial_guess() {
        return initial_guess_;
    }

    gtsam::Pose3 ErrorStateKalmanFilter::get_state_matrix() {
        return error_state_pose_;
    }

    Eigen::Matrix<double, 15, 6> ErrorStateKalmanFilter::get_KFGain() {
        return KF_Gain_;
    }




}