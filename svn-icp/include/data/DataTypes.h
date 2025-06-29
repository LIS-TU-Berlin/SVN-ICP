/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 XXX
    email: XXX

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include <shared_mutex>

#include <rclcpp/rclcpp.hpp>
#include <gtsam/base/Vector.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/linear/NoiseModel.h>

#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<torch/torch.h>

namespace svnicp::data_types{

    using Point_t = pcl::PointXYZI;
    using Cloud_t = pcl::PointCloud<Point_t>;
    using Device_type = c10::DeviceType;
    using at::indexing::Slice;

    struct Pose
    {
        rclcpp::Time timestamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
        gtsam::Pose3 pose;
        // should be a vector with 6 entries
        gtsam::Matrix66 poseVar = (gtsam::Vector6() <<
                                                    5.0/180 * M_PI, 5.0/180 * M_PI, 5.0/180 * M_PI, 5, 5, 10).finished().asDiagonal();

        std::shared_mutex mutex;

        Pose() = default;
        inline Pose& operator =(const Pose& ori)
        {
          timestamp = ori.timestamp;
          pose = ori.pose;
          poseVar = ori.poseVar;
          return *this;
        }

        Pose(const Pose& ori) {
          timestamp = ori.timestamp;
          pose = ori.pose;
          poseVar = ori.poseVar;
        }
    };

    struct State
    {
        rclcpp::Time timestamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
        gtsam::NavState state{};
        // should be a vector with 6 entries
        gtsam::Matrix66 poseVar = (gtsam::Vector6() <<
                5.0/180 * M_PI, 5.0/180 * M_PI, 5.0/180 * M_PI, 5, 5, 10).finished().asDiagonal();

        gtsam::Matrix33 velVar = gtsam::I_3x3;    // should be a vector with 6 entries, including omega

        gtsam::imuBias::ConstantBias imuBias;
        gtsam::Matrix66 imuBiasVar = (gtsam::Matrix66() << 0.1 * gtsam::I_3x3, gtsam::Z_3x3,
                                      gtsam::Z_3x3, 0.01/180 * M_PI * gtsam::I_3x3).finished(); // should be a vector with 6 entries acc then gyro

        gtsam::Vector2 cbd{};
        // should be a vector with 2 entries
        gtsam::Matrix22 cbdVar = (gtsam::Matrix22() << 1000,0,0,10).finished();

        gtsam::Vector ddIntAmb = gtsam::Vector1(0);
        gtsam::Matrix ddIntAmbVar = gtsam::I_1x1;

        gtsam::Vector3 omega{};
        gtsam::Matrix33 omegaVar = 0.1/180*M_PI * gtsam::I_3x3;

        gtsam::Vector6 accMeasured{};
        //gtsam::Matrix66 accVar = 0.05 * gtsam::I_6x6;

        std::shared_mutex mutex;

        State() = default;


        inline State& operator =(const State& ori)
        {
          timestamp = ori.timestamp;
          state = ori.state;
          poseVar = ori.poseVar;
          velVar = ori.velVar;
          imuBias = ori.imuBias;
          imuBiasVar = ori.imuBiasVar;
          cbd = ori.cbd;
          cbdVar = ori.cbdVar;
          ddIntAmb = ori.ddIntAmb;
          ddIntAmbVar = ori.ddIntAmbVar;
          omega = ori.omega;
          omegaVar = ori.omegaVar;
          accMeasured = ori.accMeasured;
          //accVar = ori.accVar;
          return *this;
        }

        State(const State& ori) {
          timestamp = ori.timestamp;
          state = ori.state;
          poseVar = ori.poseVar;
          velVar = ori.velVar;
          imuBias = ori.imuBias;
          imuBiasVar = ori.imuBiasVar;
          cbd = ori.cbd;
          cbdVar = ori.cbdVar;
          ddIntAmb = ori.ddIntAmb;
          ddIntAmbVar = ori.ddIntAmbVar;
          omega = ori.omega;
          accMeasured = ori.accMeasured;
         // accVar = ori.accVar;
        }
    };

  struct PPS {
        std::atomic_uint_fast64_t counter;
        std::atomic_int_fast64_t localDelay;  // in milliseconds
        rclcpp::Time lastPPSTime;
    };

    struct IMUMeasurement {
        rclcpp::Time timestamp{0};  // timestamp of current sensor meas
        double dt{};  // dt between consequent meas
        gtsam::Quaternion AHRSOri{};
        gtsam::Vector9 AHRSOriCov{};
        gtsam::Vector3 accLin{};
        gtsam::Vector9 accLinCov{};
        gtsam::Vector3 velRot{};
        gtsam::Vector9 accRotCov{};
        gtsam::Vector3 gyro{};
        gtsam::Vector9 gyroCov{};
        gtsam::Vector3 mag{};
        gtsam::Vector9 magCov{};
    };

     struct Odom
     {
         size_t frameIndexCurrent;
         size_t frameIndexPrevious;
         double timestampPrevious;
         double timestampCurrent;
         gtsam::Pose3 poseFromLocalWorld;
         gtsam::Pose3 poseFromECEF;
         gtsam::Pose3 poseToLocalWorld;
         gtsam::Pose3 poseToECEF;
         gtsam::Pose3 poseRelativeECEF;
         gtsam::Vector6 noise;
     };

}