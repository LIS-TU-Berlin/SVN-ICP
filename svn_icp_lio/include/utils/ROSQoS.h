//
// Created by haoming on 20.07.22.
//

#ifndef ONLINE_FGO_ROSQOS_H
#define ONLINE_FGO_ROSQOS_H

#pragma once

#include <rclcpp/rclcpp.hpp>

namespace utils::ros
{
    inline rmw_qos_profile_t qos_profile{
        RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        1,
        RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        RMW_QOS_POLICY_DURABILITY_VOLATILE,
        RMW_QOS_DEADLINE_DEFAULT,
        RMW_QOS_LIFESPAN_DEFAULT,
        RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
        RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
        false
    };

    inline auto QoSGeneral = rclcpp::QoS(
        rclcpp::QoSInitialization(
            qos_profile.history,
            qos_profile.depth
        ),
        qos_profile);

    inline rmw_qos_profile_t qos_profile_imu{
        RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        2000,
        RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        RMW_QOS_POLICY_DURABILITY_VOLATILE,
        RMW_QOS_DEADLINE_DEFAULT,
        RMW_QOS_LIFESPAN_DEFAULT,
        RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
        RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
        false
    };

    inline auto QoSIMU = rclcpp::QoS(
        rclcpp::QoSInitialization(
            qos_profile_imu.history,
            qos_profile_imu.depth
        ),
        qos_profile_imu);

    inline rmw_qos_profile_t qos_profile_lidar{
        RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        5,
        RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        RMW_QOS_POLICY_DURABILITY_VOLATILE,
        RMW_QOS_DEADLINE_DEFAULT,
        RMW_QOS_LIFESPAN_DEFAULT,
        RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
        RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
        false
    };

    inline auto QoSLiDAR = rclcpp::QoS(
        rclcpp::QoSInitialization(
            qos_profile_lidar.history,
            qos_profile_lidar.depth
        ),
        qos_profile_lidar);

}
#endif //ONLINE_FGO_ROSQOS_H
