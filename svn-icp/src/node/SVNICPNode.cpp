/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 Shiping Ma and Haoming Zhang
    email: shiping.ma@tu-berlin.de and haoming.zhang@rwth-aachen.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    SVNICPNode.cpp
 * @brief   SVN-ICP excuteable
 * @author  Shiping Ma*
 * @author  Haoming Zhang
 * @date    June 22, 2025
 */

#include <rclcpp/rclcpp.hpp>
#include "core/OdometryPipeline.h"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    const auto node = std::make_shared<svnicp::OdometryPipeline>();
    static constexpr size_t THREAD_NUM = 7;
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), THREAD_NUM);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();

    return 0;
}