//
// Created by haoming on 27.06.23.
//


#include <rclcpp/rclcpp.hpp>
#include "OdometryPipeline.h"

int main(int argc, char *argv[])
{

    rclcpp::init(argc, argv);
    auto node = std::make_shared<stein_icp::SteinICPOdometry>();
    static const size_t THREAD_NUM = 7;
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), THREAD_NUM);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();

    return 0;
}