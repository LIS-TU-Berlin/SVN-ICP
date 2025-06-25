//
// Created by haoming on 19.07.22.
//
#pragma once

#ifndef ONLINE_FGO_ROSBASE_H
#define ONLINE_FGO_ROSBASE_H

#include <rclcpp/rclcpp.hpp>

namespace utils::ros
{
    class ROSBase
    {
    protected:
        rclcpp::Node::SharedPtr nodePtr_;
    };

}
#endif //ONLINE_FGO_ROSBASE_H
