//
// Created by haoming on 01.06.23.
//

#ifndef STEIN_ICP_TYPES_H
#define STEIN_ICP_TYPES_H

#pragma once

#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<torch/torch.h>


using Point_t = pcl::PointXYZI;
using Cloud_t = pcl::PointCloud<Point_t>;
using Device_type = c10::DeviceType;
using at::indexing::Slice;



#endif //STEIN_ICP_TYPES_H
