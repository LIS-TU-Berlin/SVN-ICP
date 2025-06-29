#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_
#pragma once

#include <Eigen/Eigen>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <array>
#include <cstdint>

#define PI 3.14159265

using namespace std;

typedef pcl::PointXYZI PointType;

struct smoothness_t {
  float value;
  size_t ind;
};

struct by_value {
  bool operator()(smoothness_t const &left, smoothness_t const &right) {
    return left.value < right.value;
  }
};

/*
    * A point cloud type that has "ring" channel
    */
struct PointXYZIR {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY

  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
                                  (float, x, x) (float, y, y)
                                  (float, z, z) (float, intensity, intensity)
                                  (uint16_t, ring, ring)
)

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY

  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x) (float, y, y)
                                  (float, z, z) (float, intensity, intensity)
                                  (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                  (double, time, time)
)

typedef PointXYZIRPYT PointTypePose;

#endif
