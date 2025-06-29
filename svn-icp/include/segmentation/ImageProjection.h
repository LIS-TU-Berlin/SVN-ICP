// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#ifndef STEINICP_SEGMENTATION_H
#define STEINICP_SEGMENTATION_H

#include "SegUtils.h"

class ImageProjection {
private:
  // Using velodyne cloud "ring" channel for image projection (other lidar may have different name for this channel, change "PointXYZIR" below)
  // In "PointXYZIR", type of ring of Ouster sensor should be uint_8t, otherwise uint_16t
  const bool useCloudRing = false; // if true, ang_res_y and ang_bottom are not used

  // VLP-16
  //    const int N_SCAN = 16;
  //    const int Horizon_SCAN = 1800;
  //    const float ang_res_x = 0.2;
  //    const float ang_res_y = 2.0;
  //    const float ang_bottom = 15.0+0.1;
  //    const int groundScanInd = 7;

  // HDL-32E
  // extern const int N_SCAN = 32;
  // extern const int Horizon_SCAN = 1800;
  // extern const float ang_res_x = 360.0/float(Horizon_SCAN);
  // extern const float ang_res_y = 41.33/float(N_SCAN-1);
  // extern const float ang_bottom = 30.67;
  // extern const int groundScanInd = 20;

  //HDL-64E
  const int N_SCAN = 64;
  const int Horizon_SCAN = 2250;
  const float ang_res_x = 360.0 / float(Horizon_SCAN);
  const float ang_res_y = 26.8 / float(N_SCAN - 1);
  const float ang_bottom = 24.8;
  const int groundScanInd = 7;

  // VLS-128
  // const int N_SCAN = 128;
  // const int Horizon_SCAN = 1800;
  // const float ang_res_x = 0.2;
  // const float ang_res_y = 0.3;
  // const float ang_bottom = 25.0;
  // const int groundScanInd = 10;

  // RS-lidar-32
  //    const int N_SCAN = 32;
  //    const int Horizon_SCAN = 2000;
  //    const float ang_res_x = 0.18;
  //    const float ang_res_y = 40 / static_cast<float>((N_SCAN - 1));
  //    const float ang_bottom = 25.0;
  //    const int groundScanInd = 2;

  // Ouster users may need to uncomment line 159 in imageProjection.cpp
  // Usage of Ouster sensor data is not fully supported yet (LeGO-LOAM needs 9-DOF IMU), please just publish point cloud data
  // Ouster OS1-16
  // extern const int N_SCAN = 16;
  // extern const int Horizon_SCAN = 1024;
  // extern const float ang_res_x = 360.0/float(Horizon_SCAN);
  // extern const float ang_res_y = 33.2/float(N_SCAN-1);
  // extern const float ang_bottom = 16.6+0.1;
  // extern const int groundScanInd = 7;

  // Ouster OS1-64
  // extern const int N_SCAN = 64;
  // extern const int Horizon_SCAN = 1024;
  // extern const float ang_res_x = 360.0/float(Horizon_SCAN);
  // extern const float ang_res_y = 33.2/float(N_SCAN-1);
  // extern const float ang_bottom = 16.6+0.1;
  // extern const int groundScanInd = 15;

  // Ouster OS0-128
  //const int N_SCAN = 128;
  //const int Horizon_SCAN = 1024;
  //const float ang_res_x = 360.0/float(Horizon_SCAN);
  //const float ang_res_y = 90/float(N_SCAN-1);
  //const float ang_bottom = 45+0.1;
  //const int groundScanInd = 11;

  const float sensorMinimumRange = 1.0;
  const float sensorMountAngle = 0.0;
  const float segmentTheta = 60.0 / 180.0 * M_PI; // decrese this value may improve accuracy
  const int segmentValidPointNum = 5;
  const int segmentValidLineNum = 3;
  const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
  const float segmentAlphaY = ang_res_y / 180.0 * M_PI;


  pcl::PointCloud<PointType>::Ptr laserCloudIn_;
  pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing_;

  pcl::PointCloud<PointType>::Ptr fullCloud_; // projected velodyne raw cloud, but saved in the form of 1-D matrix
  pcl::PointCloud<PointType>::Ptr fullInfoCloud_; // same as fullCloud_, but with intensity - range

  pcl::PointCloud<PointType>::Ptr groundCloud_;
  pcl::PointCloud<PointType>::Ptr segmentedCloud_;
  pcl::PointCloud<PointType>::Ptr segmentedCloudPure_;
  pcl::PointCloud<PointType>::Ptr outlierCloud_;

  PointType nanPoint; // fill in fullCloud_ at each iteration

  //    cv::Mat rangeMat_; // range matrix for range image
  //    cv::Mat labelMat_; // label matrix for segmentaiton marking
  //    cv::Mat groundMat_; // ground matrix for ground cloud marking
  Eigen::MatrixXf rangeMat_;
  Eigen::MatrixXi labelMat_;
  Eigen::MatrixXi groundMat_;

  int labelCount_;

  double startOrientation_;
  double endOrientation_;
  double orientationDiff_;

  //    std_msgs::Header cloudHeader;

  std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

  uint16_t *allPushedIndX_; // array for tracking points of a segmented object
  uint16_t *allPushedIndY_;

  uint16_t *queueIndX_; // array for breadth-first search process of segmentation, for speed
  uint16_t *queueIndY_;

public:
  ImageProjection() {
    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    allocateMemory();
    resetParameters();
  }

  void allocateMemory() {
    laserCloudIn_.reset(new pcl::PointCloud<PointType>());
    laserCloudInRing_.reset(new pcl::PointCloud<PointXYZIR>());

    fullCloud_.reset(new pcl::PointCloud<PointType>());
    fullInfoCloud_.reset(new pcl::PointCloud<PointType>());

    groundCloud_.reset(new pcl::PointCloud<PointType>());
    segmentedCloud_.reset(new pcl::PointCloud<PointType>());
    segmentedCloudPure_.reset(new pcl::PointCloud<PointType>());
    outlierCloud_.reset(new pcl::PointCloud<PointType>());

    fullCloud_->points.resize(N_SCAN * Horizon_SCAN);
    fullInfoCloud_->points.resize(N_SCAN * Horizon_SCAN);

    //        segMsg.startRingIndex.assign(N_SCAN, 0);
    //        segMsg.endRingIndex.assign(N_SCAN, 0);
    //
    //        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
    //        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
    //        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = 1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 0;
    neighbor.second = -1;
    neighborIterator.push_back(neighbor);
    neighbor.first = 1;
    neighbor.second = 0;
    neighborIterator.push_back(neighbor);

    allPushedIndX_ = new uint16_t[N_SCAN * Horizon_SCAN];
    allPushedIndY_ = new uint16_t[N_SCAN * Horizon_SCAN];

    queueIndX_ = new uint16_t[N_SCAN * Horizon_SCAN];
    queueIndY_ = new uint16_t[N_SCAN * Horizon_SCAN];
  }

  void resetParameters() {
    laserCloudIn_->clear();
    groundCloud_->clear();
    segmentedCloud_->clear();
    segmentedCloudPure_->clear();
    outlierCloud_->clear();

    //        rangeMat_ = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    rangeMat_ = -100000 * Eigen::MatrixXf::Ones(N_SCAN, Horizon_SCAN);
    //        groundMat_ = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    //        labelMat_ = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
    labelMat_ = Eigen::MatrixXi::Zero(N_SCAN, Horizon_SCAN);
    groundMat_ = Eigen::MatrixXi::Zero(N_SCAN, Horizon_SCAN);
    labelCount_ = 1;

    std::fill(fullCloud_->points.begin(), fullCloud_->points.end(), nanPoint);
    std::fill(fullInfoCloud_->points.begin(), fullInfoCloud_->points.end(), nanPoint);
  }

  ~ImageProjection() {
  }

  void copyPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr &laserCloudMsg) {
    //        cloudHeader = laserCloudMsg->header;
    // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
    pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn_);
    // Remove Nan points

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn_, *laserCloudIn_, indices);
    // have "ring" channel in the cloud
    if (useCloudRing == true) {
      pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing_);
      if (laserCloudInRing_->is_dense == false) {
        //                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
        //                ros::shutdown();
      }
    }
  }

  void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr &laserCloudMsg) {
    resetParameters();
    // 1. Convert ros message to pcl point cloud
    copyPointCloud(laserCloudMsg);
    // 2. Start and end angle of a scan
    findStartEndAngle();
    // 3. Range image projection
    projectPointCloud();
    // 4. Mark ground points
    groundRemoval();
    // 5. Point cloud segmentation
    cloudSegmentation();
    // 6. Publish all clouds
    //        publishCloud();
    // 7. Reset parameters for next iteration
    //        resetParameters();
  }

  void findStartEndAngle() {
    // start and end orientation of this cloud
    startOrientation_ = -atan2(laserCloudIn_->points[0].y, laserCloudIn_->points[0].x);
    endOrientation_ = -atan2(laserCloudIn_->points[laserCloudIn_->points.size() - 1].y,
                             laserCloudIn_->points[laserCloudIn_->points.size() - 1].x) + 2 * M_PI;
    if (endOrientation_ - startOrientation_ > 3 * M_PI) {
      endOrientation_ -= 2 * M_PI;
    } else if (endOrientation_ - startOrientation_ < M_PI)
      endOrientation_ += 2 * M_PI;
    orientationDiff_ = endOrientation_ - startOrientation_;
  }

  void projectPointCloud() {
    // range image projection
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize;
    PointType thisPoint;

    cloudSize = laserCloudIn_->points.size();

    for (size_t i = 0; i < cloudSize; ++i) {
      thisPoint.x = laserCloudIn_->points[i].x;
      thisPoint.y = laserCloudIn_->points[i].y;
      thisPoint.z = laserCloudIn_->points[i].z;
      // find the row and column index in the iamge for this point
      if (useCloudRing == true) {
        rowIdn = laserCloudInRing_->points[i].ring;
      } else {
        verticalAngle =
            atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
        rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
      }
      if (rowIdn < 0 || rowIdn >= N_SCAN)
        continue;

      horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

      columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
      if (columnIdn >= Horizon_SCAN)
        columnIdn -= Horizon_SCAN;

      if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
        continue;

      range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
      if (range < sensorMinimumRange)
        continue;

      //            rangeMat_.at<float>(rowIdn, columnIdn) = range;
      rangeMat_(rowIdn, columnIdn) = range;
      thisPoint.intensity = (float) rowIdn + (float) columnIdn / 10000.0;

      index = columnIdn + rowIdn * Horizon_SCAN;
      fullCloud_->points[index] = thisPoint;
      fullInfoCloud_->points[index] = thisPoint;
      fullInfoCloud_->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
    }
  }


  void groundRemoval() {
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    // groundMat_
    // -1, no valid info to check if ground of not
    //  0, initial value, after validation, means not ground
    //  1, ground
    for (size_t j = 0; j < Horizon_SCAN; ++j) {
      for (size_t i = 0; i < groundScanInd; ++i) {
        lowerInd = j + (i) * Horizon_SCAN;
        upperInd = j + (i + 1) * Horizon_SCAN;

        if (fullCloud_->points[lowerInd].intensity == -1 ||
            fullCloud_->points[upperInd].intensity == -1) {
          // no info to check, invalid points
          groundMat_(i, j) = -1;
          continue;
        }

        diffX = fullCloud_->points[upperInd].x - fullCloud_->points[lowerInd].x;
        diffY = fullCloud_->points[upperInd].y - fullCloud_->points[lowerInd].y;
        diffZ = fullCloud_->points[upperInd].z - fullCloud_->points[lowerInd].z;

        angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

        if (abs(angle - sensorMountAngle) <= 10) {
          groundMat_(i, j) = 1;
          groundMat_(i + 1, j) = 1;
        }
      }
    }
    // extract ground cloud (groundMat_ == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~N_SCAN-1, need rangeMat_ for mark label matrix for the 16th scan
    for (size_t i = 0; i < N_SCAN; ++i) {
      for (size_t j = 0; j < Horizon_SCAN; ++j) {
        if (groundMat_(i, j) == 1 || rangeMat_(i, j) == -100000) {
          labelMat_(i, j) = -1;
        }
      }
    }

    for (size_t i = 0; i <= groundScanInd; ++i) {
      for (size_t j = 0; j < Horizon_SCAN; ++j) {
        if (groundMat_(i, j) == 1)
          groundCloud_->push_back(fullCloud_->points[j + i * Horizon_SCAN]);
      }
    }
  }

  void cloudSegmentation() {
    // segmentation process
    for (size_t i = 0; i < N_SCAN; ++i)
      for (size_t j = 0; j < Horizon_SCAN; ++j)
        if (labelMat_(i, j) == 0)
          labelComponents(i, j);

    int sizeOfSegCloud = 0;
    // extract segmented cloud for lidar odometry
    for (size_t i = 0; i < N_SCAN; ++i) {
      //            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

      for (size_t j = 0; j < Horizon_SCAN; ++j) {
        if (labelMat_(i, j) > 0 || groundMat_(i, j) == 1) {
          // outliers that will not be used for optimization (always continue)
          if (labelMat_(i, j) == 999999) {
            if (i > groundScanInd && j % 5 == 0) {
              outlierCloud_->push_back(fullCloud_->points[j + i * Horizon_SCAN]);
              continue;
            } else {
              continue;
            }
          }
          // majority of ground points are skipped
          if (groundMat_(i, j) == 1) {
            if (j % 5 != 0 && j > 5 && j < Horizon_SCAN - 5)
              continue;
          }
          // mark ground points so they will not be considered as edge features later
          //                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat_.at<int8_t>(i, j) == 1);
          //                    // mark the points' column index for marking occlusion later
          //                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
          //                    // save range info
          //                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat_.at<float>(i, j);
          // save seg cloud
          segmentedCloud_->push_back(fullCloud_->points[j + i * Horizon_SCAN]);
          // size of seg cloud
          ++sizeOfSegCloud;
        }
      }

      //            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
    }

    // extract segmented cloud for visualization

    for (size_t i = 0; i < N_SCAN; ++i) {
      for (size_t j = 0; j < Horizon_SCAN; ++j) {
        if (labelMat_(i, j) > 0 && labelMat_(i, j) != 999999) {
          segmentedCloudPure_->push_back(fullCloud_->points[j + i * Horizon_SCAN]);
          segmentedCloudPure_->points.back().intensity = labelMat_(i, j);
        }
      }
    }
  }

  void labelComponents(int row, int col) {
    // use std::queue std::vector std::deque will slow the program down greatly
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY;
    //        bool lineCountFlag[N_SCAN] = {false};
    vector<bool> lineCountFlag(N_SCAN, false);

    queueIndX_[0] = row;
    queueIndY_[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX_[0] = row;
    allPushedIndY_[0] = col;
    int allPushedIndSize = 1;

    while (queueSize > 0) {
      // Pop point
      fromIndX = queueIndX_[queueStartInd];
      fromIndY = queueIndY_[queueStartInd];
      --queueSize;
      ++queueStartInd;
      // Mark popped point
      labelMat_(fromIndX, fromIndY) = labelCount_;
      // Loop through all the neighboring grids of popped grid
      for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter) {
        // new index
        thisIndX = fromIndX + (*iter).first;
        thisIndY = fromIndY + (*iter).second;
        // index should be within the boundary
        if (thisIndX < 0 || thisIndX >= N_SCAN)
          continue;
        // at range image margin (left or right side)
        if (thisIndY < 0)
          thisIndY = Horizon_SCAN - 1;
        if (thisIndY >= Horizon_SCAN)
          thisIndY = 0;
        // prevent infinite loop (caused by put already examined point back)
        if (labelMat_(thisIndX, thisIndY) != 0)
          continue;

        //                d1 = std::max(rangeMat_.at<float>(fromIndX, fromIndY),
        //                              rangeMat_.at<float>(thisIndX, thisIndY));
        //                d2 = std::min(rangeMat_.at<float>(fromIndX, fromIndY),
        //                              rangeMat_.at<float>(thisIndX, thisIndY));

        d1 = std::max(rangeMat_(fromIndX, fromIndY),
                      rangeMat_(thisIndX, thisIndY));
        d2 = std::min(rangeMat_(fromIndX, fromIndY),
                      rangeMat_(thisIndX, thisIndY));

        if ((*iter).first == 0)
          alpha = segmentAlphaX;
        else
          alpha = segmentAlphaY;

        angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

        if (angle > segmentTheta) {
          queueIndX_[queueEndInd] = thisIndX;
          queueIndY_[queueEndInd] = thisIndY;
          ++queueSize;
          ++queueEndInd;

          labelMat_(thisIndX, thisIndY) = labelCount_;
          lineCountFlag[thisIndX] = true;

          allPushedIndX_[allPushedIndSize] = thisIndX;
          allPushedIndY_[allPushedIndSize] = thisIndY;
          ++allPushedIndSize;
        }
      }
    }

    // check if this segment is valid
    bool feasibleSegment = false;
    if (allPushedIndSize >= 30)
      feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum) {
      int lineCount = 0;
      for (size_t i = 0; i < N_SCAN; ++i)
        if (lineCountFlag[i] == true)
          ++lineCount;
      if (lineCount >= segmentValidLineNum)
        feasibleSegment = true;
    }
    // segment is valid, mark these points
    if (feasibleSegment == true) {
      ++labelCount_;
    } else {
      // segment is invalid, mark these points
      for (size_t i = 0; i < allPushedIndSize; ++i) {
        labelMat_(allPushedIndX_[i], allPushedIndY_[i]) = 999999;
      }
    }
  }

  pcl::PointCloud<PointType>::Ptr GetSegmentedCloudPure() {
    return segmentedCloud_;
  }


  //    void publishCloud(){
  //        // 1. Publish Seg Cloud Info
  //        segMsg.header = cloudHeader;
  //        pubSegmentedCloudInfo.publish(segMsg);
  //        // 2. Publish clouds
  //        sensor_msgs::PointCloud2 laserCloudTemp;
  //
  //        pcl::toROSMsg(*outlierCloud_, laserCloudTemp);
  //        laserCloudTemp.header.stamp = cloudHeader.stamp;
  //        laserCloudTemp.header.frame_id = "base_link";
  //        pubOutlierCloud.publish(laserCloudTemp);
  //        // segmented cloud with ground
  //        pcl::toROSMsg(*segmentedCloud_, laserCloudTemp);
  //        laserCloudTemp.header.stamp = cloudHeader.stamp;
  //        laserCloudTemp.header.frame_id = "base_link";
  //        pubSegmentedCloud.publish(laserCloudTemp);
  //        // projected full cloud
  //        if (pubFullCloud.getNumSubscribers() != 0){
  //            pcl::toROSMsg(*fullCloud_, laserCloudTemp);
  //            laserCloudTemp.header.stamp = cloudHeader.stamp;
  //            laserCloudTemp.header.frame_id = "base_link";
  //            pubFullCloud.publish(laserCloudTemp);
  //        }
  //        // original dense ground cloud
  //        if (pubGroundCloud.getNumSubscribers() != 0){
  //            pcl::toROSMsg(*groundCloud_, laserCloudTemp);
  //            laserCloudTemp.header.stamp = cloudHeader.stamp;
  //            laserCloudTemp.header.frame_id = "base_link";
  //            pubGroundCloud.publish(laserCloudTemp);
  //        }
  //        // segmented cloud without ground
  //        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
  //            pcl::toROSMsg(*segmentedCloudPure_, laserCloudTemp);
  //            laserCloudTemp.header.stamp = cloudHeader.stamp;
  //            laserCloudTemp.header.frame_id = "base_link";
  //            pubSegmentedCloudPure.publish(laserCloudTemp);
  //        }
  //        // projected full cloud info
  //        if (pubFullInfoCloud.getNumSubscribers() != 0){
  //            pcl::toROSMsg(*fullInfoCloud_, laserCloudTemp);
  //            laserCloudTemp.header.stamp = cloudHeader.stamp;
  //            laserCloudTemp.header.frame_id = "base_link";
  //            pubFullInfoCloud.publish(laserCloud
};


#endif
