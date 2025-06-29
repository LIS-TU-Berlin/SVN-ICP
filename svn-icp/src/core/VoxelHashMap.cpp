/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 XXX
    email: XXX

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    VoxelHashMap.cpp
 * @brief   VoxelHashMap
 * @author  UNKNOWN
 * @author  UNKNOWN
 * @date    June 22, 2025
 */

#include <pcl/filters/voxel_grid.h>
#include "core/VoxelHashMap.h"


namespace svnicp {
  void VoxelHashMap::AddPointCloud(const pcl::PointCloud<Point_t> &new_cloud, const gtsam::Pose3 &new_pose) {
    const auto new_pose_matrix = new_pose.matrix();
    pcl::PointCloud<Point_t> cloud_in_map;
    pcl::transformPointCloud(new_cloud, cloud_in_map, new_pose_matrix);
    const auto &current_pos = new_pose.translation();

    for (const auto &point: cloud_in_map.points) {
      const auto voxel_idx = VoxelIdx((point.getVector3fMap() / voxel_size_).cast<int>());
      if (const auto search = map_.find(voxel_idx); search != map_.end()) {
        if (auto &voxel_cloud = search.value(); voxel_cloud.size() < max_pointscount_)
          voxel_cloud.push_back(point);
      } else {
        pcl::PointCloud<Point_t> new_voxel;
        new_voxel.push_back(point);
        map_.insert({voxel_idx, new_voxel});
      }
    }

    this->RemoveFarPointCloud(current_pos);
  }

  pcl::PointCloud<Point_t> VoxelHashMap::GetMap() {
    pcl::PointCloud<Point_t> map_cloud;
    for (const auto &[voxel_idx, voxel_cloud]: map_) {
      map_cloud += voxel_cloud;
    }
    return map_cloud;
  }

  pcl::PointCloud<Point_t> VoxelHashMap::GetMap(const gtsam::Pose3 &pose, const double &max_range) {
    pcl::PointCloud<Point_t> map_cloud;
    const auto &current_position = pose.translation();
    for (const auto &[voxel_idx, voxel_cloud]: map_) {
      const auto &pt = voxel_cloud.front();
      if (const auto distance_square = (pt.getVector3fMap().cast<double>() - current_position).squaredNorm();
        distance_square < max_range * max_range)
        map_cloud += voxel_cloud;
    }
    return map_cloud;
  }

  pcl::PointCloud<Point_t> VoxelHashMap::GetNeighbourMap(const pcl::PointCloud<Point_t> &source_cloud) {
    pcl::PointCloud<Point_t> neighbour_cloud;
    tsl::robin_map<VoxelIdx, pcl::PointCloud<Point_t>, VoxelHash> neighbour_map;
    for (const auto &point: source_cloud.points) {
      const auto voxel_idx = VoxelIdx((point.getVector3fMap() / voxel_size_).cast<int>());

      for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
          for (int k = -1; k < 2; k++) {
            const auto tmp_idx = VoxelIdx(voxel_idx[0] + i, voxel_idx[1] + j, voxel_idx[2] + k);
            if (const auto map_search = map_.find(tmp_idx); map_search != map_.end()) {
              if (neighbour_map.contains(tmp_idx))
                continue;
              neighbour_map.insert({tmp_idx, map_search.value()});
            }
          }
        }
      }
    }
    if (neighbour_map.empty()) {
      std::cout << "!!!!!!!!!!!!!!!!! No Neighbour Found !!!!!!!!!!!!!!!!" << std::endl;
      //            neighbour_map = map_;
    }
    for (const auto &[voxel, voxel_cloud]: neighbour_map) {
      neighbour_cloud += voxel_cloud;
    }

    return neighbour_cloud;
  }

  void VoxelHashMap::RemoveFarPointCloud(const Eigen::Vector3d &current_position) {
    for (const auto &[voxel_idx, voxel_cloud]: map_) {
      const auto pt = voxel_cloud.front();
      if (const auto distance_square = (pt.getVector3fMap().cast<double>() - current_position).squaredNorm();
        distance_square > max_range_ * max_range_) {
        map_.erase(voxel_idx);
      }
    }
  }

  pcl::PointCloud<Point_t> VoxelHashMap::MergePoints(const pcl::PointCloud<Point_t> &voxel_cloud) {
    pcl::PointCloud<Point_t>::Ptr voxel_cloud_ptr;
    voxel_cloud_ptr.reset(new pcl::PointCloud<Point_t>);
    *voxel_cloud_ptr = voxel_cloud;
    pcl::VoxelGrid<Point_t> voxel_grid;
    voxel_grid.setInputCloud(voxel_cloud_ptr);
    voxel_grid.setLeafSize(0.5, 0.5, 0.5);
    voxel_grid.filter(*voxel_cloud_ptr);

    return *voxel_cloud_ptr;
  }
}
