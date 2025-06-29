/*  ------------------------------------------------------------------
    Copyright (c) 2020-2025 XXX
    email: XXX

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

/**
 * @file    VoxelHashMap.h
 * @brief   VoxelHashMap
 * @author  UNKNOWN
 * @date    June 22, 2025
 */

#ifndef VOXELHASHMAP_H
#define VOXELHASHMAP_H

#include <tsl/robin_map.h>
#include <gtsam/geometry/Pose3.h>
#include <pcl/common/transforms.h>

#include "data/DataTypes.h"


namespace svnicp
{
    using namespace data_types;

    struct VoxelHashMap
    {
        using VoxelIdx = Eigen::Vector3i;

        double voxel_size_ = 1.0;
        double max_range_ = 80;
        int max_pointscount_ = 20;

        VoxelHashMap() = default;

        explicit  VoxelHashMap(double voxel_size, double max_range, int max_pointscount)
        :voxel_size_(voxel_size),
         max_range_(max_range),
         max_pointscount_(max_pointscount){}

        struct VoxelHash{
            size_t operator()(const VoxelIdx &voxel) const {
                const auto vec = reinterpret_cast<const uint32_t *>(voxel.data());
                return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
            }

        };

        tsl::robin_map<VoxelIdx, pcl::PointCloud<Point_t>, VoxelHash> map_;

        void Clear(){ map_.clear(); }
        [[nodiscard]] bool Empty() const { return map_.empty(); }
        [[nodiscard]] size_t Size() const { return map_.size(); }

    public:
        void AddPointCloud(const pcl::PointCloud<Point_t> &new_cloud, const gtsam::Pose3 &new_pose);

        pcl::PointCloud<Point_t> GetMap();

        pcl::PointCloud<Point_t> GetMap(const gtsam::Pose3 &pose, const double &max_range);

        pcl::PointCloud<Point_t> GetNeighbourMap(const pcl::PointCloud<Point_t> &source_cloud);

    private:
        void RemoveFarPointCloud(const Eigen::Vector3d &current_position);

        static pcl::PointCloud<Point_t> MergePoints(const pcl::PointCloud<Point_t> &voxel_cloud);
    };
}


#endif //VOXELHASHMAP_H
