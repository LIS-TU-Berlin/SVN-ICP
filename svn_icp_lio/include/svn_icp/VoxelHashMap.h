//
// Created by haoming on 05.01.24.
//

#ifndef STEINICP_OM_VOXELHASHMAP_H
#define STEINICP_OM_VOXELHASHMAP_H

#include <tsl/robin_map.h>
#include <gtsam/geometry/Pose3.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include "types.h"


namespace stein_icp
{
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


//        struct VoxelBlock{
//
//        };

        struct VoxelHash{
            size_t operator()(const VoxelIdx &voxel) const {
                const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
                return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
            }

        };

        tsl::robin_map<VoxelIdx, pcl::PointCloud<Point_t>, VoxelHash> map_;


        inline void Clear(){ map_.clear(); }
        inline bool Empty(){ return map_.empty(); }
        inline size_t Size(){ return map_.size(); }

    public:
        void AddPointCloud(const pcl::PointCloud<Point_t> &new_cloud, const gtsam::Pose3 &new_pose);

        pcl::PointCloud<Point_t> GetMap();

        pcl::PointCloud<Point_t> GetMap(const gtsam::Pose3 &pose, const double &max_range);

        pcl::PointCloud<Point_t> GetNeighbourMap(const pcl::PointCloud<Point_t> &source_cloud);

    private:
        void RemoveFarPointCloud(Eigen::Vector3d current_position);

        pcl::PointCloud<Point_t> MergePoints(const pcl::PointCloud<Point_t> &voxel_cloud);



    };
}


#endif //STEINICP_OM_VOXELHASHMAP_H
