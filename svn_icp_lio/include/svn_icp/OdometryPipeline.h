//
// Created by haoming on 27.06.23.
//

#ifndef STEINICP_STEINICP_ROS2_H
#define STEINICP_STEINICP_ROS2_H

#pragma once

//general
#include <iostream>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>


//third party
#include <torch/torch.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <gtsam/geometry/Pose3.h>
#include <tbb/parallel_for.h>
#include <torch/torch.h>
#include <Eigen/Eigen>


//ros
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <stein_particle_msgs/msg/stein_particle.hpp>
#include <stein_particle_msgs/msg/stein_parameters.hpp>
#include <stein_particle_msgs/msg/stein_particle_array.hpp>
#include <stein_particle_msgs/msg/runtime.hpp>
#include <stein_particle_msgs/msg/variance.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>


//self defined
#include "Buffer.h"
#include "DataTypes.h"

#include "SVGD_ICP.h"
#include "SVN_ICP.h"
#include "utils.h"
#include "types.h"
#include "VoxelHashMap.h"
#include "imageProjection.h"
#include "DataTypes.h"
#include "ESEKF.h"


namespace stein_icp {
    enum Estimator{KF, ICP};

     class MaxSlidingWindow {
     private:
      std::deque<std::pair<double, int>> window;
      int windowSize;
      int index = 0;

     public:
      MaxSlidingWindow(int size) : windowSize(size) {}

      double filter(double value) {
       // Remove values that are outside the window
       while (!window.empty() && window.front().second <= index - windowSize) {
        window.pop_front();
       }

       // Remove values smaller than the current one
       while (!window.empty() && window.back().first <= value) {
        window.pop_back();
       }

       // Add the new value
       window.emplace_back(value, index++);

       // The maximum value is at the front
       return window.front().first;
      }
     };

    class SteinICPOdometry : public rclcpp::Node {
    private:
        std::string integrator_name_;
        rclcpp::Time timestamp_cloud_;
        std::atomic_bool is_firstScan_ = true;
        std::atomic_bool is_firts_IMU_ = true;
        std::atomic_bool odom_finished_ = false;
        std::mutex variable_mutex_;

        //Estimator
        Estimator estimator;
        std::function<gtsam::Pose3 (rclcpp::Time)> Predictor_;
        std::function<void (const gtsam::Pose3 &initial_guess,
                            gtsam::Pose3 &pose3_current,
                            const torch::Tensor &correction_tr,
                            gtsam::Vector6 &var_vector,
                            const std::vector<double> &cov_mat,
                            const rclcpp::Time &lidar_time)> Updater_;
        std::shared_ptr<estimator::ErrorStateKalmanFilter> kalman_filter_;

        //Cloud Segmentation
        std::shared_ptr<ImageProjection> cloud_segmentation_;

        //Buffer
        fgo::buffer::CircularDataBuffer<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_msg_buffer_;
        fgo::buffer::CircularDataBuffer<fgo::data_types::IMUMeasurement> imu_measurement_buffer_;
        fgo::buffer::CircularDataBuffer<fgo::data_types::Odom> odom_buffer_;
        fgo::buffer::CircularDataBuffer<std::pair<std::vector<double>, std::vector<double>>> stein_particle_weight_buffer_;
        fgo::buffer::CircularDataBuffer<gtsam::Pose3> poses_buffer;
        fgo::buffer::CircularDataBuffer<gtsam::Pose3> predict_pose_buffer_;
        fgo::buffer::CircularDataBuffer<fgo::data_types::State> bodystate_buffer_;
        std::vector<gtsam::Pose3> covariance_buffer_;
        std::vector<std::pair<rclcpp::Time, pcl::PointCloud<Point_t>>> raw_cloud_buffer_;
        std::vector<std::pair<rclcpp::Time, pcl::PointCloud<Point_t>>> source_cloud_buffer_;
        std::vector<std::pair<rclcpp::Time, pcl::PointCloud<Point_t>>> target_cloud_buffer_;
        fgo::buffer::CircularDataBuffer<gtsam::Matrix66> cov_matrix_buffer_;

        //timestamps, control scan gap for registration
        rclcpp::Time lidar_timestamp_last_, lidar_timestamp_current_, timestamp_odom_;
        rclcpp::Time imu_timestamp_last_, imu_timestamp_current_;
        double msg_buffer_gap_;

        //imu parameters
        std::vector<double> rot_random_walk_, vel_random_walk_, ba_std_, bg_std_;
        IMU::ImuRandomWalk imu_noise_;

        //lio parameter
        IMU::LioParam lio_param_;
        bool use_constCov_;
        std::vector<double> constCov_;

        //SteinICP odom
        std::string class_type_;
        std::unique_ptr<stein_icp::SteinICP> steinicp_odom_;
        stein_icp::SteinICPParam steinicp_config_;
        stein_icp::ParticleWeightOpt stein_micp_opt_;
        bool deskew_cloud_;
        std::string cloud_topic_;
        std::string imu_topic_;
        torch::Tensor init_pose_;
        torch::Tensor source_tr_;
        torch::Tensor target_tr_;
        pcl::PointCloud<Point_t>::Ptr source_pcl_;
        pcl::PointCloud<Point_t>::Ptr target_pcl_;
        pcl::PointCloud<Point_t>::Ptr total_map_;
        fgo::data_types::State body_state_, body_state_pre_;

        int particle_count_ = 100;

        double steinicp_runtime_, preprocessing_runtime_;
        stein_icp::VoxelHashMap local_map_;
        double max_range_;
        double min_range_;
        double scan_max_range_;

        double dataset_duration_;
        std::vector<double> icp_cov_scales_{1., 1., 1., 1., 1., 1. };

        bool use_Segmentation_ = true;
        bool use_BetaSteinICP_ = true;

        double voxel_size_ = 1.0;
        pcl::VoxelGrid<Point_t> voxel_grid_;
        pcl::UniformSampling<Point_t> uniform_sampling_;

        //Subscriber & Publisher
        bool pub_cloud_;
        bool save_particles_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr state_publisher_;
        rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr prediction_publisher_;
        nav_msgs::msg::Path path_msg_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr source_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr localmap_publisher_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr neighbourmap_publisher_;
        rclcpp::Publisher<stein_particle_msgs::msg::SteinParticle>::SharedPtr last_particle_publisher_;
        rclcpp::Publisher<stein_particle_msgs::msg::SteinParticleArray>::SharedPtr all_particle_publisher_;
        rclcpp::Publisher<stein_particle_msgs::msg::SteinParameters>::SharedPtr stein_param_publisher_;
        rclcpp::Publisher<stein_particle_msgs::msg::Runtime>::SharedPtr runtime_publisher_;
        rclcpp::Publisher<stein_particle_msgs::msg::Variance>::SharedPtr variance_publisher_;

        rclcpp::TimerBase::SharedPtr pub_cloud_timer_;

        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        //debug publisher
        rclcpp::Publisher<stein_particle_msgs::msg::SteinParticle>::SharedPtr KF_Gain_publisher_;
        bool ds_published_;

        rclcpp::TimerBase::SharedPtr odom_timer_;
        rclcpp::TimerBase::SharedPtr odom_statistic_timer_;

        std::shared_ptr<std::thread> steinicp_thread_;
        std::shared_ptr<std::thread> publish_thread_;

        std::array<std::unique_ptr<MaxSlidingWindow>, 6> max_sliding_window_filter_;

    private:
        /**
         * @brief load SteinICP parameters from .yaml file
         */
        void load_param();

        /**
         * @brief initialize variables, initialize pointer
         */
        void allocateMemory();


        /**
         * @brief receive point cloud msg form topic "velodyne_points"
         *        and buffer them if time step is bigger than 0.15s, buffer per 0.2s
         * @param msg
         */
        void lidar_msg_cb(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

        void imu_msg_cb(const sensor_msgs::msg::Imu::SharedPtr &msg);

        /**
         * @brief deskew point clouds
         */
        pcl::PointCloud<Point_t> deskew_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

        /**
         *@brief process the SteinICP if all condition is fullfilled,
         *       and buffer the odometry in a buffer
         */
        void ICP_processing();

        /**
         *@brief  voxelization downsample
         */
        pcl::PointCloud<Point_t> downsample_voxel(const pcl::PointCloud<Point_t>::Ptr &cloud, double voxel_size);

        pcl::PointCloud<Point_t> downsample_uniform(const pcl::PointCloud<Point_t>::Ptr &cloud, double voxel_size);

        pcl::PointCloud<Point_t> crop_pointcloud(const pcl::PointCloud<Point_t>::Ptr &pointcloud);

        gtsam::Pose3 pose_prediction(const rclcpp::Time &new_time);

        void kf_updater(gtsam::Pose3 &pose3_current,
                        const torch::Tensor &correction_tr,
                        gtsam::Vector6 &var_vector,
                        const std::vector<double> &cov_mat,
                        const rclcpp::Time &lidar_time);

        void cov_smoother(gtsam::Matrix66 &cov, double window_size);

        void variance_prediction();

        void publish_stein_param();

        /**
         * @brief publish the position
         */
        void publish_odometry();


        void publish_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr&,
                           std::vector<std::pair<rclcpp::Time, pcl::PointCloud<Point_t>>> &);

        void publish_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &,
                           const pcl::PointCloud<Point_t>&, rclcpp::Time);

        void publish_particle_info();

        void publish_all_particles(const std::vector<std::vector<float>> &particles);

        void publish_runtime();

        void publish_thread_cb();


    public:
        explicit SteinICPOdometry();

        ~SteinICPOdometry() {
            steinicp_thread_->join();
//            publish_thread_->join();
        }

        /**
         * @brief show whether there is already computed odmometry
         * @return
         */
        bool has_odom();

        /**
         * @return the odometry from buffer
         */
        std::vector<fgo::data_types::Odom> get_odom();

        void set_initpose();

        void set_initpose(const gtsam::Vector6 &cov);

        //TODO: Some update API function should be declared and defined. Here is only an example
        void update_opt_state();


    };
}


#endif //STEINICP_STEINICP_ROS2_H
