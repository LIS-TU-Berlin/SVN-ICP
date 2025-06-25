//
// Created by haoming on 26.06.23.
//



#include "OdometryPipeline.h"
#include <gtsam/slam/BetweenFactor.h>

namespace stein_icp
{


    SteinICPOdometry::SteinICPOdometry() : rclcpp::Node("SteinMICP")
    {
        // parameters
        this->load_param();

        // initialize SteinICP class and buffer
        this->allocateMemory();

        // pre defined prediction and update
        if (estimator==ICP){
            Predictor_ = [&](rclcpp::Time lidar_time)->gtsam::Pose3{
                return this->pose_prediction(lidar_time);
            };
            Updater_ = [&](const gtsam::Pose3 &initial_guess,
                           gtsam::Pose3 &pose3_current,
                           const torch::Tensor &correction_tr,
                           gtsam::Vector6 &var_vector,
                           const std::vector<double> &cov_mat,
                           const rclcpp::Time &lidar_time)->void{
                auto correction_pose = stein_icp::tensor2gtsamPose3(correction_tr);
                pose3_current =  gtsam::Pose3(initial_guess.matrix()*correction_pose.matrix());
            };
        }else{
            Predictor_ = [&](rclcpp::Time lidar_time)->gtsam::Pose3{
               return  kalman_filter_->get_initial_guess();
            };
            Updater_ = [&](const gtsam::Pose3 &initial_guess,
                           gtsam::Pose3 &pose3_current,
                           const torch::Tensor &correction_tr,
                           gtsam::Vector6 &var_vector,
                           const std::vector<double> &cov_mat,
                           const rclcpp::Time &lidar_time)->void{
                this->kf_updater(pose3_current, correction_tr, var_vector, cov_mat, lidar_time);
            };
        }



        // filter
        for(size_t i=0;i<6;i++) {
            max_sliding_window_filter_[i] = std::make_unique<MaxSlidingWindow>(10);
        }

        auto sub_opt = rclcpp::SubscriptionOptions();
        sub_opt.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // subscribe point cloud
        cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(cloud_topic_,
                                                                                     rclcpp::SensorDataQoS().reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE),
                                                                                     [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg)->void
                                                                                     {
                                                                                         this->lidar_msg_cb(msg);
                                                                                     },
                                                                                     sub_opt);
        // subscribe IMU
        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic_,
                                                                                   rclcpp::SensorDataQoS().reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE),
                                                                                   [this](const sensor_msgs::msg::Imu::SharedPtr msg)->void
                                                                                   {
                                                                                       this->imu_msg_cb(msg);
                                                                                   },
                                                                                   sub_opt);
        // publish the ego-centric frame
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        // publishers
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/stein_icp/trajectories", 10);
        state_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/stein_icp/body_state", 10);
        prediction_publisher_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/stein_icp/prediction",10);
        raw_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stein_icp/raw_cloud", 1);
        source_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stein_icp/source_cloud", 1);
        localmap_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stein_icp/localmap_cloud", 1);
        neighbourmap_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/stein_icp/neighbourmap_cloud", 1);
        last_particle_publisher_ = this->create_publisher<stein_particle_msgs::msg::SteinParticle>("/stein_icp/particles", 10);
        stein_param_publisher_ = this->create_publisher<stein_particle_msgs::msg::SteinParameters>("/stein_icp/parameters",10);
        runtime_publisher_ = this->create_publisher<stein_particle_msgs::msg::Runtime>("/stein_icp/runtime", 10);
        all_particle_publisher_ = this->create_publisher<stein_particle_msgs::msg::SteinParticleArray>("/stein_icp/all_particles",10);
        variance_publisher_ = this->create_publisher<stein_particle_msgs::msg::Variance>("/stein_icp/variance", 10);
        KF_Gain_publisher_ = this->create_publisher<stein_particle_msgs::msg::SteinParticle>("/stein_icp/kf_gain", 10);

        // ICP loop
        steinicp_thread_ = std::make_shared<std::thread>(
                [this]()->void{
                    this->ICP_processing();
                }
                );

//        publish_thread_ = std::make_shared<std::thread>(
//                [this]()->void{
//                    this->publish_thread_cb();
//                }
//                );
    }

    void SteinICPOdometry::load_param()
    {
        //SteinICP parameters
        std::string estimator_class;
        this->declare_parameter("estimator", "KF");
        this->get_parameter("estimator", estimator_class);
        if(estimator_class=="KF") estimator=KF; else estimator=ICP;

        this->declare_parameter("class_type", "SteinMICP");
        this->get_parameter("class_type", class_type_);
        this->declare_parameter("SteinICP_parameters.optimizer", "Adam");
        this->get_parameter("SteinICP_parameters.optimizer", steinicp_config_.optimizer);
        this->declare_parameter("SteinICP_parameters.iterations", 50);
        this->get_parameter("SteinICP_parameters.iterations", steinicp_config_.iterations);
        this->declare_parameter("SteinICP_parameters.batch_size", 50);
        this->get_parameter("SteinICP_parameters.batch_size", steinicp_config_.batch_size);
        this->declare_parameter("SteinICP_parameters.normalize", true);
        this->get_parameter("SteinICP_parameters.normalize", steinicp_config_.normalize_cloud);
        this->declare_parameter("SteinICP_parameters.lr", 0.02);
        this->get_parameter("SteinICP_parameters.lr", steinicp_config_.lr);
        this->declare_parameter("SteinICP_parameters.max_dist", 2.8);
        this->get_parameter("SteinICP_parameters.max_dist", steinicp_config_.max_dist);
        this->declare_parameter("SteinICP_parameters.knn_count", 100);
        this->get_parameter("SteinICP_parameters.knn_count", steinicp_config_.KNN_count);
        this->declare_parameter("SteinICP_parameters.particle_size", 100);
        this->get_parameter("SteinICP_parameters.particle_size", particle_count_);
        this->declare_parameter("SteinICP_parameters.frame_gap_seconds", 0.35);
        this->get_parameter("SteinICP_parameters.frame_gap_seconds", msg_buffer_gap_);
        this->declare_parameter("SteinICP_parameters.using_EarlyStop", false);
        this->get_parameter("SteinICP_parameters.using_EarlyStop", steinicp_config_.check_early_stop);
        this->declare_parameter("SteinICP_parameters.convergence_steps", 5);
        this->get_parameter("SteinICP_parameters.convergence_steps", steinicp_config_.convergence_steps);
        this->declare_parameter("SteinICP_parameters.convergence_threshold", 1e-5);
        this->get_parameter("SteinICP_parameters.convergence_threshold", steinicp_config_.convergence_threshold);
        this->declare_parameter("lidar_param.topic", "/velodyne_points");
        this->get_parameter("lidar_param.topic", cloud_topic_);
        this->declare_parameter("SteinICP_parameters.deskew_cloud", false);
        this->get_parameter("SteinICP_parameters.deskew_cloud", deskew_cloud_);
        this->declare_parameter("SteinICP_parameters.save_particles", false);
        this->get_parameter("SteinICP_parameters.save_particles", save_particles_);

        this->declare_parameter("SteinICP_parameters.max_range", 80.0);
        this->get_parameter("SteinICP_parameters.max_range", max_range_);
        this->declare_parameter("SteinICP_parameters.min_range", 0.0);
        this->get_parameter("SteinICP_parameters.min_range", min_range_);

        this->declare_parameter("SteinICP_parameters.pub_cloud", false);
        this->get_parameter("SteinICP_parameters.pub_cloud", pub_cloud_);
        this->declare_parameter("SteinICP_parameters.map_voxel_max_points", 10);
        this->get_parameter("SteinICP_parameters.map_voxel_max_points", local_map_.max_pointscount_);

        this->declare_parameter("SteinICP_parameters.map_range", 100.);
        this->get_parameter("SteinICP_parameters.map_range", local_map_.max_range_);

        this->declare_parameter("SteinICP_parameters.map_voxel_size", 1.0);
        this->get_parameter("SteinICP_parameters.map_voxel_size", local_map_.voxel_size_);


        this->declare_parameter("SteinICP_parameters.voxel_size", 0.01);
        this->get_parameter("SteinICP_parameters.voxel_size", voxel_size_);

        this->declare_parameter("SteinICP_parameters.USE_Segmentation", true);
        this->get_parameter("SteinICP_parameters.USE_Segmentation", use_Segmentation_);
        if (cloud_topic_ == "/rslidar_points")
            use_Segmentation_ = false;
        this->declare_parameter("SteinICP_parameters.use_weight_mean", false);
        this->get_parameter("SteinICP_parameters.use_weight_mean", stein_micp_opt_.use_weight_mean);


        std::string cov_filter_type;
        this->declare_parameter("SteinICP_parameters.cov_filter_type", "none");
        this->get_parameter("SteinICP_parameters.cov_filter_type", cov_filter_type);

        if (cov_filter_type == "mean") {
            steinicp_config_.cov_filter_type = CovFilterType::MEAN;
        }
        else if (cov_filter_type == "max_sliding_window") {
            steinicp_config_.cov_filter_type = CovFilterType::MAX_SLIDING_WINDOW;
        }
        else if(cov_filter_type == "none"){
            steinicp_config_.cov_filter_type = CovFilterType::NONE;
        }

        this->declare_parameter("SteinICP_parameters.SVNFullGrad", false);
        this->get_parameter("SteinICP_parameters.SVNFullGrad", steinicp_config_.SVN_full_grad);

        // imu parameter
        this->declare_parameter("imu_param.topic", "/imu/data");
        this->get_parameter("imu_param.topic", imu_topic_);
        this->declare_parameter("imu_param.rot_rw", std::vector<double>(3));
        this->get_parameter("imu_param.rot_rw", rot_random_walk_);
        this->declare_parameter("imu_param.vel_rw", std::vector<double>(3));
        this->get_parameter("imu_param.vel_rw", vel_random_walk_);
        this->declare_parameter("imu_param.gbstd", std::vector<double>(3));
        this->get_parameter("imu_param.gbstd", bg_std_);
        this->declare_parameter("imu_param.gastd", std::vector<double>(3));
        this->get_parameter("imu_param.gastd", ba_std_);
        std::copy(rot_random_walk_.begin(), rot_random_walk_.end(), imu_noise_.rot_rw.begin());
        std::copy(vel_random_walk_.begin(), vel_random_walk_.end(), imu_noise_.vel_rw.begin());
        std::copy(bg_std_.begin(), bg_std_.end(), imu_noise_.bg_std.begin());
        std::copy(ba_std_.begin(), ba_std_.end(), imu_noise_.ba_std.begin());
        imu_noise_.rot_rw *= (M_PI/ 180.0 /60.0);

        std::cout << " rot RW: " << imu_noise_.rot_rw << std::endl;
        imu_noise_.vel_rw /= 60.0;

        std::cout << " vel RW: " << imu_noise_.vel_rw << std::endl;
        imu_noise_.bg_std *= (M_PI / 180.0 / 3600.0) ;
        imu_noise_.ba_std *= 1e-5;

        // lio parameter
        this->declare_parameter("lio_param.icp_cov_scales", icp_cov_scales_);
        this->get_parameter("lio_param.icp_cov_scales", icp_cov_scales_);

        std::cout << "icp_cov_scales_: " << std::endl;
        for(const auto &s : icp_cov_scales_)
            std::cout << s << " : " << std::endl;

        std::vector<double> init_pos_std, init_vel_std, init_rot_std, extrinsic_t, extrinsic_R;
        gtsam::Matrix3 tmp;
        this->declare_parameter("lio_param.use_constCov", false);
        this->get_parameter("lio_param.use_constCov", use_constCov_);
        if(use_constCov_ || particle_count_ == 1) {
            this->declare_parameter("lio_param.const_LidarCov", std::vector<double>(2));
            this->get_parameter("lio_param.const_LidarCov", constCov_);
        }
        this->declare_parameter("lio_param.initposstd", std::vector<double>(3));
        this->get_parameter("lio_param.initposstd", init_pos_std);
        std::copy(init_pos_std.begin(), init_pos_std.end(), lio_param_.init_pos_std.begin());
        this->declare_parameter("lio_param.initvelstd", std::vector<double>(3));
        this->get_parameter("lio_param.initvelstd", init_vel_std);
        std::copy(init_vel_std.begin(), init_vel_std.end(), lio_param_.init_vel_std.begin());
        this->declare_parameter("lio_param.initattstd", std::vector<double>(3));
        this->get_parameter("lio_param.initattstd", init_rot_std);
        std::copy(init_rot_std.begin(), init_rot_std.end(), lio_param_.init_rot_std.begin());
        this->declare_parameter("lio_param.extrinsic_t", std::vector<double>(3));
        this->get_parameter("lio_param.extrinsic_t", extrinsic_t);
        std::copy(extrinsic_t.begin(), extrinsic_t.end(), lio_param_.t_lidar_imu.begin());
        this->declare_parameter("lio_param.extrinsic_R", std::vector<double>(9));
        this->get_parameter("lio_param.extrinsic_R", extrinsic_R);
        memcpy(tmp.data(), &extrinsic_R[0], 9 * sizeof(double));
        lio_param_.R_lidar_imu = tmp.transpose();
        std::cout << constCov_ << std::endl;
        lio_param_.T_lidar_imu = gtsam::Pose3(gtsam::Rot3(lio_param_.R_lidar_imu), lio_param_.t_lidar_imu);

        this->declare_parameter("dataset_duration", 0.);
        this->get_parameter("dataset_duration", dataset_duration_);


        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: SteinICP parameter loaded\n");
        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: Particle Count: " << particle_count_
                                                                              << " Iterations: " << steinicp_config_.iterations
                                                                              << " Batch Size: " << steinicp_config_.batch_size
                                                                              << " Learning Rate: " << steinicp_config_.lr);
        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: Subscribe topic " << cloud_topic_);
        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: Deskew pointcloud " << deskew_cloud_);
        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: Voxel Size " << voxel_size_);
        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: Use Beta-Stein-ICP " << use_BetaSteinICP_);
        RCLCPP_INFO_STREAM(this->get_logger()," use weight mean: " << stein_micp_opt_.use_weight_mean );

    }

    void SteinICPOdometry::allocateMemory()
    {
        this->set_initpose();
        kalman_filter_ = std::make_shared<estimator::ErrorStateKalmanFilter>(lio_param_, imu_noise_);
        if(class_type_ == "SteinMICP"){
            std::cout << "Lidar odometry with SteinMICP " << std::endl;
            steinicp_odom_ = std::make_unique<stein_icp::SteinMICP>(steinicp_config_, init_pose_, stein_micp_opt_);
        } else {
            std::cout << "Lidar odometry with SteinICP " << std::endl;
            steinicp_odom_ = std::make_unique<stein_icp::SteinICP>(steinicp_config_, init_pose_);
        }
        source_pcl_.reset(new pcl::PointCloud<Point_t>());
        target_pcl_.reset(new pcl::PointCloud<Point_t>());
        total_map_.reset(new pcl::PointCloud<Point_t>());

        cloud_msg_buffer_.resize_buffer(10000);
        imu_measurement_buffer_.resize_buffer(10000);
        odom_buffer_.resize_buffer(10000);
        poses_buffer.resize_buffer(10000);
        bodystate_buffer_.resize_buffer(10000);
        predict_pose_buffer_.resize_buffer(10000);
        stein_particle_weight_buffer_.resize_buffer(10000);
        cov_matrix_buffer_.resize_buffer(10000);
        body_state_.state = gtsam::NavState(gtsam::Rot3(), gtsam::Vector3::Zero(), gtsam::Vector3::Zero());
        cloud_segmentation_ = std::make_shared<ImageProjection>();
//        poses_buffer.resize(1000);

    }


    void SteinICPOdometry::imu_msg_cb(const sensor_msgs::msg::Imu::SharedPtr &msg)
    {
        fgo::data_types::IMUMeasurement imu_measurement;
        imu_timestamp_current_ = rclcpp::Time(msg->header.stamp.sec, msg->header.stamp.nanosec, RCL_ROS_TIME);
        imu_measurement.timestamp = imu_timestamp_current_;
        if (is_firts_IMU_){
            imu_timestamp_last_ = imu_timestamp_current_;
            is_firts_IMU_ = false;
        }
        imu_measurement.dt = imu_timestamp_current_.seconds() - imu_timestamp_last_.seconds();
        imu_measurement.accLin << msg->linear_acceleration.x,
                                  msg->linear_acceleration.y,
                                  msg->linear_acceleration.z;
        imu_measurement.velRot << msg->angular_velocity.x,
                                  msg->angular_velocity.y,
                                  msg->angular_velocity.z;

        imu_timestamp_last_ = imu_timestamp_current_;

        imu_measurement_buffer_.update_buffer(imu_measurement, imu_timestamp_current_);

    }


    void SteinICPOdometry::lidar_msg_cb(const sensor_msgs::msg::PointCloud2::SharedPtr &msg)
    {

        lidar_timestamp_current_ = rclcpp::Time(msg->header.stamp.sec, msg->header.stamp.nanosec, RCL_ROS_TIME);

        //// Point cloud segmentation
        // This method depends on the sensor parameters, like channel and frequence, so no points should be filterd
        // in this step. This means that this step is the first step in pre-processing. So we put it in this thread.
        sensor_msgs::msg::PointCloud2::SharedPtr segmented_msg;
        segmented_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        if(use_Segmentation_){
            cloud_segmentation_->cloudHandler(msg);
            pcl::PointCloud<Point_t>::Ptr segmented_cloud;
            segmented_cloud.reset(new pcl::PointCloud<Point_t>());
            segmented_cloud = cloud_segmentation_->GetSegmentedCloudPure();
            std::cout << "Segmented Cloud Points Number:" << segmented_cloud->size() << std::endl;
            pcl::toROSMsg(*segmented_cloud, *segmented_msg);
            segmented_msg->header = msg->header;
        } else {
            segmented_msg = msg;
        }

        pcl::PointCloud<Point_t> tmp_cloud;
        pcl::fromROSMsg(*segmented_msg, tmp_cloud);
        pcl::transformPointCloud(tmp_cloud, tmp_cloud, lio_param_.T_lidar_imu.matrix());
        pcl::toROSMsg(tmp_cloud, *segmented_msg);
        cloud_msg_buffer_.update_buffer(segmented_msg, lidar_timestamp_current_);
        RCLCPP_INFO_STREAM(this->get_logger(), "[SteinICP]: Msg buffer size: " << cloud_msg_buffer_.size());
        lidar_timestamp_last_ = lidar_timestamp_current_;
    }


    pcl::PointCloud<Point_t> SteinICPOdometry::deskew_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr &msg)
    {
        pcl::PointCloud<Point_t>::Ptr frame;
        frame.reset(new pcl::PointCloud<Point_t>());
        pcl::fromROSMsg(*msg, *frame);
        auto frame_points = frame->points;

        sensor_msgs::msg::PointField timestamp_field;
        for(auto field : msg->fields)
        {
            if(field.name == "t" || field.name == "timestamp" || field.name == "time")
                timestamp_field = field;
        }

        // timestamps for each point
        const size_t point_number = msg->width * msg->height;
        const auto new_frame_timestamp = rclcpp::Time(msg->header.stamp.sec, msg->header.stamp.nanosec, RCL_ROS_TIME);
        auto extract_timestamps = [&]<typename T>(sensor_msgs::PointCloud2ConstIterator<T> &&it)->std::vector<double>{
            std::vector<double> timestamps;

            timestamps.reserve(point_number);
            for(size_t i = 0; i < point_number; ++it, ++i)
            {
                timestamps.emplace_back(static_cast<double>(*it));
            }

            return timestamps;
        };

        std::vector<double> extracted_timestamps(frame_points.size());
        // correct KITTI scan and calculate timestamp manually
        if (cloud_topic_ == "/kitti/velo/pointcloud")
        {
            constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;

            tbb::parallel_for(size_t(0), frame->size(), [&](size_t i){
                Eigen::Vector3d pt(frame_points[i].x, frame_points[i].y, frame_points[i].z);
                Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
                auto corrected_frame =
                        Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
                frame_points[i].x = corrected_frame.x();
                frame_points[i].y = corrected_frame.y();
                frame_points[i].z = corrected_frame.z();
                auto x = frame_points[i].x ;
                auto y = frame_points[i].y ;
                auto yaw = -std::atan2(y, x);
                extracted_timestamps[i] = 0.5 * (yaw / M_PI + 1.0);
            });
        }
        //direct extract timestamp for other datasets
        else
        {
            if (timestamp_field.datatype == sensor_msgs::msg::PointField::UINT32)
                extracted_timestamps = extract_timestamps(
                        sensor_msgs::PointCloud2ConstIterator<uint32_t>(*msg, timestamp_field.name));
            else if (timestamp_field.datatype == sensor_msgs::msg::PointField::FLOAT32)
                extracted_timestamps = extract_timestamps(
                        sensor_msgs::PointCloud2ConstIterator<float>(*msg, timestamp_field.name));
            else if (timestamp_field.datatype == sensor_msgs::msg::PointField::FLOAT64)
                extracted_timestamps = extract_timestamps(
                        sensor_msgs::PointCloud2ConstIterator<double>(*msg, timestamp_field.name));
        }
        const auto [min_time_it, max_time_it] = std::minmax_element(extracted_timestamps.begin(), extracted_timestamps.end());
        const double min_timestamp = *min_time_it;
        const double max_timestamp = *max_time_it;
        if (min_timestamp == max_timestamp) return *frame;
        std::vector<double> timestamps(extracted_timestamps.size());
        std::transform(extracted_timestamps.begin(), extracted_timestamps.end(), timestamps.begin(),
                       [&](const auto &timestamp){
            return (timestamp-min_timestamp) / (max_timestamp-min_timestamp) ;
        });

        auto pose_number = poses_buffer.size();
        auto start_frame = poses_buffer.get_buffer_from_id(pose_number - 2);
        auto start_time = poses_buffer.time_buffer[pose_number - 2];
        auto finish_frame = poses_buffer.get_last_buffer();
        auto finish_time = poses_buffer.get_last_time();
//        auto start_frame = poses_buffer.get_last_buffer();
//        auto finish_frame = body_state_.state.pose();
        auto delta_pose = gtsam::Pose3::Logmap(gtsam::Pose3(start_frame.matrix().inverse()*finish_frame.matrix()));
//        auto delta_time_ratio = (new_frame_timestamp.seconds()-finish_time.seconds()) / (finish_time.seconds()-start_time.seconds());
//        auto new_time_normalized = (new_frame_timestamp.seconds()-min_timestamp) / (max_timestamp-min_timestamp);

        pcl::PointCloud<Point_t>::Ptr deskewed_pcl;
        deskewed_pcl.reset(new pcl::PointCloud<Point_t>());
        deskewed_pcl->resize(point_number);
        tbb::parallel_for(size_t(0), point_number, [&](size_t i){
           const auto motion = gtsam::Pose3::Expmap((timestamps[i]-0.5) * delta_pose );
           auto deskewed_point = motion.transformFrom(gtsam::Point3(frame_points[i].x, frame_points[i].y,frame_points[i].z));
           deskewed_pcl->points[i] = Point_t(deskewed_point.x(), deskewed_point.y(), deskewed_point.z());
        });

        return *deskewed_pcl;

    }

    void SteinICPOdometry::ICP_processing()
    {

        torch::Tensor odom_tr_last = torch::zeros({6}).to(torch::kCUDA);
        pcl::PointCloud<Point_t>::Ptr deskewed_cloud;
        deskewed_cloud = std::make_shared<pcl::PointCloud<Point_t>>();
        pcl::PointCloud<Point_t>::Ptr voxelized_cloud;
        voxelized_cloud = std::make_shared<pcl::PointCloud<Point_t>>();
        pcl::PointCloud<Point_t>::Ptr voxelized_cloud_toMap;
        voxelized_cloud_toMap = std::make_shared<pcl::PointCloud<Point_t>>();
        pcl::PointCloud<Point_t>::Ptr cropped_cloud;
        cropped_cloud = std::make_shared<pcl::PointCloud<Point_t>>();
        pcl::PointCloud<Point_t>::Ptr target_cloud_lidar_frame;
        target_cloud_lidar_frame = std::make_shared<pcl::PointCloud<Point_t>>();
        pcl::PointCloud<Point_t>::Ptr source_cloud_global;
        source_cloud_global = std::make_shared<pcl::PointCloud<Point_t>>();
        rclcpp::Time lidar_time, imu_time, imu_time_pre;
        fgo::data_types::IMUMeasurement imu_previous, imu_current;
        sensor_msgs::msg::PointCloud2::SharedPtr source_msg;
        source_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        bool first_imu = true;
        bool imu_interpolation = false;
        bool synced = false;
        static auto first_time = rclcpp::Time();
        static bool first_time_set = false;
        while(rclcpp::ok()){

            // sync IMU and LiDAR
            if (imu_measurement_buffer_.size() != 0 && estimator==KF){
                if(imu_interpolation){
                    kalman_filter_->predict(body_state_, imu_previous, imu_current);
                    imu_previous = imu_current;
                    imu_interpolation = false;
                }
                if(first_imu || synced)
                std::tie(imu_current, imu_time) = imu_measurement_buffer_.get_first_buffer_time_pair_and_pop();
                if (first_imu) {
                    imu_previous = imu_current;
                    first_imu = false;
                }

                if (cloud_msg_buffer_.size() == 0 && synced){
                    body_state_pre_ = body_state_;
                    kalman_filter_->predict(body_state_, imu_previous, imu_current);
                    imu_previous = imu_current;
                    continue;
                }
                else if (cloud_msg_buffer_.size() !=0){
                    synced = true;
                    auto first_lidar_time =  cloud_msg_buffer_.get_first_time();
                    // timestamp matches well
                    if ( abs(first_lidar_time.seconds() - imu_time.seconds()) < 0.001 ){
                        body_state_pre_ = body_state_;
                        kalman_filter_->predict(body_state_, imu_previous, imu_current);
                        imu_previous = imu_current;
                    }
                    // LiDAR timestamp is slower than imu
                    else if (first_lidar_time.seconds() < imu_time.seconds()){
                        // LiDAR between two imu, interpolate IMU
                        if (first_lidar_time.seconds() > imu_previous.timestamp.seconds() ){
                            auto interpolate_imu = IMU::ImuInterpolation(first_lidar_time, imu_previous, imu_current);
                            kalman_filter_->predict(body_state_, imu_previous, interpolate_imu);
                            imu_previous = interpolate_imu;
                            imu_interpolation = true;
                        }
                        else{
                            synced = false;
                            cloud_msg_buffer_.get_first_buffer_time_pair_and_pop();
                            continue;
                        }
                    }
                    // IMU is slower than LiDAR
                    else if(first_lidar_time.seconds() > imu_time.seconds()){
                        body_state_pre_ = body_state_;
                        kalman_filter_->predict(body_state_, imu_previous, imu_current);
                        imu_previous = imu_current;
                        continue;
                    }
                    auto [origin_msg, timestamp] = cloud_msg_buffer_.get_first_buffer_time_pair_and_pop();
//                    auto dummy  = imu_measurement_buffer_.get_first_buffer_time_pair_and_pop();
                    std::cout << "time diff " << timestamp.seconds() - imu_time.seconds() << std::endl;
                    source_msg = origin_msg;
                    lidar_time = timestamp;
                    pcl::fromROSMsg(*source_msg, *deskewed_cloud);
                    if (lidar_time.seconds() - timestamp_odom_.seconds() < msg_buffer_gap_)
                        continue;
                }
                else continue;
            }else if(estimator==ICP && cloud_msg_buffer_.size() !=0){
                auto [origin_msg, timestamp] = cloud_msg_buffer_.get_first_buffer_time_pair_and_pop();
//                    auto dummy  = imu_measurement_buffer_.get_first_buffer_time_pair_and_pop();
                source_msg = origin_msg;
                lidar_time = timestamp;
                pcl::fromROSMsg(*source_msg, *deskewed_cloud);
                if (lidar_time.seconds() - timestamp_odom_.seconds() < msg_buffer_gap_)
                    continue;
            }else{
                continue;
            }

            if(!first_time_set)
            {
                first_time = rclcpp::Time(source_msg->header.stamp.sec, source_msg->header.stamp.nanosec, RCL_ROS_TIME);
                first_time_set = true;
            }

            //time gap between two frames
            auto preprocessing_start = std::chrono::steady_clock::now();

            // deskew
            if(deskew_cloud_ && poses_buffer.size() >= 2){
                *deskewed_cloud = this->deskew_pointcloud(source_msg);
            }
            // crop cloud
            *cropped_cloud = this->crop_pointcloud(deskewed_cloud);

            // downsample cloud
            *voxelized_cloud_toMap = this->downsample_uniform(cropped_cloud, 0.5*voxel_size_);
            *voxelized_cloud = this->downsample_uniform(voxelized_cloud_toMap, 1.5*voxel_size_);

            //// predict pose
            gtsam::Pose3 initial_guess;
            initial_guess = this->Predictor_(lidar_time);
//            if (estimator==KF) initial_guess = kalman_filter_->get_initial_guess();
//            else initial_guess = this->pose_prediction(lidar_time);

            *source_pcl_ = *voxelized_cloud;
            //transform source cloud to global used to find closest neighbours.
            pcl::transformPointCloud(*source_pcl_, *source_cloud_global, initial_guess.matrix());

            // initialize particles
            this->set_initpose();
            source_tr_ = stein_icp::vector2tensor(stein_icp::pointcloud2vector(source_pcl_), torch::kCUDA);
            // get map
            if (!local_map_.Empty()){
                *target_pcl_ = local_map_.GetMap(initial_guess, scan_max_range_ + 10.);
                if(target_pcl_->size() == 0)
                {
                    *target_pcl_ = local_map_.GetMap();
                }
                target_tr_ = stein_icp::vector2tensor(stein_icp::pointcloud2vector(target_pcl_), torch::kCUDA);
                steinicp_odom_->add_cloud(source_tr_, target_tr_, init_pose_.clone());
            } else {
                // add the cloud at the first frame
                local_map_.AddPointCloud(*cropped_cloud, initial_guess);
                poses_buffer.update_buffer(initial_guess, lidar_time);
                predict_pose_buffer_.update_buffer(initial_guess, lidar_time);
                cov_matrix_buffer_.update_buffer(1e-4*gtsam::Matrix66::Identity(), lidar_time);
                this->publish_odometry();
                continue;
            }
            auto preprocessing_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> preprocessing_duration = preprocessing_end-preprocessing_start;
            std::cout << "Preprocessing Time: " << preprocessing_duration.count() << std::endl;

            //// ICP
            auto align_start = std::chrono::steady_clock::now();
            steinicp_odom_->set_initial_mean(initial_guess); //TODO
            stein_icp::SteinICPState align_state = steinicp_odom_->stein_align();
            if(align_state != SteinICPState::ALIGN_SUCCESS){
                continue;
            }
            auto correction_tr = steinicp_odom_->get_transformation();       // the output pose, size 6*1
            auto cov_tr = steinicp_odom_->get_distribution();                // variance, size 6*1
            auto cov_mat = steinicp_odom_->get_cov_matrix();

            auto stein_particles = steinicp_odom_->get_particles();   // particles in the last iteration
            auto stein_weights = steinicp_odom_->get_particle_weight();  // weights

            gtsam::Vector6 var_vector, var_vector_max_sliding;
            var_vector = tensor2Vector6(cov_tr);
            gtsam::Matrix6 cov_mat_diag = var_vector.asDiagonal();

            //// correction step
            gtsam::Pose3 pose3_current;
            Updater_(initial_guess, pose3_current, correction_tr, var_vector, cov_mat, lidar_time);
//            if(estimator==ICP){
//                auto correction_pose = stein_icp::tensor2gtsamPose3(correction_tr);
//                pose3_current =  gtsam::Pose3(initial_guess.matrix()*correction_pose.matrix());
//            }
//            else {
//                this->kf_updater(pose3_current, correction_tr, var_vector, cov_mat, lidar_time);
//            }

            auto align_end = std::chrono::steady_clock::now();
            std::chrono::duration<float> align_duration = align_end - align_start;
            RCLCPP_INFO_STREAM(this->get_logger(),
                               "Align Process Time : " << align_duration.count());

            // update map
            local_map_.AddPointCloud(*voxelized_cloud_toMap, pose3_current);

            // save in buffer
            poses_buffer.update_buffer(pose3_current, lidar_time);
            predict_pose_buffer_.update_buffer(initial_guess, lidar_time);
            stein_particle_weight_buffer_.update_buffer({stein_particles,stein_weights},lidar_time) ;
            bodystate_buffer_.update_buffer(body_state_, lidar_time);
            timestamp_odom_ = lidar_time;
            auto cloud_pub = this->crop_pointcloud(deskewed_cloud);
            pcl::transformPointCloud(cloud_pub, cloud_pub, pose3_current.matrix());
            raw_cloud_buffer_.emplace_back(lidar_time, cloud_pub);
            source_cloud_buffer_.emplace_back(lidar_time, *source_cloud_global);
            target_cloud_buffer_.emplace_back(lidar_time, *target_cloud_lidar_frame);
            steinicp_runtime_ = align_duration.count();
            preprocessing_runtime_ = preprocessing_duration.count();
            cov_matrix_buffer_.update_buffer(cov_mat_diag, lidar_time);
            odom_finished_ = true;
            this->publish_thread_cb();

            // check progress
            if(dataset_duration_ != 0)
            {
                const auto current_time = rclcpp::Time(source_msg->header.stamp.sec, source_msg->header.stamp.nanosec, RCL_ROS_TIME);
                const auto duration = current_time - first_time;
                std::cout << "Current Progress: " << double(duration.nanoseconds()) / dataset_duration_ * 100. << "%" <<std::endl;
            }
        }

    }

    // initialize as uniform distribution
    void SteinICPOdometry::set_initpose()
    {
        torch::Tensor lb = torch::tensor({-0.3, -0.2, -0.1, -0.004, -0.004, -0.012}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
        torch::Tensor ub = torch::tensor({0.3, 0.2, 0.1, 0.004, 0.004, 0.012}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
        init_pose_ = stein_icp::initialize_particles(particle_count_, torch::kCUDA, ub, lb);
    }

    // initializa as a Gaussian distribution
    void SteinICPOdometry::set_initpose(const gtsam::Vector6 &cov)
    {
        init_pose_ = stein_icp::initialize_particles_gaussian(particle_count_, torch::kCUDA, cov);
    }

    // downsample using pcl voxel, mean point in each voxel
    pcl::PointCloud<Point_t> SteinICPOdometry::downsample_voxel(const pcl::PointCloud<Point_t>::Ptr &cloud, double voxel_size)
    {
        voxel_grid_.setInputCloud(cloud);
        voxel_grid_.setLeafSize(voxel_size, voxel_size, voxel_size);
        voxel_grid_.filter(*cloud);
        return *cloud;
    }

    //downsample using pcl uniform, randomly pick a point
    pcl::PointCloud<Point_t> SteinICPOdometry::downsample_uniform(const pcl::PointCloud<Point_t>::Ptr &cloud, double voxel_size)
    {
        uniform_sampling_.setInputCloud(cloud);
        uniform_sampling_.setRadiusSearch(voxel_size);
        uniform_sampling_.filter(*cloud);
        return *cloud;
    }

    pcl::PointCloud<Point_t> SteinICPOdometry::crop_pointcloud(
            const pcl::PointCloud<Point_t>::Ptr &pointcloud)
    {
        pcl::PointCloud<Point_t> cropped_cloud;
        std::copy_if(pointcloud->points.begin(),
                         pointcloud->points.end(),
                         std::back_inserter(cropped_cloud.points),
                         [&](const auto &pt){
                        auto norm = pt.x*pt.x + pt.y*pt.y + pt.z*pt.z;
                        if (norm > scan_max_range_) scan_max_range_ = norm;
                        return norm < max_range_*max_range_ && norm > min_range_*min_range_;
        });
        return cropped_cloud;
    }

    gtsam::Pose3 SteinICPOdometry::pose_prediction(const rclcpp::Time &new_time)
    {
        const size_t N = poses_buffer.size();
        auto delta_pose = gtsam::Pose3();

        if (N < 2)
        {
//            delta_pose_.push_back(delta_pose);
            if (poses_buffer.size() == 0)
                return gtsam::Pose3();
            else
                return poses_buffer.get_last_buffer() * gtsam::Pose3();
        }

        auto start_frame = poses_buffer.get_buffer_from_id(N - 2);
        auto start_time = poses_buffer.time_buffer[N - 2];
        auto finish_frame = poses_buffer.get_last_buffer();
        auto finish_time = poses_buffer.get_last_time();
        auto delta_time = finish_time.seconds()-start_time.seconds();
        auto new_delta_time = new_time.seconds() - finish_time.seconds();

        // dT = inv(T_k-2) * T_k-1
        delta_pose = gtsam::Pose3(start_frame.matrix().inverse() * finish_frame.matrix());
        auto time_ratio = new_delta_time/delta_time;
        delta_pose = gtsam::Pose3::Expmap(time_ratio * gtsam::Pose3::Logmap(delta_pose));

        // T_pred = T_k-1 * dT
        auto initial_guess = gtsam::Pose3(finish_frame.matrix() * delta_pose.matrix());

        body_state_.state = gtsam::NavState(initial_guess, gtsam::Pose3::Expmap(gtsam::Pose3::Logmap(delta_pose)/delta_time).translation());
        return initial_guess;
    }

    void SteinICPOdometry::kf_updater(gtsam::Pose3 &pose3_current,
                                      const torch::Tensor &correction_tr,
                                      gtsam::Vector6 &var_vector,
                                      const std::vector<double> &cov_mat,
                                      const rclcpp::Time &lidar_time)
    {
        gtsam::Vector6 var_vector_max_sliding;
        gtsam::Matrix6 cov_mat_diag = var_vector.asDiagonal();
        gtsam::Vector6 icp_correction = stein_icp::tensor2Vector6(correction_tr);
        for(size_t i = 0; i < 6; ++i)
            var_vector[i] *= icp_cov_scales_[i];

        gtsam::Matrix66 cov_smooth;
        this->cov_smoother(cov_smooth, 20);
        stein_particle_msgs::msg::Variance var_msg;
        var_msg.header.stamp = lidar_time;
        const auto rw_var = kalman_filter_->get_random_walk_variance();

        for(uint i = 0; i < 6; ++i) {
            var_vector_max_sliding(i) = max_sliding_window_filter_[i]->filter(var_vector(i));
            var_msg.var_icp[i] = var_vector(i);
            var_msg.var_mean_filtered[i] = cov_smooth(i, i);
            var_msg.var_maxsliding_filtered[i] = var_vector_max_sliding(i);
            var_msg.var_random_walk[i] = rw_var(i);
        }

        variance_publisher_->publish(var_msg);

        //TODO:T_pred * T_icp, transform the map to local
        if (use_constCov_ || particle_count_ == 1)
            cov_mat_diag = gtsam::Vector6(constCov_[0], constCov_[0], constCov_[0], constCov_[1], constCov_[1], constCov_[1]).asDiagonal();

        switch (steinicp_config_.cov_filter_type) {
            case CovFilterType::NONE: {
                kalman_filter_->update(body_state_, cov_mat_diag, icp_correction);
                break;
            }
            case CovFilterType::MEAN:  {
                kalman_filter_->update(body_state_, cov_smooth, icp_correction);
                std::cout << "cov filter: MEAN" << std::endl;
                break;
            }
            case CovFilterType::MAX_SLIDING_WINDOW: {
                kalman_filter_->update(body_state_, var_vector_max_sliding.asDiagonal(), icp_correction);
                std::cout << "cov filter: MAX_SLIDING_WINDOW" << std::endl;
                break;
            }
        }
        const auto kf_cov = kalman_filter_->get_cov();
        pose3_current = body_state_.state.pose();
        auto kf_gain = kalman_filter_->get_KFGain();
        stein_particle_msgs::msg::SteinParticle kf_gain_msg;
        kf_gain_msg.header.stamp = lidar_time;
        kf_gain_msg.x = std::vector<double>(kf_gain.data(), kf_gain.data()+90);
        kf_gain_msg.y = std::vector<double>(kf_cov.data(), kf_cov.data()+225);
        kf_gain_msg.z = std::vector<double>(cov_mat.data(), cov_mat.data()+36);
        KF_Gain_publisher_->publish(kf_gain_msg);
    }

    void SteinICPOdometry::cov_smoother( gtsam::Matrix66 &cov, double window_size)
    {
        int N = cov_matrix_buffer_.size();
        if (N == 0) cov = gtsam::Vector6(constCov_[0], constCov_[0], constCov_[0], constCov_[1], constCov_[1], constCov_[1]).asDiagonal();
        if (N < window_size) window_size = N;
        gtsam::Matrix66 sum = gtsam::Matrix66::Zero();
        for (int i = 0; i <window_size; i++){
            auto cov_pre = cov_matrix_buffer_.get_buffer_from_id(N-i-1);
            sum += cov_pre;
        }
        cov = sum/window_size;
    }

    void SteinICPOdometry::variance_prediction()
    {
        if(particle_count_ == 1)
        {
            init_pose_ = torch::zeros({6,1,1}).to(torch::kCUDA);
            return;
        }
        int N = cov_matrix_buffer_.size();
        if (N < 2)
            this->set_initpose();
        else
        {
            auto cov_begin = cov_matrix_buffer_.get_buffer_from_id(N-2);
            auto cov_end = cov_matrix_buffer_.get_buffer_from_id(N-1);
            gtsam::Vector6 cov = (cov_begin + cov_end).diagonal();

            this->set_initpose(cov);

            double delta = sqrt((cov.x()+cov.y()+cov.z())) + 2.0 * max_range_ * std::sin(sqrt(cov[3] + cov[4] + cov[5]) / 2.0);
            int k = local_map_.max_pointscount_ * round(3.0*delta);
            if (delta < 0.2) k = 10;
            else if (delta > 1) k = 100;

            std::cout << "k =" << k << std::endl;
            steinicp_odom_->set_threshold(3.0*delta);
            steinicp_odom_->set_k(k);
        }
    }

    void SteinICPOdometry::publish_stein_param()
    {
         stein_particle_msgs::msg::SteinParameters stein_param_msg;
         stein_param_msg.optimizer = steinicp_config_.optimizer;
         stein_param_msg.learning_rate = steinicp_config_.lr;
         stein_param_msg.iterations = steinicp_config_.iterations;
         stein_param_msg.batch_size = steinicp_config_.batch_size;
         stein_param_msg.particle_count = particle_count_;
         stein_param_msg.early_stop = steinicp_config_.check_early_stop;
         stein_param_msg.converge_steps = steinicp_config_.convergence_steps;
         stein_param_msg.converge_threshold = steinicp_config_.convergence_threshold;
         stein_param_msg.correspondence_distance = steinicp_config_.max_dist;

         stein_param_msg.point_range = {min_range_, max_range_};
         stein_param_msg.voxel_size = voxel_size_;
         stein_param_msg.map_voxel_size = local_map_.voxel_size_;
         stein_param_msg.map_voxel_max_points = local_map_.max_pointscount_;
         stein_param_msg.weight_mean = stein_micp_opt_.use_weight_mean;

         stein_param_publisher_->publish(stein_param_msg);

    }



    void SteinICPOdometry::publish_odometry()
    {
         const auto last_pose = poses_buffer.get_last_buffer();
         auto timestamp = poses_buffer.get_last_time();

         auto pose_msg = geometry_msgs::msg::PoseStamped();
         auto pose = geometry_msgs::msg::Pose() ;
         pose_msg.header.frame_id = "odom_steinicp";
         pose_msg.header.stamp = timestamp;
         pose_msg.pose.position.x = last_pose.x();
         pose_msg.pose.position.y = last_pose.y();
         pose_msg.pose.position.z = last_pose.z();
         pose_msg.pose.orientation.x = last_pose.rotation().toQuaternion().x();
         pose_msg.pose.orientation.y = last_pose.rotation().toQuaternion().y();
         pose_msg.pose.orientation.z = last_pose.rotation().toQuaternion().z();
         pose_msg.pose.orientation.w = last_pose.rotation().toQuaternion().w();

         path_msg_.poses.push_back(pose_msg);
         path_msg_.header.frame_id = "odom_steinicp";
//         path_publisher_->publish(path_msg_);

//         auto cov = *(cov_matrix_buffer_.end()-1);
         gtsam::Matrix6 cov_matrix = cov_matrix_buffer_.get_last_buffer();
//         cov_matrix.block<3,3>(0,0) = last_pose.rotation().matrix()* cov_matrix.block<3,3>(0,0)*last_pose.rotation().matrix().transpose();
         auto cov = std::vector<double>(cov_matrix.data(), cov_matrix.data()+36);

         auto state_msg = nav_msgs::msg::Odometry();
         auto body_state = bodystate_buffer_.get_last_buffer();
         state_msg.header = pose_msg.header;
         state_msg.pose.pose = pose_msg.pose;
         std::copy_n(cov.begin(), 36, state_msg.pose.covariance.begin());
         state_msg.twist.twist.linear.x = body_state.state.v().x();
         state_msg.twist.twist.linear.y = body_state.state.v().y();
         state_msg.twist.twist.linear.z = body_state.state.v().z();
         state_publisher_->publish(state_msg);

        auto predict_pose = predict_pose_buffer_.get_last_buffer();
        auto prediction_stamp = predict_pose_buffer_.get_last_time();
        geometry_msgs::msg::PoseWithCovarianceStamped prediction_msg;
        prediction_msg.header.frame_id = "odom_steinicp";
        prediction_msg.header.stamp = prediction_stamp;
        prediction_msg.pose.pose.position.x = predict_pose.x();
        prediction_msg.pose.pose.position.y = predict_pose.y();
        prediction_msg.pose.pose.position.z = predict_pose.z();
        prediction_msg.pose.pose.orientation.x = predict_pose.rotation().toQuaternion().x();
        prediction_msg.pose.pose.orientation.y = predict_pose.rotation().toQuaternion().y();
        prediction_msg.pose.pose.orientation.z = predict_pose.rotation().toQuaternion().z();
        prediction_msg.pose.pose.orientation.w = predict_pose.rotation().toQuaternion().w();
        prediction_publisher_->publish(prediction_msg);

         geometry_msgs::msg::TransformStamped transformStamped_msg ;
         transformStamped_msg.header = pose_msg.header;
         transformStamped_msg.child_frame_id = "test";
         transformStamped_msg.transform.translation.x = pose_msg.pose.position.x;
         transformStamped_msg.transform.translation.y = pose_msg.pose.position.y;
         transformStamped_msg.transform.translation.z = pose_msg.pose.position.z;
         transformStamped_msg.transform.rotation.x = pose_msg.pose.orientation.x;
         transformStamped_msg.transform.rotation.y = pose_msg.pose.orientation.y;
         transformStamped_msg.transform.rotation.z = pose_msg.pose.orientation.z;
         transformStamped_msg.transform.rotation.w = pose_msg.pose.orientation.w;
         tf_broadcaster_->sendTransform(transformStamped_msg);

    }


    void SteinICPOdometry::publish_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher,
                                         std::vector<std::pair<rclcpp::Time, pcl::PointCloud<Point_t>>> &cloud_vec)
    {
        pcl::PointCloud<Point_t> cloud_pcl;
        for (const auto& cloud_pair : cloud_vec)
        {
            this->publish_cloud(publisher, cloud_pair.second, cloud_pair.first);
        }
        cloud_vec.clear();
    }

    void SteinICPOdometry::publish_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
                                         const pcl::PointCloud<Point_t>& cloud,
                                         rclcpp::Time timestamp)
    {
        auto cloud_msg = sensor_msgs::msg::PointCloud2();
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg.header.frame_id = "odom_steinicp";
        cloud_msg.header.stamp = timestamp;
        publisher->publish(cloud_msg);
    }

    void SteinICPOdometry::publish_particle_info()
    {
        if (stein_particle_weight_buffer_.size() > 0) {
            stein_particle_msgs::msg::SteinParticle particle_msg;
            auto particle_weight_to_pub = stein_particle_weight_buffer_.get_last_buffer();
            auto timestamp = stein_particle_weight_buffer_.get_last_time();
            auto particles = particle_weight_to_pub.first;
            auto weights = particle_weight_to_pub.second;
            particle_msg.x = std::vector<double>(particles.begin(), particles.begin() + particle_count_);
            particle_msg.y = std::vector<double>(particles.begin() + particle_count_,
                                                 particles.begin() + 2 * particle_count_);
            particle_msg.z = std::vector<double>(particles.begin() + 2 * particle_count_,
                                                 particles.begin() + 3 * particle_count_);
            particle_msg.roll = std::vector<double>(particles.begin() + 3 * particle_count_,
                                                    particles.begin() + 4 * particle_count_);
            particle_msg.pitch = std::vector<double>(particles.begin() + 4 * particle_count_,
                                                     particles.begin() + 5 * particle_count_);
            particle_msg.yaw = std::vector<double>(particles.begin() + 5 * particle_count_,
                                                   particles.begin() + 6 * particle_count_);
            particle_msg.weights = weights;
            particle_msg.header.stamp = timestamp;
            last_particle_publisher_->publish(particle_msg);
        }

    }

    void SteinICPOdometry::publish_all_particles(const std::vector<std::vector<float>> &particles)
    {
        stein_particle_msgs::msg::SteinParticleArray all_particle_msg;
        all_particle_msg.header.frame_id = "odom_steinicp";
        for (auto particle_per_loop : particles)
        {
            stein_particle_msgs::msg::SteinParticle particle_msg;
            particle_msg.x = std::vector<double>(particle_per_loop.begin(), particle_per_loop.begin()+particle_count_);
            particle_msg.y = std::vector<double>(particle_per_loop.begin()+particle_count_, particle_per_loop.begin()+2*particle_count_);
            particle_msg.z = std::vector<double>(particle_per_loop.begin()+2*particle_count_, particle_per_loop.begin()+3*particle_count_);
            particle_msg.roll = std::vector<double>(particle_per_loop.begin()+3*particle_count_, particle_per_loop.begin()+4*particle_count_);
            particle_msg.pitch = std::vector<double>(particle_per_loop.begin()+4*particle_count_, particle_per_loop.begin()+5*particle_count_);
            particle_msg.yaw = std::vector<double>(particle_per_loop.begin()+5*particle_count_, particle_per_loop.begin()+6*particle_count_);
            all_particle_msg.stein_particle_array.push_back(particle_msg);
        }
       all_particle_publisher_->publish(all_particle_msg);
    }

    void SteinICPOdometry::publish_runtime()
    {
        stein_particle_msgs::msg::Runtime runtime_msg;
        runtime_msg.preprocessing_time = preprocessing_runtime_;
        runtime_msg.steinicp_time = steinicp_runtime_;
        //runtime_msg.knn_time = stein_partial_runtime_[0] ;
        //runtime_msg.update_time = stein_partial_runtime_[1];
        //runtime_msg.finish_iter = stein_partial_runtime_[2];
        runtime_publisher_->publish(runtime_msg);
    }

    void SteinICPOdometry::publish_thread_cb()
    {
        //sensor_msgs::msg::PointCloud2 total_map_msg;
//        while(rclcpp::ok())
//        {
//            std::this_thread::sleep_for(std::chrono::seconds(2));
            //if(!odom_finished_){
           //     return;
            //}
            //odom_finished_ = false;
            this->publish_odometry();
            this->publish_runtime();

            if(pub_cloud_){
                this->publish_cloud(source_publisher_, source_cloud_buffer_);
                this->publish_cloud(neighbourmap_publisher_, target_cloud_buffer_);
                this->publish_cloud(raw_cloud_publisher_, raw_cloud_buffer_);
                *total_map_ = local_map_.GetMap();
                if(total_map_ != nullptr && !total_map_->empty()) {
                    this->publish_cloud(localmap_publisher_, *total_map_, timestamp_odom_);
                }
            }

            if (save_particles_){
                this->publish_particle_info();
                const auto particles = steinicp_odom_->get_particle_history();
                this->publish_all_particles(particles);
            }

//        }
    }



}

