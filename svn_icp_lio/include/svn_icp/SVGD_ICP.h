//
// Created by haoming on 26.05.23.
//

#ifndef STEIN_ICP_STEINICP_H
#define STEIN_ICP_STEINICP_H

#pragma once

#include <tuple>
#include <vector>
#include <chrono>

#include <pcl/filters/filter.h>
#include <torch/torch.h>
#include <Eigen/Eigen>

#include "knn.h"
#include "types.h"
#include "utils.h"


namespace stein_icp
{
    /**
     *  parameter structure
     */
    enum CovFilterType{MEAN, MAX_SLIDING_WINDOW,NONE};
    struct SteinICPParam
    {
        SteinICPParam() = default;

        int iterations = 50;
        bool use_minibatch = false;
        int batch_size = 50;
        double lr = 0.02;
        double max_dist = 1.0;
        bool normalize_cloud = true;
        std::string optimizer = "Adam";
        bool check_early_stop = false;
        int convergence_steps = 5;
        double convergence_threshold = 1e-5;
        int KNN_count=100;
        bool SVN_full_grad = true;
        CovFilterType cov_filter_type = NONE;
    };

    enum SteinICPState{
        ALIGN_SUCCESS = 1,
        NO_OPTIMIZER = 2,
    };

    class SteinICP{
    public: // functions

      // ToDo: @shiping, do we need the initial pose in the constructor?
        explicit SteinICP(SteinICPParam &parameters, const torch::Tensor &init_pose);
        ~SteinICP() = default;

        /** @brief add new souce and target cloud to register */
        void add_cloud(const torch::Tensor &new_cloud, const torch::Tensor &target, const torch::Tensor &init_pose);

        /** \brief stein-icp algorithm
        * \return pose particles, size: particle_count*6*/
        virtual SteinICPState stein_align();

        /** @brief get the raw particles of poses*/
        std::vector<double> get_particles();

        /** @brief get the mean value of poses */
        virtual torch::Tensor get_transformation();

        /** @brief return the distribution of particles */
        virtual torch::Tensor get_distribution();

        virtual std::vector<double> get_cov_matrix();

        std::vector<std::vector<float>> get_particle_history();

        virtual std::vector<double> get_particle_weight();

        inline std::vector<double> get_runtime()
        {

            return {knn_duration_, update_duration_, finish_iter_};
        };

        inline void set_k(int k){ K_source_ = k;};

        inline void set_threshold(double max_dist){config_.max_dist = max_dist;};

        inline void set_initial_mean(const gtsam::Pose3 &pose)
        {

            auto R0 = torch::from_blob(pose.rotation().matrix().data(), {3,3},
                                        torch::TensorOptions().dtype(data_type)).to(torch::kCUDA);

            auto t0 = torch::from_blob((void *) pose.translation().data(), {3, 1},
                                   torch::TensorOptions().dtype(data_type)).to(torch::kCUDA);
            R0_ = R0.transpose(0,1); t0_ = t0;
        };

    protected: // functions
        void set_optimizer();

        void allocate_memory_for_log();




        void knn_source_cloud();

        /** \brief  generate mini-batch cloud from source cloud */
        virtual torch::Tensor mini_batch_generator();
        /** @brief generate mini-batch before loop */
        virtual std::tuple<torch::Tensor, torch::Tensor> mini_batch_pair_generator();

        /** \brief angle parameter to rotation matrix particle_count*3*3 */
        virtual  torch::Tensor to_rotation_tensor(const torch::Tensor &r, const torch::Tensor &p, const torch::Tensor &y);

        /** \brief translation parameter to matrix particle_count *3 */
        torch::Tensor to_translation_tensor(const torch::Tensor &x, const torch::Tensor &y, const torch::Tensor &z);

        virtual torch::Tensor rotm_to_ypr_tensor(const torch::Tensor &R);

        /**
         * \brief ignore the dimension of particle_size.
         * \param source                    particle*batch_size * 3
         * \param transformed_source        particle*batch_size * 3
         * \param target                    target_point_count * 3
         *
         *  \return source_paired           particle_size *batch_size*3
         *          transform_paired        particle_sile *batch_size*3
         *          target_paired           particle_size *batch_size*3
         */
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_correspondence(
                const torch::Tensor &source,
                const torch::Tensor &transformed_source,
                const torch::Tensor &target);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_correspondence_fast(
                const torch::Tensor &source,                /// particle*config_.batch_size * 3
                const torch::Tensor &transformed_source,    /// particle*config_.batch_size * 3
                const torch::Tensor &target                 /// target_point_count * 3
        );

        /**
         *  \brief filter the points
         */
        torch::Tensor point_filter(const torch::Tensor &cloud, const torch::Tensor &distance);

        /** \brief calculation the partial derivative of rotation matrix */
        torch::Tensor partial_derivative(const torch::Tensor &roll,
                                                const torch::Tensor &pitch,
                                                const torch::Tensor &yaw );

        torch::Tensor sgd_grad(const torch::Tensor &source_paired,
                               const torch::Tensor &transformed_s_paired,
                               const torch::Tensor &target_paired);

        /** @param pose_parameters   size: particle_count*6@param sgd_grad*/
        torch::Tensor svgd_grad(const torch::Tensor& pose_parameters, const torch::Tensor &sgd_grad);


        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rbf_kernel(const torch::Tensor& x);

        virtual void pose_update(const torch::Tensor &stein_grad);

    protected: // vairables
        /**
         * \param x, y, z, roll, pitch, yaw:  with size particle_size*1*1, device type "cuda"
         * \param source_cloud,  target_cloud:  tensors point_count*3, device type "cuda"
         * \param max_dist: using to filter the correspondence
         */
        //from constructor or update function
        torch::Tensor source_cloud_, target_cloud_;
        torch::Tensor sourceKNN_idx_;
        int K_source_ = 100;
        torch::Tensor x_, y_, z_, rx_, ry_, rz_;
        torch::Tensor x_last_, y_last_, z_last_, roll_last_, pitch_last_, yaw_last_;
        torch::Tensor R_, t_;
        torch::Tensor R0_, t0_, R_total_, t_total_;
        torch::Tensor loss_;

        SteinICPParam config_;

        //internal private
        bool optimizer_set_ = true;
        long particle_size_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_ptr_;
        torch::Tensor normalize_factor_;
        torch::Tensor gradient_scaling_factor_;

        torch::Tensor particle_stack_, weight_stack_;

        //result output
        torch::Tensor pose_particles_;

        torch::Tensor J_l_;
        torch::ScalarType data_type = torch::kFloat64;

        Timer knn_timer_, update_timer_;
        double knn_duration_, update_duration_, finish_iter_ ;

    };
}

#endif //STEIN_ICP_STEINICP_H
