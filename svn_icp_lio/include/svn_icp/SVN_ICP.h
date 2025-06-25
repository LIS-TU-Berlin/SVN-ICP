//
// Created by haoming on 22.08.23.
//
#pragma once
#ifndef STEINICP_BETASTEINICP_H
#define STEINICP_BETASTEINICP_H

#include "SVGD_ICP.h"

namespace stein_icp {
    struct ParticleWeightOpt {
        bool use_weight_mean = false;
    };

    class SteinMICP : public SteinICP {
    public:
        explicit SteinMICP(SteinICPParam &param,
                           const torch::Tensor &init_pose,
                           const ParticleWeightOpt &opt);

        SteinICPState stein_align() override;

        torch::Tensor get_transformation() override;

        torch::Tensor get_distribution() override;

        std::vector<double> get_cov_matrix() override;

        std::vector<double> get_particle_weight() override;

//        std::tuple<gtsam::Matrix6, gtsam::Vector6> Hessian_for_KF();

    private:

        /** @brief Gaussian Newton for right perturbation */
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Newton_grad_right(const torch::Tensor &source_paired,
                                                                   const torch::Tensor &transformed_s_paired,
                                                                   const torch::Tensor &target_paired);

        torch::Tensor to_rotation_tensor(const torch::Tensor &r, const torch::Tensor &p, const torch::Tensor &y) override;

        torch::Tensor rotm_to_ypr_tensor(const torch::Tensor &R) override;

        /** SVGD gradient scaled by Hessian (also called pre-Conditioned SVGD)*/
        torch::Tensor svgd_grad(const torch::Tensor &pose_parameters,
                                const torch::Tensor &newton_grad,
                                const torch::Tensor &H) ;

        /** Full SVN gradient */
        torch::Tensor svn_full_grad(const torch::Tensor &pose_parameters,
                                    const torch::Tensor &H,
                                    const torch::Tensor &b);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rbf_hessian_kernel(const torch::Tensor &x1, const torch::Tensor &H);

        void pose_update(const torch::Tensor &stein_grad) override;



    private:
        ParticleWeightOpt weight_config_;
        torch::Tensor particle_weight_;
        torch::Tensor beta_stein_weight_;
        torch::Tensor pose_parameters_tsr_;
        torch::Tensor sgd_gradient_;
        torch::Tensor target_knn_;

    }; // class SteinMICP
} //namespace svn_icp

#endif //STEINICP_BETASTEINICP_H
