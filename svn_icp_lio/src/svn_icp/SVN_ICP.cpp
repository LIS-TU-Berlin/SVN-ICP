//
// Created by haoming on 22.08.23.
//

#include "SVN_ICP.h"

namespace stein_icp
{
        SteinMICP::SteinMICP(SteinICPParam   &parameters,
                             const torch::Tensor   &init_pose,
                             const ParticleWeightOpt &opt):
                                   SteinICP(parameters, init_pose.clone())
        {
            particle_size_ = init_pose.size(1);
            x_     = init_pose[0].view({particle_size_, 1, 1});
            y_     = init_pose[1].view({particle_size_, 1, 1});
            z_     = init_pose[2].view({particle_size_, 1, 1});
            rx_  = init_pose[3].view({particle_size_, 1, 1});
            ry_ = init_pose[4].view({particle_size_, 1, 1});
            rz_   = init_pose[5].view({particle_size_, 1, 1});

            weight_config_ = opt;
            particle_weight_ = torch::ones({particle_size_, 1}).to(torch::kCUDA).to(data_type) / particle_size_;

            R_ = to_rotation_tensor(rx_, ry_, rz_);
            t_ = to_translation_tensor(x_, y_, z_);
            pose_particles_ = torch::cat({t_.view({particle_size_, 3}).transpose(0, 1), rotm_to_ypr_tensor(R_).transpose(0, 1)}, 0);
        }


        SteinICPState SteinMICP::stein_align()
        {
            auto Newton_gradient = torch::zeros({particle_size_,6}).to(torch::kCUDA).to(data_type);
            auto Hessian_mean = torch::eye(6).expand({particle_size_, 6, 6}).to(torch::kCUDA).to(data_type);
            beta_stein_weight_ = torch::ones({particle_size_, 1}).to(torch::kCUDA);
            particle_weight_ = torch::ones({particle_size_, 1}).to(torch::kCUDA) / particle_size_;

            /** SVGD iterations */
            this->allocate_memory_for_log();
            const auto [mini_batch_all_epochs, target_batch] = mini_batch_pair_generator();

            for(int epoch = 0; epoch < config_.iterations; epoch++){
                torch::Tensor mini_batch_epoch = mini_batch_all_epochs[epoch].expand({particle_size_, config_.batch_size, 3}).to(torch::kCUDA).to(data_type);
                torch::Tensor target_batch_it = target_batch[epoch];

                // update the full pose
                R_total_ = R0_.matmul(R_);
                t_total_ = t0_ + R0_.matmul(t_);

                // transform points from all batches according to the predicted states
                torch::Tensor source_transformed = mini_batch_epoch.matmul(R_total_.transpose(1, 2))+ t_total_.view({particle_size_, 1, 3});

                // find corresponding points
                const auto [source_paired, transformed_s_paired, target_paired] =
                        get_correspondence_fast(mini_batch_epoch, source_transformed, target_batch_it);

                // Newton gradient and Hessian matrix
                const auto [newton_grad, Hessian, b] = Newton_grad_right(source_paired, transformed_s_paired, target_paired);

                // SO3-> so3
                pose_particles_ = torch::cat({t_.view({particle_size_, 3}).transpose(0, 1), rotm_to_ypr_tensor(R_).transpose(0, 1)}, 0);
                torch::Tensor stein_grad = torch::zeros_like(newton_grad) ;

                // check particle size
                if (particle_size_ > 1){
                    if (config_.SVN_full_grad) {
                        stein_grad = svn_full_grad(pose_particles_.transpose(0, 1), Hessian, -b);
                    }
                    else {
                        Hessian_mean = torch::mean(Hessian, 0).expand({particle_size_, 6, 6});
                        stein_grad = svgd_grad(pose_particles_.transpose(0, 1), -newton_grad, Hessian_mean);
                    }
                }
                else
                    stein_grad = -newton_grad.reshape({1,6});

                // Update pose
                this->pose_update(stein_grad);

                // Early stop
                if(config_.check_early_stop ){
                    if(torch::lt(stein_grad.norm(2,1).mean(0),
                                 torch::tensor({config_.convergence_threshold}
                                 ).to(torch::kCUDA)).to(torch::kCPU).item<bool>()){
                        std::cout << "Align process converged in epoch:" << epoch << std::endl;
                        break;
                    }
                }
                // save convergence process
                pose_particles_ = torch::cat({t_.view({particle_size_, 3}).transpose(0, 1), rotm_to_ypr_tensor(R_).transpose(0, 1)}, 0);
                particle_stack_[epoch] = pose_particles_.view({6, particle_size_});
            }

            //6*particle_size
            pose_particles_ = torch::cat({t_.view({particle_size_, 3}).transpose(0, 1), rotm_to_ypr_tensor(R_).transpose(0, 1)}, 0);
            return SteinICPState::ALIGN_SUCCESS;
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SteinMICP::Newton_grad_right(const torch::Tensor &source_paired,
                                                                              const torch::Tensor &transformed_s_paired,
                                                                              const torch::Tensor &target_paired)
        {
            auto error = transformed_s_paired - target_paired;   ///particle*config_.batch_size*3
            const auto error_squared = torch::norm(error, 2, 2, true);
            //        // M-estimator to filter outliers
            const auto weight= torch::square(config_.max_dist / (config_.max_dist + 3 * error_squared)) ;
            error = weight * error;

            const auto zero = torch::zeros({particle_size_,config_.batch_size}).to(torch::kCUDA).to(data_type);
            const auto s_hat = torch::stack({
                                                    torch::stack({zero,
                                                                  - source_paired.index({Slice(), Slice(), 2}),
                                                                  source_paired.index({Slice(), Slice(), 1})}, 2),
                                                    torch::stack({source_paired.index({Slice(), Slice(), 2}),
                                                                  zero,
                                                                  - source_paired.index({Slice(), Slice(), 0})},2),
                                                    torch::stack({-source_paired.index({Slice(), Slice(), 1}),
                                                                  source_paired.index({Slice(), Slice(), 0}),
                                                                  zero}, 2)
                                            },2);


            auto R_compound = R0_.matmul(R_).unsqueeze(1).expand({particle_size_, config_.batch_size, 3, 3});
            const auto J = torch::cat({R_compound, -R_compound.matmul(s_hat)}, 3);
            const auto J_T = J.transpose(2,3);

            const auto H = torch::einsum("pbik,pbil->pkl", {J,
                                                            J.mul(weight.view({particle_size_,config_.batch_size,1,1})) })
                           + 1e-6*torch::eye(6).to(torch::kCUDA).to(data_type) ;
            const auto b = torch::einsum("pbik, pbij->pk", {J,
                                                            error.view({particle_size_, config_.batch_size,3,1})});

            //// H: [particles, 6, 6]
            //// b: [particles, 6, 1]
            //// Newton_grad: [particles, 6, 1]
            const auto Newton_grad = torch::linalg::solve(H, b, true);
            return {Newton_grad, H, b};
        }

        torch::Tensor SteinMICP::to_rotation_tensor(const torch::Tensor &r, const torch::Tensor &p, const torch::Tensor &y) {

            auto zeros = torch::zeros({particle_size_}).to(torch::kCUDA).to(data_type);
            auto angle_axis = torch::cat({r, p, y}, 1).view({particle_size_, 3});

            auto angle = torch::norm(angle_axis, 2, 1).view({particle_size_, 1});
            auto zero_index = torch::lt(angle, 1e-12).view({particle_size_,1});
            auto axis = torch::where(zero_index, torch::zeros({particle_size_,3}).to(torch::kCUDA).to(data_type), angle_axis.div(angle));
            auto cos_a = torch::cos(angle);
            auto sin_a = torch::sin(angle);
            auto a_hat = torch::stack({
                                              torch::stack({zeros, -axis.index({Slice(), 2}), axis.index({Slice(), 1})}, 1),
                                              torch::stack({axis.index({Slice(), 2}), zeros, -axis.index({Slice(), 0})}, 1),
                                              torch::stack({-axis.index({Slice(), 1}), axis.index({Slice(), 0}), zeros}, 1)
                                      }, 2).transpose(1,2);

            torch::Tensor R = torch::mul(cos_a.view({particle_size_, 1, 1}),
                                         torch::eye(3).expand({particle_size_, 3, 3}).to(torch::kCUDA).to(data_type))
                              + (1 - cos_a).view({particle_size_, 1, 1}).mul(
                    torch::matmul(axis.view({particle_size_, 3, 1}), axis.view({particle_size_, 1, 3})))
                              + sin_a.view({particle_size_, 1, 1}).mul(a_hat);

            J_l_ = torch::mul((sin_a.div(angle)).view({particle_size_,1,1}),
                              torch::eye(3).expand({particle_size_, 3, 3}).to(torch::kCUDA)).to(data_type)
                   + (1-sin_a.div(angle)).view({particle_size_,1,1}).mul(
                    torch::matmul(axis.view({particle_size_, 3, 1}), axis.view({particle_size_, 1, 3})))
                   + (1-cos_a).div(angle).view({particle_size_,1,1}).mul(a_hat);
            return R;
        }

        torch::Tensor SteinMICP::rotm_to_ypr_tensor(const torch::Tensor &R) {

            auto angle = torch::acos(
                    torch::clip(
                            0.5 * (R.index({Slice(), 0, 0}) + R.index({Slice(), 1, 1}) + R.index({Slice(), 2, 2}) - 1),
                            -1, 1)
            ).view({particle_size_, 1});

            auto sin_angle = torch::sin(angle);

            auto nonzero_mask = (sin_angle.abs() > 1e-12).view({particle_size_,1});
            torch::Tensor angle_axis = torch::zeros({particle_size_, 3}).to(torch::kCUDA).to(data_type);
            angle_axis = 0.5 / sin_angle.view({particle_size_,1}).masked_fill(~nonzero_mask, 1) * angle
                         * torch::stack({
                                                R.index({Slice(), 2, 1}) - R.index({Slice(), 1, 2}),
                                                R.index({Slice(), 0, 2}) - R.index({Slice(), 2, 0}),
                                                R.index({Slice(), 1, 0}) - R.index({Slice(), 0, 1})
                                        }, 1).view({particle_size_, 3});
            angle_axis = torch::masked_fill(angle_axis, ~nonzero_mask, 0);
            return angle_axis;

        }


        torch::Tensor SteinMICP::svgd_grad(const torch::Tensor& pose_parameters, const torch::Tensor &newton_grad, const torch::Tensor &H)
        {

            const auto [Kernel, bandwidth, pair_difference] = rbf_hessian_kernel(pose_parameters, H);
            const auto grad = 2/bandwidth*torch::mul(pair_difference.view({particle_size_,particle_size_,6}),
                                               Kernel.view({particle_size_, particle_size_, 1})).sum(1);

            // K(i,j) * gradient + grad_K(i,j)
//            std::cout << "Kernel sum " << Kernel.sum(1, true)<< std::endl;
            return (Kernel.matmul(newton_grad) +
            torch::matmul(torch::linalg::inv(H), grad.view({particle_size_,6,1})).squeeze(2)
//            / source_cloud_.size(0)
            )/ Kernel.sum(1, true);
        }

        torch::Tensor SteinMICP::svn_full_grad(const torch::Tensor &pose_parameters, const torch::Tensor &H,
                                               const torch::Tensor &b)
        {
            ////Kernel: [p, p]
            const auto [Kernel, bandwidth, pair_difference] = rbf_hessian_kernel(pose_parameters, H);
            const auto grad = 2/bandwidth*torch::mul(pair_difference.view({particle_size_,particle_size_,6}),
                                                     Kernel.view({particle_size_, particle_size_, 1})).unsqueeze(-1);
            ////H_svn: [p, 6, 6]
            const auto grad2 = torch::matmul(grad.view({particle_size_, particle_size_, 6, 1}), grad.view({particle_size_, particle_size_, 1, 6})).sum(1);
            const auto Kernel_square = Kernel.square();
            const auto H_mean = ((
                                         Kernel_square.unsqueeze(-1).unsqueeze(-1)*      // Comment this line: becomes Average of Hessian
                                         H.unsqueeze(0)).sum(1)
                                 + grad2) / particle_size_;                                         // Comment this line: omit the second order of kernel

            const auto svgd_update = (Kernel.matmul(b.squeeze()).view({particle_size_, 6, 1}) + grad.sum(1)) / particle_size_;

            // auto update =  torch::linalg::inv(H_mean).matmul(Kernel.matmul(b.squeeze()).view({particle_size_, 6, 1}) + grad.sum(1)

            const auto kernel_sum = Kernel.sum(1,true);

            const auto update =  config_.lr * torch::linalg::inv(H_mean).matmul(svgd_update
                    //                          /source_cloud_.size(0)
            ).squeeze()
            /// kernel_sum
            ;
            //
            return update;
        }

         std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SteinMICP::rbf_hessian_kernel(const torch::Tensor& x1, const torch::Tensor& H)
        {
             // || xi - xj ||
             const auto pair_difference = (x1.view({particle_size_, 1, 6}) - x1);
             // square
             const auto pairwise_square_norm =  torch::mul(pair_difference,
                                                             pair_difference.transpose(1,2).transpose(1,2)).sum(2) ;
             // bandwidth
             const auto h = torch::median(pairwise_square_norm) / log(x1.size(0) + 1);
             // Kernel
             auto Kernel = torch::exp(-pairwise_square_norm / h);
             return {Kernel, h, pair_difference};
        }

        void SteinMICP::pose_update(const torch::Tensor &stein_grad)
        {
            auto d_R = to_rotation_tensor(stein_grad.index({Slice(), 3}).reshape({particle_size_, 1, 1}),
                                          stein_grad.index({Slice(), 4}).reshape({particle_size_, 1, 1}),
                                          stein_grad.index({Slice(), 5}).reshape({particle_size_, 1, 1}));
            auto d_t = to_translation_tensor(stein_grad.index({Slice(), 0}).reshape({particle_size_, 1, 1}),
                                             stein_grad.index({Slice(), 1}).reshape({particle_size_, 1, 1}),
                                             stein_grad.index({Slice(), 2}).reshape({particle_size_, 1, 1}));
            d_t = J_l_.matmul(d_t);
            // update pose
            R_ = R_.matmul(d_R);
            t_ = R_.matmul(d_t) + t_;
        }

        std::vector<double> SteinMICP::get_particle_weight()
        {
                auto weight_cpu = particle_weight_.to(torch::kCPU);
                return std::vector<double>(weight_cpu.data_ptr<float>(), weight_cpu.data_ptr<float>()+particle_size_);
        }

        torch::Tensor SteinMICP::get_transformation()
        {
    //        return torch::mean(pose_particles_, 1);
                auto weighted_mean = torch::mul(pose_particles_, particle_weight_.transpose(0, 1)).sum(1);
                return weighted_mean;
        }

        torch::Tensor SteinMICP::get_distribution() {
            auto weight_mean = this->get_transformation();
            auto var = torch::mul((pose_particles_ - weight_mean.view({6,1})).square(), particle_weight_.transpose(0, 1)).sum(1);
            return var;
        }

        std::vector<double> SteinMICP::get_cov_matrix()
        {

            auto weight_mean = this->get_transformation();
            auto difference = pose_particles_ - weight_mean.view({6,1});
            auto Sigma = torch::sum(particle_weight_.view({particle_size_,1,1})
                                 *torch::matmul(difference.transpose(0,1).view({particle_size_,6,1}),
                                               difference.transpose(0,1).view({particle_size_,1,6})), 0);
            auto Sigma_cpu = Sigma.view({36}).to(torch::kCPU);
            auto Sigma_vector = std::vector<double>(Sigma_cpu.data_ptr<double>(), Sigma_cpu.data_ptr<double>()+36);
            return Sigma_vector;
        }


}