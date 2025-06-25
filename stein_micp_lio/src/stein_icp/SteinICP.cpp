//
// Created by haoming on 26.05.23.
//


#include <memory>
#include "SteinICP.h"


namespace stein_icp {

    SteinICP::SteinICP(SteinICPParam &parameters, const torch::Tensor &init_pose) {
        particle_size_ = init_pose.size(1);
        x_ = init_pose[0].view({particle_size_, 1, 1});
        y_ = init_pose[1].view({particle_size_, 1, 1});
        z_ = init_pose[2].view({particle_size_, 1, 1});
        rx_ = init_pose[3].view({particle_size_, 1, 1});
        ry_ = init_pose[4].view({particle_size_, 1, 1});
        rz_ = init_pose[5].view({particle_size_, 1, 1});

        config_ = parameters;
        normalize_factor_ = torch::tensor({1.0}).to(torch::kCUDA).to(data_type);
        pose_particles_ = torch::stack({x_ * normalize_factor_, y_ * normalize_factor_, z_ * normalize_factor_, rx_, ry_, rz_}).squeeze({2, 3});
        R_ = to_rotation_tensor(rx_, ry_, rz_); //p*3*3
        t_ = to_translation_tensor(x_, y_, z_); //p*3*1
        R0_ = torch::eye(3, torch::TensorOptions().dtype(data_type).device(torch::kCUDA));
        t0_ = torch::zeros({3,1}, torch::TensorOptions().dtype(data_type).device(torch::kCUDA));
        R_total_ = R0_.matmul(R_);
        t_total_ = t0_ + R0_.matmul(t_);
        finish_iter_ = config_.iterations;
        K_source_ = parameters.KNN_count;

    }

    void SteinICP::add_cloud(const torch::Tensor &source, const torch::Tensor &target, const torch::Tensor &init_pose) {
        particle_size_ = init_pose.size(1);
        x_ = init_pose[0].view({particle_size_, 1, 1});
        y_ = init_pose[1].view({particle_size_, 1, 1});
        z_ = init_pose[2].view({particle_size_, 1, 1});
        rx_ = init_pose[3].view({particle_size_, 1, 1});
        ry_ = init_pose[4].view({particle_size_, 1, 1});
        rz_ = init_pose[5].view({particle_size_, 1, 1});

        target_cloud_ = target.clone();
        source_cloud_ = source.clone();

        gradient_scaling_factor_ = torch::tensor({source_cloud_.size(0)}).to(torch::kCUDA).to(data_type);

        R_ = to_rotation_tensor(rx_, ry_, rz_);
        t_ = to_translation_tensor(x_, y_, z_);
    }


    /*Stein ICP process*/
    SteinICPState SteinICP::stein_align() {
        knn_duration_ = 0;
        update_duration_ = 0;
        auto pose_particles_old = pose_particles_.clone();
        auto pose_difference = torch::zeros_like(pose_particles_);

        // optimizer
        this->set_optimizer();
        if (!optimizer_set_)
            return SteinICPState::NO_OPTIMIZER;

        this->allocate_memory_for_log();
        // mini-batch
        const auto [mini_batch_all_epochs, target_batch] = mini_batch_pair_generator();
        /** SVGD iterations */
        for (int epoch = 0; epoch < config_.iterations; epoch++) {
            //expand minibatch and target to match the particle size
            torch::Tensor mini_batch_epoch = mini_batch_all_epochs[epoch].expand({particle_size_, config_.batch_size, 3}).to(torch::kCUDA).to(data_type);
            torch::Tensor target_batch_it = target_batch[epoch];

            // uptate the full pose
            R_ = to_rotation_tensor(rx_, ry_, rz_);
            t_ = to_translation_tensor(x_, y_, z_);
            R_total_ = R0_.matmul(R_);
            t_total_ = t0_ + R0_.matmul(t_);

            // transform points from all batches according to the predicted states
            torch::Tensor source_transformed = mini_batch_epoch.matmul(R_total_.transpose(1, 2)) + t_total_.view({particle_size_, 1, 3});

            knn_timer_.reset();
            // find corresponding points
            const auto [source_paired, transformed_s_paired, target_paired] =
                    get_correspondence_fast(mini_batch_epoch, source_transformed, target_batch_it);
            knn_duration_ +=  knn_timer_.duration();
            update_timer_.reset();

            // ICP Gradient
            const auto sgd_gradient = sgd_grad(source_paired, transformed_s_paired, target_paired);

            torch::Tensor stein_grad = torch::zeros_like(sgd_gradient) ;
            if (particle_size_ > 1)
                stein_grad = svgd_grad(pose_particles_.transpose(0,1), -sgd_gradient);
            else
                stein_grad = -sgd_gradient;

            pose_particles_old = pose_particles_.clone();
            this->pose_update(stein_grad);
            update_duration_ += update_timer_.duration();

            pose_particles_ = torch::stack({x_ * normalize_factor_, y_ * normalize_factor_, z_ * normalize_factor_, rx_, ry_, rz_})
                    .squeeze({2, 3});
            pose_difference = pose_particles_ - pose_particles_old;
            if (config_.check_early_stop) {
                    if(torch::lt(pose_difference.norm(2,0).mean(0),
                                 torch::tensor({config_.convergence_threshold}
                                 ).to(torch::kCUDA)).to(torch::kCPU).item<bool>()){
                        std::cout << "Align process converged in epoch:" << epoch << std::endl;
                        finish_iter_ = epoch + 1;
                        break;
                    }
            }

            particle_stack_[epoch] = pose_particles_.view({6, particle_size_});
        }
        //6 * particle_size
        pose_particles_ = torch::stack({x_ * normalize_factor_, y_ * normalize_factor_, z_ * normalize_factor_, rx_, ry_, rz_}).squeeze({2, 3});
        return SteinICPState::ALIGN_SUCCESS;
    }

    void SteinICP::set_optimizer() {
        auto pose_vector = {x_, y_, z_, rx_, ry_, rz_};
        if (config_.optimizer == "Adam") {
            std::cout << "Optimization using: " << config_.optimizer << std::endl;
            torch::optim::AdamOptions adamopt(config_.lr);
            adamopt.betas(std::make_tuple(0.9, 0.999));
            optimizer_ptr_ = std::make_unique<torch::optim::Adam>(pose_vector, adamopt);
            optimizer_set_ = true;
        } else if (config_.optimizer == "RMSprop") {
            std::cout << "Optimization using: " << config_.optimizer << std::endl;
            torch::optim::RMSpropOptions rmspropopt(config_.lr);
            rmspropopt.weight_decay(1e-8);
            rmspropopt.momentum(0.9);
            optimizer_ptr_ = std::make_unique<torch::optim::RMSprop>(pose_vector, rmspropopt);
            optimizer_set_ = true;
        } else if (config_.optimizer == "SGD") {
            std::cout << "Optimization using: " << config_.optimizer << std::endl;
            optimizer_ptr_ = std::make_unique<torch::optim::SGD>(pose_vector, torch::optim::SGDOptions(config_.lr));
            optimizer_set_ = true;
        } else if (config_.optimizer == "Adagrad") {
            std::cout << "Optimization using: " << config_.optimizer << std::endl;
            optimizer_ptr_ = std::make_unique<torch::optim::Adagrad>(pose_vector,
                                                                     torch::optim::AdagradOptions(config_.lr));
            optimizer_set_ = true;
        } else {
            std::cout << "No optimizer chosen" << std::endl;
            optimizer_set_ = false;
        }
    }

    void SteinICP::allocate_memory_for_log() {
        particle_stack_ = torch::zeros({config_.iterations, 6, particle_size_}).to(torch::kCUDA);
    }

    std::tuple<torch::Tensor, torch::Tensor> SteinICP::mini_batch_pair_generator()
    {
        torch::Tensor idx = torch::tensor({}).to(torch::kCUDA).to(data_type);
        if (config_.use_minibatch)
            idx = torch::randint(source_cloud_.size(0), config_.batch_size * config_.iterations).to(torch::kCUDA);
        else{
            config_.batch_size = source_cloud_.size(0);
            idx = torch::arange(source_cloud_.size(0)).reshape({config_.batch_size,1})
                    .expand({config_.batch_size, config_.iterations}).transpose(0,1)
                    .reshape({config_.batch_size * config_.iterations}).to(torch::kCUDA) ;
        }
//        std::cout << "idx \n" << idx << std::endl;
        torch::Tensor mini_batch = torch::empty({config_.batch_size * config_.iterations, 3}).to(torch::kCUDA).to(data_type);
        mini_batch = torch::index_select(source_cloud_, 0, idx);
        this->knn_source_cloud();
        auto sourceKNN_batch_idx = torch::index_select(sourceKNN_idx_, 0, idx).view(
                {config_.batch_size * config_.iterations * K_source_}).to(torch::kCUDA).to(torch::kLong);
        auto target_batch = torch::index_select(target_cloud_, 0, sourceKNN_batch_idx).to(data_type);
        //std::cout<<mini_batch<<std::endl;
        return {mini_batch.reshape({config_.iterations, config_.batch_size, 3}),
                target_batch.reshape({config_.iterations, config_.batch_size, K_source_, 3})};
    }

    void SteinICP::knn_source_cloud()
    {
        const auto N_s = source_cloud_.size(0);
        const auto N_t = target_cloud_.size(0);
        auto transformed_source = source_cloud_.matmul(R0_.transpose(0,1)) + t0_.view({1,3});
        auto knn = KNearestNeighborIdx(
                transformed_source.view({1, N_s, 3}),
                target_cloud_.view({1, N_t, 3}),
                torch::tensor({N_s}).to(torch::kCUDA),
                torch::tensor({N_t}).to(torch::kCUDA),
                2,
                K_source_,
                -1
        );
        sourceKNN_idx_ = std::get<0>(knn).to(torch::kCUDA).reshape({N_s, K_source_});

    }

    torch::Tensor SteinICP::mini_batch_generator() {
        const auto idx = torch::randint(source_cloud_.size(0), config_.batch_size * config_.iterations).to(torch::kCUDA);
        torch::Tensor mini_batch = torch::empty({config_.batch_size * config_.iterations, 3}).to(torch::kCUDA).to(data_type);
        mini_batch = torch::index_select(source_cloud_, 0, idx);
        //std::cout<<mini_batch<<std::endl;
        return mini_batch.reshape({config_.iterations, config_.batch_size, 3});
    }

    torch::Tensor SteinICP::to_rotation_tensor(const torch::Tensor &r, const torch::Tensor &p, const torch::Tensor &y) {
        const auto Cosyaw = torch::cos(y);
        const auto Sinyaw = torch::sin(y);
        const auto Cospitch = torch::cos(p);
        const auto Sinpitch = torch::sin(p);
        const auto Cosroll = torch::cos(r);
        const auto Sinroll = torch::sin(r);
        torch::Tensor R = torch::stack(
                {torch::stack(
                        { Cospitch*Cosyaw,
                          Sinroll*Sinpitch*Cosyaw - Cosroll*Sinyaw,
                          Sinroll*Sinyaw + Cosroll*Sinpitch*Cosyaw
                        }, 3
                ).squeeze(1),
                 torch::stack(
                         { Cospitch*Sinyaw,
                           Cosroll*Cosyaw + Sinroll*Sinpitch*Sinyaw,
                           Cosroll*Sinpitch*Sinyaw - Sinroll*Cosyaw
                         }, 3
                 ).squeeze(1),
                 torch::stack(
                         { -Sinpitch,
                           Sinroll*Cospitch,
                           Cosroll*Cospitch
                         }, 3
                 ).squeeze(1)
                }, 2
        ).squeeze(1);

        return R;
    }

    torch::Tensor SteinICP::to_translation_tensor(const torch::Tensor &x, const torch::Tensor &y, const torch::Tensor &z) {
        return torch::cat({x, y, z}, 1);
    }

    torch::Tensor SteinICP::rotm_to_ypr_tensor(const torch::Tensor &R) {
        const auto yaw = torch::atan2(R.index({Slice(), 1, 0}), R.index({Slice(), 0,0})).view({particle_size_, 1});
        const auto pitch = torch::atan2(-R.index({Slice(), 2, 0}),
                                   torch::sqrt(1-R.index({Slice(),2,0}).square())).view({particle_size_,1});
        const auto roll = torch::atan2(R.index({Slice(), 2, 1}), R.index({Slice(), 2, 2})).view({particle_size_, 1});
        return torch::cat({roll, pitch, yaw}, 1);

    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SteinICP::get_correspondence(
            const torch::Tensor &source,                /// particle*config_.batch_size * 3
            const torch::Tensor &transformed_source,    /// particle*config_.batch_size * 3
            const torch::Tensor &target                 /// target_point_count * 3
    ) {
        std::vector<torch::Tensor> cloud_paired;
        const auto knn = KNearestNeighborIdx( transformed_source,
                                              target.expand({particle_size_, target.size(0), 3}).to(torch::kCUDA).to(data_type),
                                              config_.batch_size*torch::ones_like(x_).reshape({particle_size_,1}).to(torch::kLong),
                                              target.size(0)*torch::ones_like(x_).reshape({particle_size_,1}).to(torch::kLong),
                                              2,
                                              1,
                                              -1);
        const auto dist_cr = std::get<1>(knn).to(torch::kCUDA);
        const auto target_index = std::get<0>(knn).to(torch::kCUDA).reshape({particle_size_, config_.batch_size});

        return {point_filter(source, dist_cr),
                point_filter(transformed_source, dist_cr),
                point_filter(target.index({target_index}), dist_cr)};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SteinICP::get_correspondence_fast(
                                const torch::Tensor &source,                /// particle*config_.batch_size * 3
                                const torch::Tensor &transformed_source,    /// particle*config_.batch_size * 3
                                const torch::Tensor &target                 /// target_point_count * 3
                        )
    {
        const auto knn = KNearestNeighborIdx(transformed_source.transpose(0, 1),
                                             target,
                                             particle_size_ * torch::ones({config_.batch_size, 1}).to(torch::kLong).to(torch::kCUDA),
                                             K_source_ * torch::ones({config_.batch_size, 1}).to(torch::kLong).to(torch::kCUDA),
                                             2,
                                             1,
                                             -1);
        const auto dist_cr = std::get<1>(knn).transpose(0, 1).to(torch::kCUDA);
        const auto target_index = std::get<0>(knn).transpose(0, 1).to(torch::kCUDA).reshape({particle_size_, config_.batch_size});

        return {
                point_filter(source, dist_cr),
                point_filter(transformed_source, dist_cr),
                point_filter((target.transpose(0, 1)).index({target_index,
                                                             torch::arange(config_.batch_size).view({config_.batch_size}).to(
                                                                     torch::kCUDA)}),
                             dist_cr)
        };
    }

    torch::Tensor SteinICP::point_filter(const torch::Tensor &cloud, const torch::Tensor &distance_cr) {
        return torch::lt(distance_cr, config_.max_dist / normalize_factor_).to(torch::kCUDA) * cloud;
    }

    torch::Tensor SteinICP::partial_derivative(const torch::Tensor &roll,
                                               const torch::Tensor &pitch,
                                               const torch::Tensor &yaw) {

        const auto Cosyaw = torch::cos(yaw);  // A
        const auto Sinyaw = torch::sin(yaw);  // B
        const auto Cospitch = torch::cos(pitch); // C
        const auto Sinpitch = torch::sin(pitch); // D
        const auto Cosroll = torch::cos(roll); // E
        const auto Sinroll = torch::sin(roll); // F

        const auto Sinpitch_Cosroll = Sinpitch * Cosroll;  // DE
        const auto Sinpitch_Sinroll = Sinpitch * Sinroll;  // DF
        const auto Cosyaw_Cospitch = Cosyaw * Cospitch;  // AC
        const auto Cosyaw_Sinroll = Cosyaw * Sinroll;  // AF
        const auto Cosyaw_Cosroll = Cosyaw * Cosroll;  // AE

        const auto Cosyaw_Sinpitch_Cosroll = Cosyaw * Sinpitch_Cosroll;  // ADE
        const auto Cosyaw_Sinpitch_Sinroll = Cosyaw * Sinpitch_Sinroll;  // ADF
        const auto Sinyaw_Cospitch = Sinyaw * Cospitch;  // BC
        const auto Sinyaw_Cosroll = Sinyaw * Cosroll;  // BE
        const auto Sinyaw_Sinroll = Sinyaw * Sinroll;  // BF
        const auto Sinyaw_Sinpitch_Cosroll = Sinyaw * Sinpitch_Cosroll; // BDE


        const auto torch_0 = torch::zeros_like(roll).to(torch::kCUDA).to(data_type);
        torch::Tensor partial_roll = torch::stack(
                {torch::stack({torch_0, Cosyaw_Sinpitch_Cosroll + Sinyaw_Sinroll, Sinyaw_Cosroll - Cosyaw_Sinpitch_Sinroll}, 3).squeeze(1),
                 torch::stack({torch_0, -Cosyaw_Sinroll + Sinyaw_Sinpitch_Cosroll, Sinyaw * (-Sinpitch_Sinroll) - Cosyaw_Cosroll}, 3).squeeze(1),
                 torch::stack({torch_0, Cospitch * Cosroll, Cospitch * (-Sinroll)}, 3).squeeze(1)
                 },
                 2).squeeze(1);

        torch::Tensor partial_pitch = torch::stack(
                {torch::stack({Cosyaw * -Sinpitch, Cosyaw_Cospitch * Sinroll, Cosyaw_Cospitch * Cosroll}, 3).squeeze(1),
                 torch::stack({Sinyaw * -Sinpitch, Sinyaw_Cospitch * Sinroll, Sinyaw_Cospitch * Cosroll}, 3).squeeze(1),
                 torch::stack({-Cospitch, -Sinpitch_Sinroll, -Sinpitch_Cosroll}, 3).squeeze(1)
                 },
                 2).squeeze(1);

        torch::Tensor partial_yaw = torch::stack(
                {torch::stack({-Sinyaw_Cospitch, -Sinyaw * Sinpitch_Sinroll - Cosyaw_Cosroll,Cosyaw_Sinroll - Sinyaw_Sinpitch_Cosroll}, 3).squeeze(1),
                 torch::stack({Cosyaw_Cospitch, -Sinyaw_Cosroll + Cosyaw_Sinpitch_Sinroll,Cosyaw_Sinpitch_Cosroll + Sinyaw_Sinroll}, 3).squeeze(1),
                 torch::stack({torch_0, torch_0, torch_0}, 3).squeeze(1),
                },
                2).squeeze(1);

        return torch::stack({R0_.matmul(partial_roll), R0_.matmul(partial_pitch), R0_.matmul(partial_yaw)});
    }

    torch::Tensor SteinICP::sgd_grad(const torch::Tensor &source_paired,
                                     const torch::Tensor &transformed_s_paired,
                                     const torch::Tensor &target_paired) {
        const auto partial_derivatives = partial_derivative(rx_, ry_, rz_);
        c10::optional<int64_t> dim = 1;

        const auto nonzero_count = (torch::count_nonzero(transformed_s_paired.sum(2), dim).to(torch::kCUDA)).to(data_type);  ///particle*1

        auto error = transformed_s_paired - target_paired;   ///particle*config_.batch_size*3
        torch::Tensor sgd_gradient = torch::zeros({particle_size_, 6}).to(torch::kCUDA).to(data_type);
//
        const auto error_squared = torch::norm(error, 2, 2, true);
        const auto weighted_error = torch::square(config_.max_dist / (config_.max_dist + 3 * error_squared)) * error;
        error = weighted_error.clone();
        // gradient of xyz
        sgd_gradient.index_put_({Slice(), Slice(0, 3)}, error.sum(1).matmul(R0_) / (nonzero_count +
                                                                        torch::ones_like(nonzero_count).to(
                                                                                torch::kCUDA).to(data_type)).reshape({particle_size_, 1}));
        // gradient of roll
        sgd_gradient.index_put_({Slice(), 3}, torch::einsum("pbc, pbc->pb",
                                                            {
                                                                    error,
                                                                    torch::einsum("prc, pbc->pbr",
                                                                                  {partial_derivatives[0],
                                                                                   source_paired})
                                                            }).sum(1) /
                                              (nonzero_count + torch::ones_like(nonzero_count).to(torch::kCUDA))) / normalize_factor_;
        // gradient of pitch
        sgd_gradient.index_put_({Slice(), 4}, torch::einsum("pbc, pbc->pb",
                                                            {
                                                                    error,
                                                                    torch::einsum("prc, pbc->pbr",
                                                                                  {partial_derivatives[1],
                                                                                   source_paired})
                                                            }).sum(1) /
                                              (nonzero_count + torch::ones_like(nonzero_count).to(torch::kCUDA).to(data_type))) / normalize_factor_;
        // gradient of yaw
        sgd_gradient.index_put_({Slice(), 5}, torch::einsum("pbc, pbc->pb",
                                                            {
                                                                    error,
                                                                    torch::einsum("prc, pbc->pbr",
                                                                                  {partial_derivatives[2],
                                                                                   source_paired})
                                                            }).sum(1) /
                                              (nonzero_count + torch::ones_like(nonzero_count).to(torch::kCUDA).to(data_type))) / normalize_factor_;

        return sgd_gradient * gradient_scaling_factor_;
    }

    torch::Tensor SteinICP::svgd_grad(const torch::Tensor& pose_parameters, const torch::Tensor &sgd_grad) {
        const auto [Kernel, bandwidth, pair_difference] = rbf_kernel(pose_parameters);
        const auto grad = 2/bandwidth*torch::mul(pair_difference.view({particle_size_,particle_size_,6}),
                                           Kernel.view({particle_size_, particle_size_, 1})).sum(1);
        return (Kernel.matmul(sgd_grad) + grad) / pose_parameters.size(0);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SteinICP::rbf_kernel(const torch::Tensor &x) {
        const auto pair_difference = (x.view({particle_size_, 1, 6}) - x);
        // square
        const auto pairwise_dist =  torch::mul(pair_difference,
                                         pair_difference.transpose(1,2).transpose(1,2)).sum(2) ;
//            auto pairwise_dist =  torch::mul(pair_difference,
//                                             torch::matmul(H, pair_difference.transpose(1,2)).transpose(1,2)).sum(2) ;
        const auto h = torch::median(pairwise_dist) / log(x.size(0) + 1);
        auto Kernel = torch::exp(-pairwise_dist / h);
        return {Kernel, h, pair_difference};
    }

    void SteinICP::pose_update(const torch::Tensor &stein_grad) {
        x_.mutable_grad() = -stein_grad.index({Slice(), 0}).reshape({particle_size_, 1, 1});
        y_.mutable_grad() = -stein_grad.index({Slice(), 1}).reshape({particle_size_, 1, 1});
        z_.mutable_grad() = -stein_grad.index({Slice(), 2}).reshape({particle_size_, 1, 1});
        rx_.mutable_grad() = -stein_grad.index({Slice(), 3}).reshape({particle_size_, 1, 1});
        ry_.mutable_grad() = -stein_grad.index({Slice(), 4}).reshape({particle_size_, 1, 1});
        rz_.mutable_grad() = -stein_grad.index({Slice(), 5}).reshape({particle_size_, 1, 1});

        x_last_ = x_.clone();
        y_last_ = y_.clone();
        z_last_ = z_.clone();
        roll_last_ = rx_.clone();
        pitch_last_ = ry_.clone();
        yaw_last_ = rz_.clone();

        optimizer_ptr_->step();
        optimizer_ptr_->zero_grad();
        optimizer_ptr_->state();
    }


    torch::Tensor SteinICP::get_transformation() {
        return torch::mean(pose_particles_, 1);
    }

    torch::Tensor SteinICP::get_distribution() {
        return torch::var(pose_particles_, 1) ;
    }

    std::vector<double> SteinICP::get_cov_matrix() {

        auto mean = this->get_transformation();
        auto difference = pose_particles_ - mean.view({6, 1});
        auto Sigma = torch::mean(torch::matmul(difference.transpose(0,1).view({particle_size_,6,1}),
                                               difference.transpose(0,1).view({particle_size_,1,6})), 0);
        auto Sigma_cpu = Sigma.view({36}).to(torch::kCPU);
        auto Sigma_vector = std::vector<double>(Sigma_cpu.data_ptr<double>(), Sigma_cpu.data_ptr<double>()+36);
        return Sigma_vector;
    }

    std::vector<double> SteinICP::get_particles() {
        auto particle_pose_cpu = pose_particles_.to(torch::kCPU);
        auto Result_particle = std::vector<double>(particle_pose_cpu.data_ptr<double>(),
                                                   particle_pose_cpu.data_ptr<double>() + 6 * particle_size_);
        return Result_particle;
    }

    std::vector<double> SteinICP::get_particle_weight(){
        return std::vector(particle_size_, 1.0);
    }

    std::vector<std::vector<float>> SteinICP::get_particle_history() {
        auto particle_stack_cpu = particle_stack_.to(torch::kCPU);
        std::vector<std::vector<float>> particles;
        for (int i = 0; i < config_.iterations; i++) {
            particles.emplace_back(particle_stack_cpu[i].data_ptr<float>(),
                                   particle_stack_cpu[i].data_ptr<float>() + particle_size_ * 6);
        }
        return particles;
    }



}
