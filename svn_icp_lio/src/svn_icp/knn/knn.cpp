//
// Created by haoming on 20.07.23.
//



#include "knn.h"

std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdx(
        const at::Tensor& p1,
        const at::Tensor& p2,
        const at::Tensor& lengths1,
        const at::Tensor& lengths2,
        const int norm,
        const int K,
        const int version) {
    if (p1.is_cuda() || p2.is_cuda()) {
#ifdef WITH_CUDA
        CHECK_CUDA(p1);
        CHECK_CUDA(p2);
        return KNearestNeighborIdxCuda(
                p1, p2, lengths1, lengths2, norm, K, version);
#else
        AT_ERROR("Not compiled with GPU support.");
#endif
    }
    return KNearestNeighborIdxCpu(p1, p2, lengths1, lengths2, norm, K);
}


std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackward(
        const at::Tensor& p1,
        const at::Tensor& p2,
        const at::Tensor& lengths1,
        const at::Tensor& lengths2,
        const at::Tensor& idxs,
        const int norm,
        const at::Tensor& grad_dists) {
    if (p1.is_cuda() || p2.is_cuda()) {
#ifdef WITH_CUDA
        CHECK_CUDA(p1);
        CHECK_CUDA(p2);
        return KNearestNeighborBackwardCuda(
                p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
#else
        AT_ERROR("Not compiled with GPU support.");
#endif
    }
    return KNearestNeighborBackwardCpu(
            p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
}


