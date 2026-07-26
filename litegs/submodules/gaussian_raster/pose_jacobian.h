#pragma once
#include <torch/extension.h>

at::Tensor pose_jacobian_backward(
    at::Tensor xyz_h,
    at::Tensor view_matrix,
    at::Tensor proj_matrix,
    at::Tensor transform_matrix,
    at::Tensor cov2d_inv,
    at::Tensor grad_ndc,
    at::Tensor grad_cov2d_inv,
    at::Tensor grad_dir,
    int64_t img_h,
    int64_t img_w
);
