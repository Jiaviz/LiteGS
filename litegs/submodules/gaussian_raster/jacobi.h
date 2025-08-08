#pragma once
#include <torch/extension.h>

/**
 * Calculate Jacobian matrix from RGB colors to direction vectors.
 * 
 * This function computes the partial derivatives dRGB/ddir for spherical harmonics
 * rendering, which is useful for pose optimization and direction-dependent shading.
 * 
 * @param degree Spherical harmonics degree (0-3 supported)
 * @param dirs Direction vectors [batch, 3, num_points]
 * @param SH_base Base SH coefficients [1, 3, num_points] 
 * @param SH_rest Rest SH coefficients [(degree+1)²-1, 3, num_points]
 * @return Jacobian matrix [batch, 3_rgb, 3_dir, num_points]
 */
torch::Tensor rgb2dir_jacobian(int64_t degree, torch::Tensor dirs, torch::Tensor SH_base, torch::Tensor SH_rest);
