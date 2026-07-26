#ifndef __CUDACC__
    #define __CUDACC__
    #define __NVCC__
#endif

#include <ATen/core/TensorAccessor.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuda_errchk.h"
#include "pose_jacobian.h"

namespace {

__device__ __forceinline__ void atomic_add_se3(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
    int batch_id,
    float gvx,
    float gvy,
    float gvz,
    float gwx,
    float gwy,
    float gwz)
{
    atomicAdd(&out[batch_id][0], gvx);
    atomicAdd(&out[batch_id][1], gvy);
    atomicAdd(&out[batch_id][2], gvz);
    atomicAdd(&out[batch_id][3], gwx);
    atomicAdd(&out[batch_id][4], gwy);
    atomicAdd(&out[batch_id][5], gwz);
}

__device__ __forceinline__ void add_xcam_to_se3(
    const float x,
    const float y,
    const float z,
    const float gx,
    const float gy,
    const float gz,
    float& gvx,
    float& gvy,
    float& gvz,
    float& gwx,
    float& gwy,
    float& gwz)
{
    // Left perturbation: x_cam' = exp(xi^) x_cam, xi = [v, w].
    // dx_cam / dv = I, dx_cam / dw = -[x_cam]_x.
    gvx += gx;
    gvy += gy;
    gvz += gz;
    gwx += y * gz - z * gy;
    gwy += z * gx - x * gz;
    gwz += x * gy - y * gx;
}

__global__ void pose_jacobian_backward_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> xyz_h,          // [4,P]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> view_matrix,    // [N,4,4], LiteGS transposed layout
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> proj_matrix,    // [N,4,4], LiteGS transposed layout
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> transform,      // [3,3,P]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      // [N,2,2,P]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_ndc,       // [N,3,P]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> grad_cov2d_inv, // [N,2,2,P]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_dir,       // [N,3,P]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_pose,            // [N,6]
    const int img_h,
    const int img_w)
{
    const int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = blockIdx.y;
    if (batch_id >= view_matrix.size(0) || point_id >= xyz_h.size(1)) {
        return;
    }

    const float wx = xyz_h[0][point_id];
    const float wy = xyz_h[1][point_id];
    const float wz = xyz_h[2][point_id];

    // LiteGS stores matrices transposed and multiplies with matrix.transpose().
    const float r00 = view_matrix[batch_id][0][0];
    const float r01 = view_matrix[batch_id][1][0];
    const float r02 = view_matrix[batch_id][2][0];
    const float r10 = view_matrix[batch_id][0][1];
    const float r11 = view_matrix[batch_id][1][1];
    const float r12 = view_matrix[batch_id][2][1];
    const float r20 = view_matrix[batch_id][0][2];
    const float r21 = view_matrix[batch_id][1][2];
    const float r22 = view_matrix[batch_id][2][2];
    const float tx = view_matrix[batch_id][3][0];
    const float ty = view_matrix[batch_id][3][1];
    const float tz = view_matrix[batch_id][3][2];

    const float x = r00 * wx + r01 * wy + r02 * wz + tx;
    const float y = r10 * wx + r11 * wy + r12 * wz + ty;
    const float z_raw = r20 * wx + r21 * wy + r22 * wz + tz;
    const float z = fmaxf(z_raw, 1.0e-2f);

    float gvx = 0.0f, gvy = 0.0f, gvz = 0.0f;
    float gwx = 0.0f, gwy = 0.0f, gwz = 0.0f;

    // d(pixel)/d(ndc) * d(ndc)/d(x_cam)
    {
        const float a00 = proj_matrix[batch_id][0][0];
        const float a01 = proj_matrix[batch_id][1][0];
        const float a02 = proj_matrix[batch_id][2][0];
        const float a03 = proj_matrix[batch_id][3][0];
        const float a10 = proj_matrix[batch_id][0][1];
        const float a11 = proj_matrix[batch_id][1][1];
        const float a12 = proj_matrix[batch_id][2][1];
        const float a13 = proj_matrix[batch_id][3][1];
        const float a20 = proj_matrix[batch_id][0][2];
        const float a21 = proj_matrix[batch_id][1][2];
        const float a22 = proj_matrix[batch_id][2][2];
        const float a23 = proj_matrix[batch_id][3][2];
        const float a30 = proj_matrix[batch_id][0][3];
        const float a31 = proj_matrix[batch_id][1][3];
        const float a32 = proj_matrix[batch_id][2][3];
        const float a33 = proj_matrix[batch_id][3][3];

        const float h0 = a00 * x + a01 * y + a02 * z_raw + a03;
        const float h1 = a10 * x + a11 * y + a12 * z_raw + a13;
        const float h2 = a20 * x + a21 * y + a22 * z_raw + a23;
        const float h3 = a30 * x + a31 * y + a32 * z_raw + a33 + 1.0e-7f;
        const float inv_h3 = 1.0f / h3;
        const float inv_h3_sq = inv_h3 * inv_h3;

        const float g0 = grad_ndc[batch_id][0][point_id];
        const float g1 = grad_ndc[batch_id][1][point_id];
        const float g2 = grad_ndc[batch_id][2][point_id];

        const float gx = g0 * (a00 * h3 - h0 * a30) * inv_h3_sq
                       + g1 * (a10 * h3 - h1 * a30) * inv_h3_sq
                       + g2 * (a20 * h3 - h2 * a30) * inv_h3_sq;
        const float gy = g0 * (a01 * h3 - h0 * a31) * inv_h3_sq
                       + g1 * (a11 * h3 - h1 * a31) * inv_h3_sq
                       + g2 * (a21 * h3 - h2 * a31) * inv_h3_sq;
        const float gz = g0 * (a02 * h3 - h0 * a32) * inv_h3_sq
                       + g1 * (a12 * h3 - h1 * a32) * inv_h3_sq
                       + g2 * (a22 * h3 - h2 * a32) * inv_h3_sq;
        add_xcam_to_se3(x, y, z_raw, gx, gy, gz, gvx, gvy, gvz, gwx, gwy, gwz);
    }

    // d(pixel)/d(color) * d(color)/d(dir) * d(dir)/d(camera center) * d(camera center)/d(se3).
    {
        const float ccx = -(tx * r00 + ty * r10 + tz * r20);
        const float ccy = -(tx * r01 + ty * r11 + tz * r21);
        const float ccz = -(tx * r02 + ty * r12 + tz * r22);
        const float qx = wx - ccx;
        const float qy = wy - ccy;
        const float qz = wz - ccz;
        const float q_norm = fmaxf(sqrtf(qx * qx + qy * qy + qz * qz), 1.0e-8f);
        const float inv_norm = 1.0f / q_norm;
        const float dxn = qx * inv_norm;
        const float dyn = qy * inv_norm;
        const float dzn = qz * inv_norm;
        const float gdx = grad_dir[batch_id][0][point_id];
        const float gdy = grad_dir[batch_id][1][point_id];
        const float gdz = grad_dir[batch_id][2][point_id];
        const float dot = gdx * dxn + gdy * dyn + gdz * dzn;
        const float gqx = (gdx - dot * dxn) * inv_norm;
        const float gqy = (gdy - dot * dyn) * inv_norm;
        const float gqz = (gdz - dot * dzn) * inv_norm;

        // q = X - C, and for left perturbation dC = -R^T dv.
        gvx += r00 * gqx + r10 * gqy + r20 * gqz;
        gvy += r01 * gqx + r11 * gqy + r21 * gqz;
        gvz += r02 * gqx + r12 * gqy + r22 * gqz;
    }

    // d(pixel)/d(cov2d_inv) * d(cov2d_inv)/d(cov2d) * d(cov2d)/d(J) * d(J)/d(x_cam) * d(x_cam)/d(se3).
    {
        const float c00 = cov2d_inv[batch_id][0][0][point_id];
        const float c01 = cov2d_inv[batch_id][0][1][point_id];
        const float c10 = cov2d_inv[batch_id][1][0][point_id];
        const float c11 = cov2d_inv[batch_id][1][1][point_id];
        const float gci00 = grad_cov2d_inv[batch_id][0][0][point_id];
        const float gci01 = grad_cov2d_inv[batch_id][0][1][point_id];
        const float gci10 = grad_cov2d_inv[batch_id][1][0][point_id];
        const float gci11 = grad_cov2d_inv[batch_id][1][1][point_id];

        // grad_cov = -C^{-T} grad_inv C^{-T}; C is symmetric but keep full form.
        const float t00 = gci00 * c00 + gci01 * c01;
        const float t01 = gci00 * c10 + gci01 * c11;
        const float t10 = gci10 * c00 + gci11 * c01;
        const float t11 = gci10 * c10 + gci11 * c11;
        const float gc00 = -(c00 * t00 + c01 * t10);
        const float gc01 = -(c00 * t01 + c01 * t11);
        const float gc10 = -(c10 * t00 + c11 * t10);
        const float gc11 = -(c10 * t01 + c11 * t11);

        const float out_fx = proj_matrix[batch_id][0][0] * img_w * 0.5f;
        const float out_fy = proj_matrix[batch_id][1][1] * img_h * 0.5f;

        // Reconstruct the same ray-space J up to the normalized focal scale used by jacobianRayspace.
        const float j00 = out_fx / z;
        const float j11 = out_fy / z;
        const float j20 = -out_fx * x / (z * z);
        const float j21 = -out_fy * y / (z * z);

        const float m00 = transform[0][0][point_id] * r00 + transform[0][1][point_id] * r10 + transform[0][2][point_id] * r20;
        const float m01 = transform[0][0][point_id] * r01 + transform[0][1][point_id] * r11 + transform[0][2][point_id] * r21;
        const float m02 = transform[0][0][point_id] * r02 + transform[0][1][point_id] * r12 + transform[0][2][point_id] * r22;
        const float m10 = transform[1][0][point_id] * r00 + transform[1][1][point_id] * r10 + transform[1][2][point_id] * r20;
        const float m11 = transform[1][0][point_id] * r01 + transform[1][1][point_id] * r11 + transform[1][2][point_id] * r21;
        const float m12 = transform[1][0][point_id] * r02 + transform[1][1][point_id] * r12 + transform[1][2][point_id] * r22;
        const float m20 = transform[2][0][point_id] * r00 + transform[2][1][point_id] * r10 + transform[2][2][point_id] * r20;
        const float m21 = transform[2][0][point_id] * r01 + transform[2][1][point_id] * r11 + transform[2][2][point_id] * r21;
        const float m22 = transform[2][0][point_id] * r02 + transform[2][1][point_id] * r12 + transform[2][2][point_id] * r22;

        const float a00 = m00 * j00 + m02 * j20;
        const float a10 = m10 * j00 + m12 * j20;
        const float a20 = m20 * j00 + m22 * j20;
        const float a01 = m01 * j11 + m02 * j21;
        const float a11 = m11 * j11 + m12 * j21;
        const float a21 = m21 * j11 + m22 * j21;

        const float ga00 = 2.0f * (a00 * gc00 + a01 * gc10);
        const float ga10 = 2.0f * (a10 * gc00 + a11 * gc10);
        const float ga20 = 2.0f * (a20 * gc00 + a21 * gc10);
        const float ga01 = 2.0f * (a00 * gc01 + a01 * gc11);
        const float ga11 = 2.0f * (a10 * gc01 + a11 * gc11);
        const float ga21 = 2.0f * (a20 * gc01 + a21 * gc11);

        const float gj00 = m00 * ga00 + m10 * ga10 + m20 * ga20;
        const float gj11 = m01 * ga01 + m11 * ga11 + m21 * ga21;
        const float gj20 = m02 * ga00 + m12 * ga10 + m22 * ga20;
        const float gj21 = m02 * ga01 + m12 * ga11 + m22 * ga21;

        float gx = gj20 * (-out_fx / (z * z));
        float gy = gj21 * (-out_fy / (z * z));
        float gz = gj00 * (-out_fx / (z * z))
                 + gj11 * (-out_fy / (z * z))
                 + gj20 * (2.0f * out_fx * x / (z * z * z))
                 + gj21 * (2.0f * out_fy * y / (z * z * z));
        add_xcam_to_se3(x, y, z_raw, gx, gy, gz, gvx, gvy, gvz, gwx, gwy, gwz);
    }

    atomic_add_se3(grad_pose, batch_id, gvx, gvy, gvz, gwx, gwy, gwz);
}

} // namespace

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
    int64_t img_w)
{
    at::DeviceGuard guard(xyz_h.device());
    const int64_t views = view_matrix.size(0);
    const int64_t points = xyz_h.size(1);
    at::Tensor grad_pose = torch::zeros({views, 6}, xyz_h.options());

    const int threads = 256;
    dim3 blocks(std::ceil(points / static_cast<float>(threads)), views, 1);
    pose_jacobian_backward_kernel<<<blocks, threads>>>(
        xyz_h.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        view_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        proj_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        transform_matrix.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_dir.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_pose.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        static_cast<int>(img_h),
        static_cast<int>(img_w));
    CUDA_CHECK_ERRORS;
    return grad_pose;
}
