#include "jabobi.h"

// Spherical harmonics constants (same as in transform.cu)
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

// Device constants for spherical harmonics coefficients
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};

__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

template <typename scalar_t, int degree>
__global__ void rgb2dir_jacobian_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dirs,        //[batch,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_base,    //[1,3,point_num] 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> SH_rest,    //[(deg+1)²-1,3,point_num] 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> jacobian          //[batch,3,3,point_num] (dRGB/ddir)
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y;
    
    if (batch_id >= dirs.size(0) || index >= dirs.size(2)) return;
    
    // Initialize Jacobian components: dRGB_i/d(x,y,z) for i=0,1,2 (RGB channels)
    scalar_t dRGB_dx[3] = {0, 0, 0};
    scalar_t dRGB_dy[3] = {0, 0, 0};  
    scalar_t dRGB_dz[3] = {0, 0, 0};
    
    // Get direction components
    scalar_t x = dirs[batch_id][0][index];
    scalar_t y = dirs[batch_id][1][index];
    scalar_t z = dirs[batch_id][2][index];
    
    // Degree 0: constant term (no direction dependence)
    // dRGB/d(x,y,z) = 0 for degree 0
    
    if (degree > 0) {
        // Degree 1: linear terms
        // Y₁₋₁ = -SH_C1 * y, Y₁⁰ = SH_C1 * z, Y₁¹ = -SH_C1 * x
        
        // Derivatives of SH basis functions
        // d(-SH_C1 * y)/dx = 0, d(-SH_C1 * y)/dy = -SH_C1, d(-SH_C1 * y)/dz = 0
        // d(SH_C1 * z)/dx = 0,  d(SH_C1 * z)/dy = 0,       d(SH_C1 * z)/dz = SH_C1
        // d(-SH_C1 * x)/dx = -SH_C1, d(-SH_C1 * x)/dy = 0, d(-SH_C1 * x)/dz = 0
        
        for (int c = 0; c < 3; c++) {
            dRGB_dx[c] += -SH_C1 * SH_rest[2][c][index];  // from Y₁¹ term
            dRGB_dy[c] += -SH_C1 * SH_rest[0][c][index];  // from Y₁₋₁ term  
            dRGB_dz[c] += SH_C1 * SH_rest[1][c][index];   // from Y₁⁰ term
        }
        
        if (degree > 1) {
            // Degree 2: quadratic terms
            scalar_t xx = x * x, yy = y * y, zz = z * z;
            scalar_t xy = x * y, yz = y * z, xz = x * z;
            
            // Y₂₋₂ = SH_C2[0] * xy
            // Y₂₋₁ = SH_C2[1] * yz  
            // Y₂⁰ = SH_C2[2] * (2z² - x² - y²)
            // Y₂¹ = SH_C2[3] * xz
            // Y₂² = SH_C2[4] * (x² - y²)
            
            for (int c = 0; c < 3; c++) {
                dRGB_dx[c] += (
                    SH_C2[0] * y * SH_rest[3][c][index] +           // d(xy)/dx = y
                    SH_C2[2] * (-2.0f * x) * SH_rest[5][c][index] + // d(2z²-x²-y²)/dx = -2x
                    SH_C2[3] * z * SH_rest[6][c][index] +           // d(xz)/dx = z
                    SH_C2[4] * (2.0f * x) * SH_rest[7][c][index]    // d(x²-y²)/dx = 2x
                );
                
                dRGB_dy[c] += (
                    SH_C2[0] * x * SH_rest[3][c][index] +           // d(xy)/dy = x
                    SH_C2[1] * z * SH_rest[4][c][index] +           // d(yz)/dy = z
                    SH_C2[2] * (-2.0f * y) * SH_rest[5][c][index] + // d(2z²-x²-y²)/dy = -2y
                    SH_C2[4] * (-2.0f * y) * SH_rest[7][c][index]   // d(x²-y²)/dy = -2y
                );
                
                dRGB_dz[c] += (
                    SH_C2[1] * y * SH_rest[4][c][index] +           // d(yz)/dz = y
                    SH_C2[2] * (4.0f * z) * SH_rest[5][c][index] +  // d(2z²-x²-y²)/dz = 4z
                    SH_C2[3] * x * SH_rest[6][c][index]             // d(xz)/dz = x
                );
            }
            
            if (degree > 2) {
                // Degree 3: cubic terms - more complex derivatives
                for (int c = 0; c < 3; c++) {
                    // Simplified derivatives for degree 3 terms
                    // Y₃₋₃ = SH_C3[0] * y * (3x² - y²)
                    dRGB_dx[c] += SH_C3[0] * SH_rest[8][c][index] * (6.0f * x * y);
                    dRGB_dy[c] += SH_C3[0] * SH_rest[8][c][index] * (3.0f * xx - 3.0f * yy);
                    
                    // Y₃₋₂ = SH_C3[1] * xyz
                    dRGB_dx[c] += SH_C3[1] * SH_rest[9][c][index] * (y * z);
                    dRGB_dy[c] += SH_C3[1] * SH_rest[9][c][index] * (x * z);
                    dRGB_dz[c] += SH_C3[1] * SH_rest[9][c][index] * (x * y);
                    
                    // Y₃₋₁ = SH_C3[2] * y * (4z² - x² - y²)
                    dRGB_dx[c] += SH_C3[2] * SH_rest[10][c][index] * (-2.0f * x * y);
                    dRGB_dy[c] += SH_C3[2] * SH_rest[10][c][index] * (4.0f * zz - xx - 3.0f * yy);
                    dRGB_dz[c] += SH_C3[2] * SH_rest[10][c][index] * (8.0f * y * z);
                    
                    // Y₃⁰ = SH_C3[3] * z * (2z² - 3x² - 3y²)
                    dRGB_dx[c] += SH_C3[3] * SH_rest[11][c][index] * (-6.0f * x * z);
                    dRGB_dy[c] += SH_C3[3] * SH_rest[11][c][index] * (-6.0f * y * z);
                    dRGB_dz[c] += SH_C3[3] * SH_rest[11][c][index] * (6.0f * zz - 3.0f * xx - 3.0f * yy);
                    
                    // Y₃¹ = SH_C3[4] * x * (4z² - x² - y²)
                    dRGB_dx[c] += SH_C3[4] * SH_rest[12][c][index] * (4.0f * zz - 3.0f * xx - yy);
                    dRGB_dy[c] += SH_C3[4] * SH_rest[12][c][index] * (-2.0f * x * y);
                    dRGB_dz[c] += SH_C3[4] * SH_rest[12][c][index] * (8.0f * x * z);
                    
                    // Y₃² = SH_C3[5] * z * (x² - y²)
                    dRGB_dx[c] += SH_C3[5] * SH_rest[13][c][index] * (2.0f * x * z);
                    dRGB_dy[c] += SH_C3[5] * SH_rest[13][c][index] * (-2.0f * y * z);
                    dRGB_dz[c] += SH_C3[5] * SH_rest[13][c][index] * (xx - yy);
                    
                    // Y₃³ = SH_C3[6] * x * (x² - 3y²)
                    dRGB_dx[c] += SH_C3[6] * SH_rest[14][c][index] * (3.0f * xx - 3.0f * yy);
                    dRGB_dy[c] += SH_C3[6] * SH_rest[14][c][index] * (-6.0f * x * y);
                }
            }
        }
    }
    
    // Store Jacobian matrix [3x3] for this point
    // jacobian[batch][rgb_channel][dir_component][point]
    for (int c = 0; c < 3; c++) {
        jacobian[batch_id][c][0][index] = dRGB_dx[c];  // dRGB_c/dx
        jacobian[batch_id][c][1][index] = dRGB_dy[c];  // dRGB_c/dy
        jacobian[batch_id][c][2][index] = dRGB_dz[c];  // dRGB_c/dz
    }
}

// Main function to calculate Jacobian from RGB colors to directions
torch::Tensor rgb2dir_jacobian(int64_t degree, torch::Tensor dirs, torch::Tensor SH_base, torch::Tensor SH_rest) 
{
    // Input validation
    TORCH_CHECK(dirs.dim() == 3, "dirs must be 3D tensor [batch, 3, points]");
    TORCH_CHECK(SH_base.dim() == 3, "SH_base must be 3D tensor [1, 3, points]");
    TORCH_CHECK(SH_rest.dim() == 3, "SH_rest must be 3D tensor [sh_coeffs, 3, points]");
    
    int batch_size = dirs.size(0);
    int num_points = dirs.size(2);
    
    // Output Jacobian tensor: [batch, 3_rgb, 3_dir, points]
    torch::Tensor jacobian = torch::zeros({batch_size, 3, 3, num_points}, dirs.options());
    
    // CUDA kernel configuration
    int threads_per_block = 256;
    int blocks_x = (num_points + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block);
    dim3 grid_size(blocks_x, batch_size);
    
    // Launch appropriate kernel based on degree
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dirs.scalar_type(), "rgb2dir_jacobian", [&] {
        switch (degree) {
            case 0:
                rgb2dir_jacobian_kernel<scalar_t, 0><<<grid_size, block_size>>>(
                    dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_base.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_rest.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    jacobian.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
                );
                break;
            case 1:
                rgb2dir_jacobian_kernel<scalar_t, 1><<<grid_size, block_size>>>(
                    dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_base.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_rest.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    jacobian.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
                );
                break;
            case 2:
                rgb2dir_jacobian_kernel<scalar_t, 2><<<grid_size, block_size>>>(
                    dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_base.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_rest.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    jacobian.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
                );
                break;
            case 3:
                rgb2dir_jacobian_kernel<scalar_t, 3><<<grid_size, block_size>>>(
                    dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_base.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    SH_rest.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    jacobian.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
                );
                break;
            default:
                TORCH_CHECK(false, "Unsupported spherical harmonics degree: ", degree);
        }
    });
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return jacobian;
}