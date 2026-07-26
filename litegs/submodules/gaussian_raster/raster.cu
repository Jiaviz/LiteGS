#ifndef __CUDACC__
    #define __CUDACC__
    #define __NVCC__
#endif
#include "cuda_runtime.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
namespace cg = cooperative_groups;

#include <c10/cuda/CUDAException.h>
#include <ATen/core/TensorAccessor.h>

#include "cuda_errchk.h"
#include "raster.h"


template <int tilesize,bool enable_trans,bool enable_depth>
__global__ void raster_forward_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_img,    //[batch,3,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_transmitance,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> output_depth,     //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> output_last_contributor,    //[batch,tile,tilesize,tilesize]
    int tiles_num_x,int img_h,int img_w
)
{


    __shared__ float2 collected_xy[tilesize * tilesize / 4];
    __shared__ float collected_depth[tilesize * tilesize / 4];
    __shared__ float collected_opacity[tilesize * tilesize / 4];
    __shared__ float3 collected_cov2d_inv[tilesize * tilesize / 4];
    __shared__ float3 collected_color[tilesize * tilesize / 4];

    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0)
    {
        tile_id = specific_tiles[batch_id][blockIdx.x];
    }

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id-1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id-1) / tiles_num_x) * tilesize + y_in_tile;

    if (tile_id!=0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];

        float transmittance = 1.0f;
        float inv_depth = 0.0f;
        bool done = false;
        float3 final_color{ 0,0,0 };
        int last_contributor = 0;
        if (start_index_in_tile != -1)
        {
            for (int offset = start_index_in_tile; offset < end_index_in_tile; offset += tilesize * tilesize / 4)
            {
                int num_done = __syncthreads_count(done);
                if (num_done == blockDim.x * blockDim.y)
                    break;

                int valid_num = min(tilesize * tilesize / 4, end_index_in_tile - offset);
                //load to shared memory
                if (threadIdx.y * blockDim.x + threadIdx.x < valid_num)
                {
                    int i = threadIdx.y * blockDim.x + threadIdx.x;
                    int index = offset + i;
                    int point_id = sorted_points[batch_id][index];
                    collected_xy[i].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                    collected_xy[i].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                    if (enable_depth)
                    {
                        collected_depth[i] = ndc[batch_id][2][point_id];
                    }
                    collected_cov2d_inv[i].x = cov2d_inv[batch_id][0][0][point_id];
                    collected_cov2d_inv[i].y = cov2d_inv[batch_id][0][1][point_id];
                    collected_cov2d_inv[i].z = cov2d_inv[batch_id][1][1][point_id];

                    collected_color[i].x = color[batch_id][0][point_id];
                    collected_color[i].y = color[batch_id][1][point_id];
                    collected_color[i].z = color[batch_id][2][point_id];
                    collected_opacity[i] = opacity[0][point_id];
                }
                __syncthreads();

                //process
                for (int i = 0; i < valid_num && done == false; i++)
                {

                    float2 xy = collected_xy[i];
                    float2 d = { xy.x - pixel_x,xy.y - pixel_y };
                    float3 cur_color = collected_color[i];
                    float cur_opacity = collected_opacity[i];
                    float3 cur_cov2d_inv = collected_cov2d_inv[i];

                    float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                    if (power > 0.0f)
                        continue;

                    float alpha = min(0.99f, cur_opacity * exp(power));
                    if (alpha < 1.0f / 255.0f)
                        continue;

                    if (transmittance * (1 - alpha) < 0.0001f)
                    {
                        done = true;
                        continue;
                    }

                    final_color.x += cur_color.x * alpha * transmittance;
                    final_color.y += cur_color.y * alpha * transmittance;
                    final_color.z += cur_color.z * alpha * transmittance;
                    if (enable_depth)
                    {
                        inv_depth += collected_depth[i] * alpha * transmittance;
                    }
                    transmittance *= (1 - alpha);
                    last_contributor = offset + i;


                }
                __syncthreads();
            }
        }
        output_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = final_color.x;
        output_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile] = final_color.y;
        output_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile] = final_color.z;
        output_transmitance[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = transmittance;
        if (enable_depth)
        {
            output_depth[batch_id][0][blockIdx.x][y_in_tile][x_in_tile] = inv_depth;
        }

        output_last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile] = last_contributor;
        
    }
}

#define RASTER_SWITCH_CASE(TILESIZE,TRANS,DEPTH) case ((TILESIZE<<8)+(TRANS<<1)+DEPTH)
#define FORWARD_CASE_KERNEL(TILESIZE,TRANS,DEPTH) RASTER_SWITCH_CASE(TILESIZE,TRANS,DEPTH):\
raster_forward_kernel<TILESIZE,TRANS,DEPTH>
#define FORWARD_KERNEL_ARGS sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),\
color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),\
specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
output_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_depth.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
output_last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
tilesnum_x, img_h, img_w


std::vector<at::Tensor> rasterize_forward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor  ndc,// 
    at::Tensor  cov2d_inv,
    at::Tensor  color,
    at::Tensor  opacity,
    std::optional<at::Tensor>  specific_tiles_arg,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w,
    bool enable_trans,
    bool enable_depth
)
{
    at::DeviceGuard guard( ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize));
    int tilesnum_y = std::ceil(img_h / float(tilesize));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }
    

    torch::TensorOptions opt_img = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(true);
    at::Tensor output_img = torch::empty({ viewsnum,3, tilesnum,tilesize,tilesize }, opt_img);

    torch::TensorOptions opt_t = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(start_index.device()).requires_grad(enable_trans);
    at::Tensor output_transmitance = torch::empty({ viewsnum,1, tilesnum, tilesize, tilesize }, opt_t);

    at::Tensor output_depth = torch::empty({ 0,0, 0, 0, 0 }, opt_t);
    if (enable_depth)
    {
        output_depth = torch::empty({ viewsnum,1, tilesnum, tilesize, tilesize }, opt_t.requires_grad(true));
    }

    std::vector<int64_t> shape_c{ viewsnum, tilesnum, tilesize, tilesize };
    torch::TensorOptions opt_c = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(start_index.device()).requires_grad(false);
    at::Tensor output_last_contributor = torch::empty(shape_c, opt_c);



    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);
    int template_code = (tilesize << 8) + (enable_trans << 1) + (enable_depth);
    switch (template_code)
    {
        FORWARD_CASE_KERNEL(8,true,true) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(8, false, true) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(8, true, false) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(8, false, false) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(16, true, true) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(16, false, true) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(16, true, false) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;
        FORWARD_CASE_KERNEL(16, false, false) << <Block3d, Thread3d >> > (FORWARD_KERNEL_ARGS);
        break;

    default:
        assert(false);

    }
    CUDA_CHECK_ERRORS;
    
    return { output_img ,output_transmitance,output_depth ,output_last_contributor };
}

// Forward-mode derivative of alpha compositing. Each thread owns one output
// pixel and propagates all six se(3) tangent directions in registers. The
// Gaussian screen-space tangents are analytical and supplied by the caller.
template <int tilesize>
__global__ void raster_pose_jacobian_kernel(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dndc_dpose,
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> dcov_dpose,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> dcolor_dpose,
    torch::PackedTensorAccessor32<float, 6, torch::RestrictPtrTraits> output,
    int tiles_num_x, int img_h, int img_w)
{
    constexpr int chunk = tilesize * tilesize / 4;
    __shared__ int point_ids[chunk];
    __shared__ float2 means[chunk];
    __shared__ float3 inv_covs[chunk];
    __shared__ float3 colors[chunk];
    __shared__ float opacities[chunk];
    __shared__ float mean_jacs[chunk][2][6];
    __shared__ float cov_jacs[chunk][3][6];
    __shared__ float color_jacs[chunk][3][6];

    const int batch = blockIdx.y;
    const int tile_index = blockIdx.x;
    const int tile_id = tile_index + 1;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int linear_thread = ty * tilesize + tx;
    const int pixel_x = (tile_index % tiles_num_x) * tilesize + tx;
    const int pixel_y = (tile_index / tiles_num_x) * tilesize + ty;
    const bool valid_pixel = pixel_x < img_w && pixel_y < img_h;

    float transmittance = 1.0f;
    float dtrans[6] = {0, 0, 0, 0, 0, 0};
    float dimage[3][6] = {{0}};
    bool done = !valid_pixel;

    if (tile_id < start_index.size(1) - 1) {
        const int begin = start_index[batch][tile_id];
        const int end = start_index[batch][tile_id + 1];
        if (begin != -1) {
            for (int offset = begin; offset < end; offset += chunk) {
                const int valid_count = min(chunk, end - offset);
                if (linear_thread < valid_count) {
                    const int p = sorted_points[batch][offset + linear_thread];
                    point_ids[linear_thread] = p;
                    means[linear_thread] = make_float2(
                        (ndc[batch][0][p] + 1.0f) * 0.5f * img_w - 0.5f,
                        (ndc[batch][1][p] + 1.0f) * 0.5f * img_h - 0.5f);
                    inv_covs[linear_thread] = make_float3(
                        cov2d_inv[batch][0][0][p], cov2d_inv[batch][0][1][p],
                        cov2d_inv[batch][1][1][p]);
                    colors[linear_thread] = make_float3(
                        color[batch][0][p], color[batch][1][p], color[batch][2][p]);
                    opacities[linear_thread] = opacity[0][p];
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) {
                        mean_jacs[linear_thread][0][k] = dndc_dpose[batch][0][p][k] * 0.5f * img_w;
                        mean_jacs[linear_thread][1][k] = dndc_dpose[batch][1][p][k] * 0.5f * img_h;
                        cov_jacs[linear_thread][0][k] = dcov_dpose[batch][0][0][p][k];
                        cov_jacs[linear_thread][1][k] = dcov_dpose[batch][0][1][p][k];
                        cov_jacs[linear_thread][2][k] = dcov_dpose[batch][1][1][p][k];
                        color_jacs[linear_thread][0][k] = dcolor_dpose[batch][0][p][k];
                        color_jacs[linear_thread][1][k] = dcolor_dpose[batch][1][p][k];
                        color_jacs[linear_thread][2][k] = dcolor_dpose[batch][2][p][k];
                    }
                }
                __syncthreads();

                for (int i = 0; i < valid_count && !done; ++i) {
                    const float dx = means[i].x - pixel_x;
                    const float dy = means[i].y - pixel_y;
                    const float3 Cinv = inv_covs[i];
                    const float power = -0.5f * (Cinv.x * dx * dx + Cinv.z * dy * dy)
                                      - Cinv.y * dx * dy;
                    if (power > 0.0f) continue;
                    const float raw_alpha = opacities[i] * expf(power);
                    const float alpha = min(0.99f, raw_alpha);
                    if (alpha < 1.0f / 255.0f) continue;
                    if (transmittance * (1.0f - alpha) < 0.0001f) {
                        done = true;
                        continue;
                    }

                    float dalpha[6];
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) {
                        const float dmx = mean_jacs[i][0][k];
                        const float dmy = mean_jacs[i][1][k];
                        const float dpower =
                            -0.5f * (cov_jacs[i][0][k] * dx * dx + cov_jacs[i][2][k] * dy * dy)
                            -cov_jacs[i][1][k] * dx * dy
                            -(Cinv.x * dx + Cinv.y * dy) * dmx
                            -(Cinv.z * dy + Cinv.y * dx) * dmy;
                        dalpha[k] = raw_alpha < 0.99f ? alpha * dpower : 0.0f;
                    }

                    const float3 rgb = colors[i];
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) {
                        dimage[0][k] += color_jacs[i][0][k] * alpha * transmittance
                                      + rgb.x * (dalpha[k] * transmittance + alpha * dtrans[k]);
                        dimage[1][k] += color_jacs[i][1][k] * alpha * transmittance
                                      + rgb.y * (dalpha[k] * transmittance + alpha * dtrans[k]);
                        dimage[2][k] += color_jacs[i][2][k] * alpha * transmittance
                                      + rgb.z * (dalpha[k] * transmittance + alpha * dtrans[k]);
                        dtrans[k] = dtrans[k] * (1.0f - alpha) - transmittance * dalpha[k];
                    }
                    transmittance *= 1.0f - alpha;
                }
                __syncthreads();
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < 3; ++c)
        #pragma unroll
        for (int k = 0; k < 6; ++k)
            output[batch][c][tile_index][ty][tx][k] = dimage[c][k];
}

at::Tensor rasterize_pose_jacobian(
    at::Tensor sorted_points, at::Tensor start_index, at::Tensor ndc,
    at::Tensor cov2d_inv, at::Tensor color, at::Tensor opacity,
    at::Tensor dndc_dpose, at::Tensor dcov2d_inv_dpose,
    at::Tensor dcolor_dpose, int64_t tilesize, int64_t img_h, int64_t img_w)
{
    at::DeviceGuard guard(ndc.device());
    const int64_t views = start_index.size(0);
    const int tiles_x = std::ceil(img_w / float(tilesize));
    const int tiles_y = std::ceil(img_h / float(tilesize));
    const int tiles = tiles_x * tiles_y;
    at::Tensor output = torch::zeros({views, 3, tiles, tilesize, tilesize, 6}, ndc.options());
    dim3 blocks(tiles, views, 1);
    dim3 threads(tilesize, tilesize, 1);
    if (tilesize == 8) {
        raster_pose_jacobian_kernel<8><<<blocks, threads>>>(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            dndc_dpose.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            dcov2d_inv_dpose.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            dcolor_dpose.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<float, 6, torch::RestrictPtrTraits>(),
            tiles_x, img_h, img_w);
    } else if (tilesize == 16) {
        raster_pose_jacobian_kernel<16><<<blocks, threads>>>(
            sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            dndc_dpose.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            dcov2d_inv_dpose.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            dcolor_dpose.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<float, 6, torch::RestrictPtrTraits>(),
            tiles_x, img_h, img_w);
    } else {
        TORCH_CHECK(false, "rasterize_pose_jacobian supports tile sizes 8 and 16");
    }
    CUDA_CHECK_ERRORS;
    return output;
}

template <int tilesize,bool enable_trans_grad,bool enable_depth_grad>
__global__ void raster_backward_kernel_warp_reduction(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w
)
{
    __shared__ float4 collected_color[tilesize * tilesize];
    __shared__ float3 collected_invcov[tilesize * tilesize];
    __shared__ float2 collected_mean[tilesize * tilesize + int(enable_depth_grad) * tilesize * tilesize / 2];
    float* collected_depth = (float*)(collected_mean + tilesize * tilesize);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int threadidx = threadIdx.y * blockDim.x + threadIdx.x;
    const int x_in_tile = (warp.meta_group_rank() % (tilesize / 8)) * 8 + warp.thread_rank() % 8;
    const int y_in_tile = (warp.meta_group_rank() / (tilesize / 8)) * 4 + warp.thread_rank() / 8;

    const int batch_id = blockIdx.y;
    const int tile_index = blockIdx.x;
    int tile_id = blockIdx.x + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0)
    {
        tile_id = specific_tiles[batch_id][blockIdx.x];
    }


    if (tile_id != 0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];
        if (start_index_in_tile != -1)
        {
            float f_transmittance = final_transmitance[batch_id][0][tile_index][y_in_tile][x_in_tile];
            float transmittance = f_transmittance;
            int pixel_lst_index = last_contributor[batch_id][tile_index][y_in_tile][x_in_tile];

            float3 d_pixel{ 0,0,0 };
            float d_trans_pixel = 0;
            float d_depth_pixel = 0;
            int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
            int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + y_in_tile;
            if (pixel_x < img_w && pixel_y < img_h)
            {
                d_pixel.x = d_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                d_pixel.y = d_img[batch_id][1][tile_index][y_in_tile][x_in_tile];
                d_pixel.z = d_img[batch_id][2][tile_index][y_in_tile][x_in_tile];
                if (enable_trans_grad)
                {
                    d_trans_pixel = d_trans_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                }
                if (enable_depth_grad)
                {
                    d_depth_pixel = d_depth_img[batch_id][0][tile_index][y_in_tile][x_in_tile];
                }
            }
            //loop points
            float3 accum_rec{ 0,0,0 };
            float accum_depth = 0;
            for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset -= (tilesize * tilesize))
            {
                int collected_num = min(tilesize * tilesize, offset - start_index_in_tile + 1);
                if (threadIdx.y * blockDim.x + threadIdx.x < collected_num)
                {
                    int index = offset - threadidx;
                    int point_id = sorted_points[batch_id][index];

                    collected_mean[threadidx].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                    collected_mean[threadidx].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                    collected_invcov[threadidx].x = cov2d_inv[batch_id][0][0][point_id];
                    collected_invcov[threadidx].y = cov2d_inv[batch_id][0][1][point_id];
                    collected_invcov[threadidx].z = cov2d_inv[batch_id][1][1][point_id];
                    collected_color[threadidx].x = color[batch_id][0][point_id];
                    collected_color[threadidx].y = color[batch_id][1][point_id];
                    collected_color[threadidx].z = color[batch_id][2][point_id];
                    collected_color[threadidx].w = opacity[0][point_id];
                    if (enable_depth_grad)
                    {
                        collected_depth[threadidx] = ndc[batch_id][2][point_id];
                    }
                }
                __syncthreads();
                for (int i = 0; i < collected_num; i++)
                {
                    int point_index = offset - i;
                    bool skip = point_index > pixel_lst_index;
                    float3 grad_color{ 0,0,0 };
                    float3 grad_invcov{ 0,0,0 };
                    float2 grad_ndc_xy{ 0,0 };
                    float grad_ndc_z=0;
                    float grad_opacity{ 0 };

                    if (skip == false)
                    {
                        float2 xy = collected_mean[i];
                        float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                        float4 cur_color = collected_color[i];
                        float3 cur_cov2d_inv = collected_invcov[i];


                        float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                        skip |= power > 0.0f;

                        float G = exp(power);
                        float alpha = min(0.99f, cur_color.w * G);
                        skip |= (alpha < 1.0f / 255.0f);
                        if (skip == false)
                        {
                            transmittance /= (1 - alpha);
                            //color
                            grad_color.x = alpha * transmittance * d_pixel.x;
                            grad_color.y = alpha * transmittance * d_pixel.y;
                            grad_color.z = alpha * transmittance * d_pixel.z;
                            grad_ndc_z = alpha * transmittance * d_depth_pixel;


                            //alpha
                            float d_alpha = 0;
                            d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                            d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                            d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;
                            accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                            accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                            accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;
                            if (enable_trans_grad)
                            {
                                d_alpha -= d_trans_pixel * f_transmittance / (1 - alpha);
                            }
                            if (enable_depth_grad)
                            {
                                d_alpha += (collected_depth[i] - accum_depth) * transmittance * d_depth_pixel;
                                accum_depth = alpha * collected_depth[i] + (1.0f - alpha) * accum_depth;
                            }

                            //opacity
                            grad_opacity = G * d_alpha;

                            //cov2d_inv
                            float d_G = cur_color.w * d_alpha;
                            float d_power = G * d_G;
                            grad_invcov.x = -0.5f * d.x * d.x * d_power;
                            grad_invcov.y = -0.5f * d.x * d.y * d_power;
                            grad_invcov.z = -0.5f * d.y * d.y * d_power;

                            //mean2d
                            float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                            float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                            grad_ndc_xy.x = d_deltax;
                            grad_ndc_xy.y = d_deltay;
                        }

                    }


                    if (warp.all(skip) == false)
                    {
                        for (int offset = 16; offset > 0; offset /= 2)
                        {
                            grad_color.x += __shfl_down_sync(0xffffffff, grad_color.x, offset);
                            grad_color.y += __shfl_down_sync(0xffffffff, grad_color.y, offset);
                            grad_color.z += __shfl_down_sync(0xffffffff, grad_color.z, offset);

                            grad_invcov.x += __shfl_down_sync(0xffffffff, grad_invcov.x, offset);
                            grad_invcov.y += __shfl_down_sync(0xffffffff, grad_invcov.y, offset);
                            grad_invcov.z += __shfl_down_sync(0xffffffff, grad_invcov.z, offset);

                            grad_ndc_xy.x += __shfl_down_sync(0xffffffff, grad_ndc_xy.x, offset);
                            grad_ndc_xy.y += __shfl_down_sync(0xffffffff, grad_ndc_xy.y, offset);
                            if (enable_depth_grad)
                            {
                                grad_ndc_z += __shfl_down_sync(0xffffffff, grad_ndc_z, offset);
                            }

                            grad_opacity += __shfl_down_sync(0xffffffff, grad_opacity, offset);
                        }
                        if (warp.thread_rank() == 0)
                        {
                            int point_id = sorted_points[batch_id][point_index];
                            atomicAdd(&d_color[batch_id][0][point_id], grad_color.x);
                            atomicAdd(&d_color[batch_id][1][point_id], grad_color.y);
                            atomicAdd(&d_color[batch_id][2][point_id], grad_color.z);

                            atomicAdd(&d_cov2d_inv[batch_id][0][0][point_id], grad_invcov.x);
                            atomicAdd(&d_cov2d_inv[batch_id][0][1][point_id], grad_invcov.y);
                            atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], grad_invcov.y);
                            atomicAdd(&d_cov2d_inv[batch_id][1][1][point_id], grad_invcov.z);

                            atomicAdd(&d_ndc[batch_id][0][point_id], grad_ndc_xy.x * 0.5f * img_w);
                            atomicAdd(&d_ndc[batch_id][1][point_id], grad_ndc_xy.y * 0.5f * img_h);
                            if (enable_depth_grad)
                            {
                                atomicAdd(&d_ndc[batch_id][2][point_id], grad_ndc_z);
                            }

                            atomicAdd(&d_opacity[0][point_id], grad_opacity);
                        }
                    }

                }
                __syncthreads();
            }

            
        }
    }
}

__device__ int atomicAggInc(int* ptr)
{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;

    // elect the first active thread to perform atomic add
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ptr, g.size());
    }

    // broadcast previous value within the warp
    // and add each active thread’s rank to it
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

template <int tilesize,bool enable_trans_grad, bool enable_depth_grad>
__global__ void raster_backward_kernel_multibatch_reduction(
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> sorted_points,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> start_index,    //[batch,tile]  p.s. tile_id 0 is invalid!
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> ndc,         //[batch,2,point_num]
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> cov2d_inv,      //[batch,2,2,point_num]
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> color,          //[batch,3,point_num]
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> opacity,          //[1,point_num]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> specific_tiles,          //[batch,tiles_num]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> final_transmitance,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> last_contributor,    //[batch,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_img,    //[batch,3,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_trans_img,    //[batch,1,tile,tilesize,tilesize]
    const torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> d_depth_img,    //[batch,1,tile,tilesize,tilesize]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_ndc,         //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> d_cov2d_inv,      //[batch,2,2,point_num]
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> d_color,          //[batch,3,point_num]
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> d_opacity,          //[1,point_num]
    int tiles_num_x, int img_h, int img_w)
{
    __shared__ float4 collected_color[tilesize * tilesize ];
    __shared__ float3 collected_invcov[tilesize * tilesize ];
    __shared__ float2 collected_mean[tilesize * tilesize + int(enable_depth_grad) * tilesize * tilesize / 2];
    float* collected_depth = (float*)(collected_mean + tilesize * tilesize);

    constexpr int property_num = 9 + enable_depth_grad;

    constexpr int threadsnum_per_property = tilesize * tilesize / property_num;
    __shared__ float gradient_buffer[(tilesize * tilesize+ threadsnum_per_property) * property_num];//"+threadsnum_per_property" to avoid bank conflict
    float* const grad_color_x = gradient_buffer;
    float* const grad_color_y = gradient_buffer + 1 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_color_z = gradient_buffer + 2 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_x = gradient_buffer + 3 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_y = gradient_buffer + 4 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_invcov_z = gradient_buffer + 5 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_ndc_x = gradient_buffer + 6 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_ndc_y = gradient_buffer + 7 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_opacity = gradient_buffer + 8 * (tilesize * tilesize + threadsnum_per_property);
    float* const grad_ndc_z = gradient_buffer + 9 * (tilesize * tilesize + threadsnum_per_property);

    __shared__ int valid_pix_num;

    const int batch_id = blockIdx.y;
    int tile_id = blockIdx.x + 1;// +1, tile_id 0 is invalid
    if (specific_tiles.size(1) != 0)
    {
        tile_id = specific_tiles[batch_id][blockIdx.x];
    }
    auto block = cg::this_thread_block();
    auto cuda_tile = cg::tiled_partition<32>(block);
    int threadidx = threadIdx.y * blockDim.x + threadIdx.x;

    const int x_in_tile = threadIdx.x;
    const int y_in_tile = threadIdx.y;

    int pixel_x = ((tile_id - 1) % tiles_num_x) * tilesize + x_in_tile;
    int pixel_y = ((tile_id - 1) / tiles_num_x) * tilesize + y_in_tile;

    float* global_grad_addr = nullptr;
    switch (threadidx) {
    case 0:
        global_grad_addr = &d_color[batch_id][0][0];
        break;
    case 1:
        global_grad_addr = &d_color[batch_id][1][0];
        break;
    case 2:
        global_grad_addr = &d_color[batch_id][2][0];
        break;
    case 3:
        global_grad_addr = &d_cov2d_inv[batch_id][0][0][0];
        break;
    case 4:
        global_grad_addr = &d_cov2d_inv[batch_id][0][1][0];
        break;
    case 5:
        global_grad_addr = &d_cov2d_inv[batch_id][1][1][0];
        break;
    case 6:
        global_grad_addr = &d_ndc[batch_id][0][0];
        break;
    case 7:
        global_grad_addr = &d_ndc[batch_id][1][0];
        break;
    case 8:
        global_grad_addr = &d_opacity[0][0];
        break;
    case 9:
        global_grad_addr = &d_ndc[batch_id][2][0];
        break;
    default:
        break;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        valid_pix_num = 0;
    }
    __syncthreads();

    if (tile_id!=0 && tile_id < start_index.size(1) - 1)
    {
        int start_index_in_tile = start_index[batch_id][tile_id];
        int end_index_in_tile = start_index[batch_id][tile_id + 1];
        if (start_index_in_tile == -1)
        {
            return;
        }

        float f_transmittance= final_transmitance[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
        float transmittance = f_transmittance;
        int pixel_lst_index = last_contributor[batch_id][blockIdx.x][y_in_tile][x_in_tile];
        float3 d_pixel{ 0,0,0 };
        float d_trans_pixel = 0;
        float d_depth_pixel = 0;
        if (pixel_x < img_w && pixel_y < img_h)
        {
            d_pixel.x = d_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.y = d_img[batch_id][1][blockIdx.x][y_in_tile][x_in_tile];
            d_pixel.z = d_img[batch_id][2][blockIdx.x][y_in_tile][x_in_tile];
            if (enable_trans_grad)
            {
                d_trans_pixel = d_trans_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            }
            if (enable_depth_grad)
            {
                d_depth_pixel = d_depth_img[batch_id][0][blockIdx.x][y_in_tile][x_in_tile];
            }
        }

        float3 accum_rec{ 0,0,0 };
        float accum_depth = 0;
        for (int offset = end_index_in_tile - 1; offset >= start_index_in_tile; offset -= (tilesize * tilesize ))
        {
            int collected_num = min(tilesize * tilesize , offset - start_index_in_tile + 1);
            if (threadIdx.y * blockDim.x + threadIdx.x < collected_num)
            {
                int index = offset - threadidx;
                int point_id = sorted_points[batch_id][index];

                collected_mean[threadidx].x = (ndc[batch_id][0][point_id] + 1.0f) * 0.5f * img_w - 0.5f;
                collected_mean[threadidx].y = (ndc[batch_id][1][point_id] + 1.0f) * 0.5f * img_h - 0.5f;
                if (enable_depth_grad)
                {
                    collected_depth[threadidx] = ndc[batch_id][2][point_id];
                }
                collected_invcov[threadidx].x = cov2d_inv[batch_id][0][0][point_id];
                collected_invcov[threadidx].y = cov2d_inv[batch_id][0][1][point_id];
                collected_invcov[threadidx].z = cov2d_inv[batch_id][1][1][point_id];
                collected_color[threadidx].x = color[batch_id][0][point_id];
                collected_color[threadidx].y = color[batch_id][1][point_id];
                collected_color[threadidx].z = color[batch_id][2][point_id];
                collected_color[threadidx].w = opacity[0][point_id];
            }
            __syncthreads();
            for (int i = 0; i < collected_num; i++)
            {
                int index = offset - i;
                bool bSkip = true;
                float alpha = 0.0f;
                float G = 0.0f;
                float2 xy = collected_mean[i];
                float2 d{ xy.x - pixel_x,xy.y - pixel_y };
                float4 cur_color = collected_color[i];
                float3 cur_cov2d_inv = collected_invcov[i];
                if (index <= pixel_lst_index)
                {
                    float power = -0.5f * (cur_cov2d_inv.x * d.x * d.x + cur_cov2d_inv.z * d.y * d.y) - cur_cov2d_inv.y * d.x * d.y;
                    G = exp(power);
                    alpha = min(0.99f, cur_color.w * G);
                    bSkip = !((power <= 0.0f) && (alpha >= 1.0f / 255.0f));
                }
                __syncthreads();
                if (bSkip == false)
                {
                    int shared_mem_offset = atomicAggInc(&valid_pix_num);
                    transmittance /= (1 - alpha);
                    //color
                    grad_color_x[shared_mem_offset] = alpha * transmittance * d_pixel.x;
                    grad_color_y[shared_mem_offset] = alpha * transmittance * d_pixel.y;
                    grad_color_z[shared_mem_offset] = alpha * transmittance * d_pixel.z;
                    if (enable_depth_grad)
                    {
                        grad_ndc_z[shared_mem_offset] = alpha * transmittance * d_depth_pixel;
                    }

                    //alpha
                    float d_alpha = 0;
                    d_alpha += (cur_color.x - accum_rec.x) * transmittance * d_pixel.x;
                    d_alpha += (cur_color.y - accum_rec.y) * transmittance * d_pixel.y;
                    d_alpha += (cur_color.z - accum_rec.z) * transmittance * d_pixel.z;
                    accum_rec.x = alpha * cur_color.x + (1.0f - alpha) * accum_rec.x;
                    accum_rec.y = alpha * cur_color.y + (1.0f - alpha) * accum_rec.y;
                    accum_rec.z = alpha * cur_color.z + (1.0f - alpha) * accum_rec.z;
                    if (enable_trans_grad)
                    {
                        d_alpha -= d_trans_pixel * f_transmittance / (1 - alpha);
                    }
                    if (enable_depth_grad)
                    {
                        d_alpha += (collected_depth[i] - accum_depth) * transmittance * d_depth_pixel;
                        accum_depth = alpha * collected_depth[i] + (1.0f - alpha) * accum_depth;
                    }

                    //opacity
                    grad_opacity[shared_mem_offset] = G * d_alpha;

                    //cov2d_inv
                    float d_G = cur_color.w * d_alpha;
                    float d_power = G * d_G;
                    grad_invcov_x[shared_mem_offset] = -0.5f * d.x * d.x * d_power;
                    grad_invcov_y[shared_mem_offset] = -0.5f * d.x * d.y * d_power;
                    grad_invcov_z[shared_mem_offset] = -0.5f * d.y * d.y * d_power;

                    //mean2d
                    float d_deltax = (-cur_cov2d_inv.x * d.x - cur_cov2d_inv.y * d.y) * d_power;
                    float d_deltay = (-cur_cov2d_inv.z * d.y - cur_cov2d_inv.y * d.x) * d_power;
                    grad_ndc_x[shared_mem_offset] = d_deltax * 0.5f * img_w;
                    grad_ndc_y[shared_mem_offset] = d_deltay * 0.5f * img_h;
                }

                __syncthreads();
                if (valid_pix_num > 0)
                {
                    int property_id = threadidx / threadsnum_per_property;
                    int ele_offset = threadidx % threadsnum_per_property;
                    if (property_id < property_num)
                    {
                        float sum = 0;
                        for (int i = ele_offset; i < valid_pix_num; i += threadsnum_per_property)
                        {
                            sum += gradient_buffer[property_id * (tilesize * tilesize + threadsnum_per_property) + i];
                        }
                        gradient_buffer[property_id * (tilesize * tilesize + threadsnum_per_property) + ele_offset] = sum;
                    }
                    __syncthreads();
                    if (threadidx < property_num)
                    {
                        float sum = 0;
                        for (int i = 0; i < threadsnum_per_property; i++)
                        {
                            sum += gradient_buffer[threadidx * (tilesize * tilesize + threadsnum_per_property) + i];
                        }
                        int point_id = sorted_points[batch_id][index];
                        atomicAdd(global_grad_addr + point_id, sum);
                        if (threadidx == 4)
                        {
                            atomicAdd(&d_cov2d_inv[batch_id][1][0][point_id], sum);
                        }
                    }
                    valid_pix_num = 0;


                }
            }
            

        }
        
    }
}


#define BACKWARD_CASE_MULTIBATCH_KERNEL(TILESIZE,TRANS,DEPTH) RASTER_SWITCH_CASE(TILESIZE,TRANS,DEPTH):\
raster_backward_kernel_multibatch_reduction<TILESIZE,TRANS,DEPTH>

#define BACKWARD_CASE_WARP_KERNEL(TILESIZE,TRANS,DEPTH) RASTER_SWITCH_CASE(TILESIZE,TRANS,DEPTH):\
raster_backward_kernel_warp_reduction<TILESIZE,TRANS,DEPTH>

#define BACKWARD_KERNEL_ARGS sorted_points.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
start_index.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),\
color.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),\
opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),\
specific_tiles.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),\
final_transmitance.packed_accessor32<float, 5, torch::RestrictPtrTraits >(),\
last_contributor.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),\
d_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_trans_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_depth_img.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),\
d_ndc.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),\
d_cov2d_inv.packed_accessor32<float, 4, torch::RestrictPtrTraits >(),\
d_color.packed_accessor32<float, 3, torch::RestrictPtrTraits >(),\
d_opacity.packed_accessor32<float, 2, torch::RestrictPtrTraits >(),\
tilesnum_x, img_h, img_w

std::vector<at::Tensor> rasterize_backward(
    at::Tensor sorted_points,
    at::Tensor start_index,
    at::Tensor ndc,// 
    at::Tensor cov2d_inv,
    at::Tensor color,
    at::Tensor opacity,
    std::optional<at::Tensor> specific_tiles_arg,
    at::Tensor final_transmitance,
    at::Tensor last_contributor,
    at::Tensor d_img,
    std::optional<at::Tensor> d_trans_img_arg,
    std::optional<at::Tensor> d_depth_img_arg,
    int64_t tilesize,
    int64_t img_h,
    int64_t img_w
)
{
    at::DeviceGuard guard(ndc.device());

    int64_t viewsnum = start_index.sizes()[0];
    int tilesnum_x = std::ceil(img_w / float(tilesize));
    int tilesnum_y = std::ceil(img_h / float(tilesize));
    int64_t tilesnum = tilesnum_x * tilesnum_y;
    at::Tensor specific_tiles;
    if (specific_tiles_arg.has_value())
    {
        specific_tiles = *specific_tiles_arg;
        tilesnum = specific_tiles.sizes()[1];
    }
    else
    {
        specific_tiles = torch::empty({ 0,0 }, ndc.options().dtype(torch::kInt32));
    }
    at::Tensor d_trans_img;
    if (d_trans_img_arg.has_value())
    {
        d_trans_img = *d_trans_img_arg;
    }
    else
    {
        d_trans_img = torch::empty({ 0,0,0,0,0 }, d_img.options());
    }
    at::Tensor d_depth_img;
    if (d_depth_img_arg.has_value())
    {
        d_depth_img = *d_depth_img_arg;
    }
    else
    {
        d_depth_img = torch::empty({ 0,0,0,0,0 }, d_img.options());
    }

    at::Tensor d_ndc = torch::zeros_like(ndc, ndc.options());
    at::Tensor d_cov2d_inv = torch::zeros_like(cov2d_inv, ndc.options());
    at::Tensor d_color = torch::zeros_like(color, ndc.options());
    at::Tensor d_opacity = torch::zeros_like(opacity, ndc.options());

    dim3 Block3d(tilesnum, viewsnum, 1);
    dim3 Thread3d(tilesize, tilesize, 1);

    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<8, false, false>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<8, true, false>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<8, false, true>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<8, true, true>, cudaFuncCachePreferShared);
    /*cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<16, false, false>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<16, true, false>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<16, false, true>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(raster_backward_kernel_multibatch_reduction<16, true, true>, cudaFuncCachePreferShared);*/

    int template_code = (tilesize << 8) + (d_trans_img_arg.has_value() << 1) + (d_depth_img_arg.has_value());
    switch (template_code)
    {
        BACKWARD_CASE_MULTIBATCH_KERNEL(8,true,true) <<<Block3d, Thread3d >>> (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_MULTIBATCH_KERNEL(8, true, false) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_MULTIBATCH_KERNEL(8, false, true) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_MULTIBATCH_KERNEL(8, false, false) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_WARP_KERNEL(16, true, true) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_WARP_KERNEL(16, true, false) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_WARP_KERNEL(16, false, true) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
        BACKWARD_CASE_WARP_KERNEL(16, false, false) << <Block3d, Thread3d >> > (BACKWARD_KERNEL_ARGS);
        break;
    default:
        assert(false);
        ;
    }
    CUDA_CHECK_ERRORS;
    
    return { d_ndc ,d_cov2d_inv ,d_color,d_opacity };
}
