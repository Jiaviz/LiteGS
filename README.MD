# LiteGS

**Fastest(about 50 seconds), Modular, and Available in Pure Python or CUDA**

This repository provides a refactored codebase aimed at improving the flexibility and performance of Gaussian splatting. 

## Training 3DGS in about 50 seconds

`python ./scripts/full_eval_aggressive.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

| Scene | Primitives | Time_3090(s) | Time_4090(s) | SSIM | PSNR | LPIPS |
|-------|------------|---------------|---------------|------|------|-------|
| bicycle | 1,000,000 | 50 | 33 | 0.7467 | 25.10 | 0.2531 |
| bonsai | 1,000,000 | 71 | 45 | 0.9430 | 32.02 | 0.1986 |
| counter | 1,000,000 | 84 | 51 | 0.9103 | 28.98 | 0.1952 |
| drjohnson | 1,000,000 | 47 | 28 | 0.9054 | 29.55 | 0.2596 |
| flowers | 1,000,000 | 60 | 40 | 0.6011 | 21.75 | 0.3500 |
| garden | 1,000,000 | 53 | 33 | 0.8471 | 27.23 | 0.1485 |
| kitchen | 1,000,000 | 82 | 50 | 0.9249 | 31.33 | 0.1321 |
| playroom | 1,000,000 | 48 | 29 | 0.9123 | 30.82 | 0.2434 |
| room | 1,000,000 | 64 | 39 | 0.9220 | 31.63 | 0.2130 |
| stump | 1,000,000 | 53 | 34 | 0.7847 | 27.10 | 0.2278 |
| train | 1,000,000 | 70 | 47 | 0.7956 | 21.50 | 0.2316 |
| treehill | 1,000,000 | 57 | 37 | 0.6382 | 23.17 | 0.3479 |
| truck | 1,000,000 | 63 | 43 | 0.8831 | 25.78 | 0.1368 |

| Dataset | Primitives | Time_3090(s) | Time_4090(s) | SSIM | PSNR | LPIPS |
|-------|------------|---------------|---------------|------|------|-------|
| mipnerf360 | 1,000,000 | 64 | 40 | 0.8131 | 27.59 | 0.2296 |
| tat | 1,000,000 | 67 | 45 | 0.8394 | 23.64 | 0.1842 |
| db | 1,000,000 | 47 | 29 | 0.9088 | 30.19 | 0.2515 |

## Background
Gaussian splatting is a powerful technique used in various computer graphics and vision applications. It involves representing 3D data as Gaussian distributions in space, allowing for efficient and accurate representation of spatial data. However, the original implementation (https://github.com/graphdeco-inria/gaussian-splatting) of Gaussian splatting in PyTorch faced several limitations: 

1. The forward and backward computations were encapsulated in two distinct PyTorch extension functions. Although this design significantly accelerated training, it restricted access to intermediate variables unless the underlying C code was modified.
2. Modifying any step of the algorithm required manually deriving gradient formulas and implementing them in the backward pass, adding considerable complexity.

## Features

1. **Modular Design**: The refactored codebase breaks forward and backward into multiple PyTorch extension functions, significantly improving modularity and enabling easier access to intermediate variables. Additionally, in some cases, leveraging PyTorch Autograd eliminates the need to manually derive gradient formulas.

2. **Flexible**: LiteGS provides two modular APIs—one implemented in CUDA and the other in Python. The Python-based API facilitates straightforward modifications to calculation logic without requiring expertise in C code, enabling rapid prototyping. Additionally, tensor dimensions are permuted to maintain competitive training speeds for the Python API. For performance-critical tasks, the CUDA-based API is fully customizable.

3. **Better Performance and Fewer Resources**: LiteGS achieves an 4.7x speed improvement over the original 3DGS implementation while reducing GPU memory usage by around 30%. These optimizations enhance training efficiency without compromising flexibility or readability.

4. **Algorithm Preservation**: LiteGS retains the core 3DGS algorithm, making only minor adjustments to the training logic due to culstering.


## Getting Started

1. Install simple-knn

    ```bash
    pip install litegs/submodules/simple-knn
    ```

2. Install fused-ssim

    ```bash
    pip install litegs/submodules/fused_ssim
    ```
    

3. Install litegs_fused
    
    ```bash
    pip install litegs/submodules/gaussian_raster
    ```

    If you need the cmake project(e.g. CUDA Debug in Visual Studio)：

    ```bash
    cd litegs/submodules/gaussian_raster
    mkdir ./build
    cd ./build
    #for Windows PowerShell: $env:CMAKE_PREFIX_PATH = (python -c "import torch; print(torch.utils.cmake_prefix_path)")
    export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
    cmake ../
    cmake --build . --config Release
    ```

4. Install requirments

    ```bash
    pip install -r requirement.txt
    ```
    
### Train
Begin training with the following command:

`./example_train.py --sh_degree 3 -s DATA_SOURCE -i IMAGE_FOLDER -m OUTPUT_PATH`

## CUDA 2D Gaussian Splatting

The `feature/2dgs-cuda` implementation represents every primitive as a
perspective-correct elliptical surfel. Forward rasterization and its backward
pass use the CUDA kernels in `litegs/submodules/diff_surfel_rasterization`, not
a Python rasterizer. Training supports masked RGB, metric/normalized camera-z
depth (`depths/<image-stem>.npy`) and view-space normal
(`normals/<image-stem>.npy`) supervision, plus the original 2DGS distortion
regularizer. If normal priors are absent, rendered normals are regularized by
normals derived from rendered depth.

Install the additional renderer in the same environment as LiteGS:

```bash
CUDA_HOME=/usr/local/cuda pip install -v \
  ./litegs/submodules/diff_surfel_rasterization
```

The fast preset disables clustering, uses the CUDA surfel path, delays geometry
losses briefly for appearance stabilization, and bounds densification:

```bash
python example_train_2dgs.py \
  -s /path/to/colmap_dataset \
  -m outputs/object_2dgs \
  --resolution 4
```

Every run writes `training_stats.json` with exact completed iterations,
iterations/s, peak allocated CUDA memory, primitive counts, and mean weighted
RGB/alpha/depth/normal/distortion losses.

### Reproduce the DTU scan-24 comparison

The scripts operate on the IDR-format sample in `/home/jarvi/data/DTU/scan24`
and use the official DTU laser scan only for constructing depth/normal training
priors and crop-aware geometry evaluation:

```bash
python scripts/prepare_dtu.py \
  --input /home/jarvi/data/DTU/scan24 \
  --output /home/jarvi/data/DTU_LiteGS/scan24

python scripts/download_dtu_scan_gt.py \
  --output /home/jarvi/data/DTU_Official_scan24 --scan 24

python scripts/generate_dtu_priors.py \
  --scan-input /home/jarvi/data/DTU/scan24 \
  --dataset-output /home/jarvi/data/DTU_LiteGS/scan24 \
  --gt-ply /home/jarvi/data/DTU_Official_scan24/Points/stl/stl024_total.ply \
  --resolution 4

python scripts/benchmark_dtu_2dgs.py \
  --dataset /home/jarvi/data/DTU_LiteGS/scan24 \
  --scan-input /home/jarvi/data/DTU/scan24 \
  --official-root /home/jarvi/data/DTU_Official_scan24 \
  --output-root /tmp/litegs_dtu_scan24
```

`comparison.json` records matched vanilla LiteGS and 2DGS timing and the DTU
crop-aware accuracy/completeness/Chamfer-L1 of Gaussian primitive centers. This
is explicitly a center-geometry metric rather than a mesh-reconstruction score.

On DTU scan 24 at 400x300, 30k fixed primitives, seed 0, and exactly 10k
optimization steps on an RTX 4070 Ti SUPER, the matched reference run measured:

| Method | Train time | Iter/s | Peak allocated VRAM | Accuracy | Completeness | Chamfer-L1 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla LiteGS | 67.00 s | 149.2 | 113.9 MB | 3.899 mm | 6.696 mm | 5.297 mm |
| CUDA 2DGS + RGB/depth/normal | 78.75 s | 127.0 | 102.5 MB | 3.796 mm | 5.927 mm | 4.862 mm |

Thus 2DGS improved center-geometry Chamfer-L1 by 8.22%, used about 10% less
peak allocated VRAM, and retained 85% of vanilla LiteGS throughput despite
rendering and supervising the additional geometric buffers. The benchmark used
fixed primitive counts to isolate rasterizer/loss cost; the fast entry point
also supports bounded tangent-plane densification for production training.

## Faster

The training results of LiteGS using the Mip-NeRF 360 dataset on an RTX 3090 are presented below. The training and evaluation command used is:

LiteGS:
`python ./full_eval.py --mipnerf360 SOURCE_PATH1 --tanksandtemples SOURCE_PATH2 --deepblending SOURCE_PATH3`

![image](doc_img/LiteGS_ParamScale.png)



## Modular

Unlike the original 3DGS, which encapsulates nearly the entire rendering process into a single PyTorch extension function, LiteGS divides the process into multiple modular functions. This design allows users to access intermediate variables and integrate custom computation logic using Python scripts, eliminating the need to modify C code. The rendering process in LiteGS is broken down into the following steps:

1. Cluster Culling
    
    LiteGS divides the Gaussian points into several chunks, with each chunk containing 1,024 points. The first step in the rendering pipeline is frustum culling, where points outside the camera's view are filtered out.

2. Cluster Compact
    
    Similar to mesh rendering, LiteGS compacts visible primitives after frustum culling. Each property of the visible points is reorganized into sequential memory to improve processing efficiency.

3. 3DGS Projection
    
    Gaussian points are projected into screen space in this step, with no modifications made compared to the original 3DGS implementation.

4. Create Visibility Table
    
    A visibility table is created in this step, mapping tiles to their visible primitives, enabling efficient parallel processing in subsequent stages.

5. Rasterization
    
    In the final step, each tile rasterizes its visible primitives in parallel, ensuring high computational efficiency.

LiteGS makes slight adjustments to density control to accommodate its clustering-based approach. 

## Flexible

The gaussian_splatting/wrapper.py file contains two sets of APIs, offering flexibility in choosing between Python-based and CUDA-based implementations. The Python-based API is invoked using call_script(), while the CUDA-based API is available via call_fused(). While the CUDA-based API delivers significant performance improvements, it lacks the flexibility. The choice between these implementations depends on the specific use case:

* python-based api: Provides greater flexibility, making it ideal for rapid prototyping and development where training speed is less critical.

* cuda-based api: Offers the highest performance and is recommended for production environments where training speed is a priority.

Additionally, an interface validate() and the accompanying check_wrapper.py script are provided to verify that both APIs produce consistent gradients.

Here is an example that demonstrates the flexibility of LiteGS. In this instance, our goal is to create a more precise bounding box for a 2D Gaussian when generating visibility tables. In the original 3DGS implementation, the bounding box is determined as three times the length of the major axis of the Gaussian. However, incorporating opacity can allow for a smaller bounding box.

To implement this change in the original 3DGS, the following steps are required:
* Modify the C++ function declarations and definitions
* Update the CUDA global function
* Recompile

In LiteGS, the same change can be achieved by simply editing a Python script.

original:
```python
axis_length=(3.0*eigen_val.abs()).sqrt().ceil()
```
modified:
```python
coefficient=2*((255*opacity).log())
axis_length=(coefficient*eigen_val.abs()).sqrt().ceil()
```

## Performance Optimization Detail

Comming soon.
