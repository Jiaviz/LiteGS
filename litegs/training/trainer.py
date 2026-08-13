import torch
from torch.utils.data import DataLoader
import fused_ssim
from tqdm import tqdm
import numpy as np
import os
import json
import time
import math
from PIL import Image
import torch.cuda.nvtx as nvtx

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..render import surfel
from .optimizer import SparseGaussianAdam
from ..utils import wrapper
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def __safe_masked_mean(values:torch.Tensor, weights:torch.Tensor)->torch.Tensor:
    weighted_sum = (values * weights).sum()
    normalizer = weights.sum().clamp_min(1.0)
    return weighted_sum / normalizer

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None,
          save_eval_images:bool=False,eval_image_count:int=8):
    
    torch.manual_seed(pp.random_seed)
    np.random.seed(pp.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pp.random_seed)

    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    mask_root=os.path.join(lp.source_path,"masks")
    depth_root=os.path.join(lp.source_path,"depths")
    normal_root=os.path.join(lp.source_path,"normals")
    has_masks=os.path.isdir(mask_root)
    has_depth_priors=os.path.isdir(depth_root)
    has_normal_priors=os.path.isdir(normal_root)
    if has_masks:
        pp.enable_transmitance=True
        print("[LiteGS] Enabling mask-aware densification for object-centric training.")
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload,mask_root=mask_root,depth_root=depth_root if has_depth_priors else None,normal_root=normal_root if has_normal_priors else None)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload,mask_root=mask_root,depth_root=depth_root if os.path.isdir(depth_root) else None,normal_root=normal_root if os.path.isdir(normal_root) else None)
        test_loader = DataLoader(testset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    init_points_num=init_xyz.shape[0]
    if has_masks and pp.cluster_size:
        print(f"[LiteGS] Disabling clustering for masked training data at {mask_root}.")
        pp.cluster_size = 0
    if pp.cluster_size and init_points_num < pp.cluster_size:
        print(f"[LiteGS] Disabling clustering because only {init_points_num} points are available (< cluster_size={pp.cluster_size}).")
        pp.cluster_size = 0
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
        if pp.cluster_size:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #init
    total_epoch=math.ceil(op.iterations/len(trainingset))
    if dp.densify_until<0:
        dp.densify_until=int(int(total_epoch/2)/dp.opacity_reset_interval)*dp.opacity_reset_interval
    density_controller=densify.DensityControllerOfficial(norm_radius,dp,pp.cluster_size>0,pp.gaussian_dimension)
    statistics_chunk_num = xyz.shape[-2] if pp.cluster_size else 1
    StatisticsHelperInst.reset(statistics_chunk_num,xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    training_started_at=time.perf_counter()
    completed_steps=0
    accumulated_losses={"rgb":0.0,"alpha":0.0,"depth":0.0,"normal":0.0,"distortion":0.0,"total":0.0}

    for epoch in range(start_epoch,total_epoch):

        with torch.no_grad():
            if epoch%pp.spatial_refine_interval==0:#spatial refine
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
            if pp.cluster_size>0 and (epoch%pp.spatial_refine_interval==0 or density_controller.is_densify_actived(epoch-1)):
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,frustumplane,gt_image,gt_mask,gt_depth,gt_normal in train_loader:
                if completed_steps >= op.iterations:
                    break
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                gt_mask=gt_mask.cuda().float()
                gt_depth=gt_depth.cuda().float()
                gt_normal=gt_normal.cuda().float()
                inverse_mask = 1.0 - gt_mask

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                if pp.gaussian_dimension == 2:
                    img,transmitance,depth,normal,surface_normal,distortion,_,_=surfel.render(
                        view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                        actived_sh_degree,gt_image.shape[2:],pp,compute_surface_normal=not has_normal_priors)
                else:
                    img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                actived_sh_degree,gt_image.shape[2:],pp)
                    surface_normal=distortion=None

                if has_masks:
                    rgb_loss_map = fused_ssim.FusedL1SSIMLossMap.apply(0.2, 0.01 ** 2, 0.03 ** 2, img, gt_image, "same", True)
                    rgb_loss = __safe_masked_mean(rgb_loss_map, gt_mask.expand_as(rgb_loss_map))
                    # bg_rgb_loss = __safe_masked_mean(img.abs(), inverse_mask.expand_as(img))
                    loss = rgb_loss # + 0.5 * bg_rgb_loss
                    alpha_loss=torch.zeros((),device=loss.device)
                    if transmitance is not None:
                        alpha = 1.0 - transmitance
                        fg_alpha_loss = __safe_masked_mean((1.0 - alpha).abs(), gt_mask)
                        bg_alpha_loss = __safe_masked_mean(alpha.abs(), inverse_mask)
                        alpha_loss=2.0 * fg_alpha_loss + bg_alpha_loss * 2
                        loss = loss + alpha_loss
                else:
                    l1_loss=__l1_loss(img,gt_image)
                    ssim_loss:torch.Tensor=fused_ssim.fused_ssim(img,gt_image)
                    loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*(1-ssim_loss)
                    rgb_loss=loss
                    alpha_loss=torch.zeros((),device=loss.device)
                depth_loss=normal_loss=distortion_loss=torch.zeros((),device=loss.device)
                if pp.gaussian_dimension == 2 and epoch * len(trainingset) >= op.geometry_start_iteration:
                    valid_depth = (gt_depth > 0).float() * gt_mask
                    if valid_depth.sum() > 0:
                        # Scale-invariant log-depth is stable across DTU normalization.
                        depth_error = (depth.clamp_min(1e-6).log() - gt_depth.clamp_min(1e-6).log()).abs()
                        depth_loss=op.lambda_depth * __safe_masked_mean(depth_error, valid_depth)
                        loss = loss + depth_loss
                    valid_normal = (gt_normal.square().sum(dim=1, keepdim=True) > 0.25).float() * gt_mask
                    target_normal = torch.nn.functional.normalize(gt_normal, dim=1, eps=1e-6)
                    if valid_normal.sum() > 0:
                        normal_error = 1.0 - (normal * target_normal).sum(dim=1, keepdim=True).abs()
                        normal_loss=op.lambda_normal * __safe_masked_mean(normal_error, valid_normal)
                        loss = loss + normal_loss
                    elif surface_normal is not None:
                        normal_error = 1.0 - (normal * surface_normal).sum(dim=1, keepdim=True).abs()
                        normal_loss=op.lambda_normal * __safe_masked_mean(normal_error, gt_mask)
                        loss = loss + normal_loss
                    distortion_loss=op.lambda_distortion * __safe_masked_mean(distortion, gt_mask)
                    loss = loss + distortion_loss
                accumulated_losses["rgb"] += float(rgb_loss.detach())
                accumulated_losses["alpha"] += float(alpha_loss.detach())
                accumulated_losses["depth"] += float(depth_loss.detach())
                accumulated_losses["normal"] += float(normal_loss.detach())
                accumulated_losses["distortion"] += float(distortion_loss.detach())
                accumulated_losses["total"] += float(loss.detach())
                completed_steps += 1
                loss.backward()
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.cluster_size and pp.sparse_grad:
                    opt.step(visible_chunkid)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
                schedular.step()

        if epoch in test_epochs:
            with torch.no_grad():
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    for eval_index,(view_matrix,proj_matrix,frustumplane,gt_image,gt_mask,gt_depth,gt_normal) in enumerate(loader):
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0
                        gt_mask=gt_mask.cuda().float()
                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        if pp.gaussian_dimension == 2:
                            img,transmitance,depth,normal,_,_,_,_=surfel.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,actived_sh_degree,gt_image.shape[2:],pp,compute_surface_normal=False)
                        else:
                            img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,actived_sh_degree,gt_image.shape[2:],pp)
                        if has_masks:
                            squared_error = (img - gt_image).square() * gt_mask
                            mse = squared_error.sum() / (gt_mask.sum() * img.shape[1]).clamp_min(1.0)
                        else:
                            mse = (img - gt_image).square().mean()
                        psnr_list.append((-10.0 * torch.log10(mse.clamp_min(1e-10))).unsqueeze(0))
                        if save_eval_images and eval_index < eval_image_count:
                            render_rgb = (
                                img[0].detach().clamp(0.0, 1.0).mul(255.0)
                                .byte().permute(1, 2, 0).cpu().numpy()
                            )
                            gt_rgb = gt_image[0].detach().clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
                            comparison = np.concatenate((gt_rgb, render_rgb), axis=1)
                            mask_rgb = (
                                gt_mask[0].detach().clamp(0.0, 1.0)
                                .permute(1, 2, 0).cpu().numpy()
                            )
                            masked_gt_rgb = (gt_rgb.astype(np.float32) * mask_rgb).astype(np.uint8)
                            masked_render_rgb = (render_rgb.astype(np.float32) * mask_rgb).astype(np.uint8)
                            masked_comparison = np.concatenate((masked_gt_rgb, masked_render_rgb), axis=1)
                            eval_dir = os.path.join(lp.model_path, "eval", "epoch_{:04d}".format(epoch), name.lower())
                            os.makedirs(eval_dir, exist_ok=True)
                            Image.fromarray(comparison).save(os.path.join(eval_dir, "{:03d}_gt_render.png".format(eval_index)))
                            Image.fromarray(masked_comparison).save(
                                os.path.join(eval_dir, "{:03d}_masked_gt_render.png".format(eval_index))
                            )
                    metric_scope = "foreground " if has_masks else ""
                    tqdm.write(
                        "\n[EPOCH {}] {} Evaluating: {}PSNR {}".format(
                            epoch,
                            name,
                            metric_scope,
                            torch.concat(psnr_list, dim=0).mean(),
                        )
                    )

        if completed_steps >= op.iterations and epoch < total_epoch-1:
            # Keep final checkpoint/export behavior tied to the actual last
            # epoch when the requested iteration budget is not divisible by
            # the number of views.
            total_epoch=epoch+1

        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch)
        progress_bar.update()  

        if epoch in save_ply or epoch==total_epoch-1:
            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            if epoch==total_epoch-1:
                ply_path=os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply")
            else:
                ply_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch),"point_cloud.ply")
            io_manager.save_ply(ply_path,*param_nyp)
            pass

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds=time.perf_counter()-training_started_at
    stats={
        "gaussian_dimension":int(pp.gaussian_dimension),
        "random_seed":int(pp.random_seed),
        "requested_iterations":int(op.iterations),
        "completed_iterations":int(completed_steps),
        "epochs":int(total_epoch-start_epoch),
        "training_seconds":float(training_seconds),
        "iterations_per_second":float(completed_steps/max(training_seconds,1e-9)),
        "initial_primitives":int(init_points_num),
        "final_primitives":int(xyz.shape[-1]),
        "has_masks":bool(has_masks),
        "has_depth_priors":bool(has_depth_priors),
        "has_normal_priors":bool(has_normal_priors),
        "mean_weighted_losses":{key:value/max(completed_steps,1) for key,value in accumulated_losses.items()},
        "peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    os.makedirs(lp.model_path,exist_ok=True)
    with open(os.path.join(lp.model_path,"training_stats.json"),"w",encoding="utf-8") as file:
        json.dump(stats,file,indent=2)
        file.write("\n")
    tqdm.write("[LiteGS] Training statistics: {:.2f}s, {:.1f} iter/s, {} primitives.".format(training_seconds,stats["iterations_per_second"],stats["final_primitives"]))
    return stats
