import torch
from torch.utils.data import DataLoader
import fused_ssim
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import torch.cuda.nvtx as nvtx

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
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
    has_masks=os.path.isdir(mask_root)
    if has_masks:
        pp.enable_transmitance=True
        print("[LiteGS] Enabling mask-aware densification for object-centric training.")
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload,mask_root=mask_root)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload,mask_root=mask_root)
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
    total_epoch=int(op.iterations/len(trainingset))
    if dp.densify_until<0:
        dp.densify_until=int(int(total_epoch/2)/dp.opacity_reset_interval)*dp.opacity_reset_interval
    density_controller=densify.DensityControllerOfficial(norm_radius,dp,pp.cluster_size>0)
    statistics_chunk_num = xyz.shape[-2] if pp.cluster_size else 1
    StatisticsHelperInst.reset(statistics_chunk_num,xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    for epoch in range(start_epoch,total_epoch):

        with torch.no_grad():
            if epoch%pp.spatial_refine_interval==0:#spatial refine
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
            if pp.cluster_size>0 and (epoch%pp.spatial_refine_interval==0 or density_controller.is_densify_actived(epoch-1)):
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,frustumplane,gt_image,gt_mask in train_loader:
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                gt_mask=gt_mask.cuda().float()
                inverse_mask = 1.0 - gt_mask

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)

                if has_masks:
                    rgb_loss_map = fused_ssim.FusedL1SSIMLossMap.apply(0.2, 0.01 ** 2, 0.03 ** 2, img, gt_image, "same", True)
                    rgb_loss = __safe_masked_mean(rgb_loss_map, gt_mask.expand_as(rgb_loss_map))
                    # bg_rgb_loss = __safe_masked_mean(img.abs(), inverse_mask.expand_as(img))
                    loss = rgb_loss # + 0.5 * bg_rgb_loss
                    if transmitance is not None:
                        alpha = 1.0 - transmitance
                        fg_alpha_loss = __safe_masked_mean((1.0 - alpha).abs(), gt_mask)
                        bg_alpha_loss = __safe_masked_mean(alpha.abs(), inverse_mask)
                        loss = loss + 2.0 * fg_alpha_loss + bg_alpha_loss * 2
                else:
                    l1_loss=__l1_loss(img,gt_image)
                    ssim_loss:torch.Tensor=fused_ssim.fused_ssim(img,gt_image)
                    loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*(1-ssim_loss)
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
                    for eval_index,(view_matrix,proj_matrix,frustumplane,gt_image,gt_mask) in enumerate(loader):
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0
                        gt_mask=gt_mask.cuda().float()
                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        img,transmitance,depth,normal=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
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
    
    return
