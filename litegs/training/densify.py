import torch
import math

from ..arguments import DensifyParams
from ..utils.statistic_helper import StatisticsHelperInst
from ..utils import qvec2rotmat
from ..scene import cluster
from ..utils import wrapper

class DensityControllerBase:
    def __init__(self,densify_params:DensifyParams,bCluster:bool) -> None:
        self.densify_params=densify_params
        self.bCluster=bCluster
        return
    
    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int):
        return
    
    @torch.no_grad()
    def _get_params_from_optimizer(self,optimizer:torch.optim.Optimizer)->list[torch.Tensor]:
        param_dict:dict[str,torch.Tensor]={}
        for param_group in optimizer.param_groups:
            name=param_group['name']
            tensor=param_group['params'][0]
            param_dict[name]=tensor
        xyz=param_dict["xyz"]
        rot=param_dict["rot"]
        scale=param_dict["scale"]
        sh_0=param_dict["sh_0"]
        sh_rest=param_dict["sh_rest"]
        opacity=param_dict["opacity"]
        return xyz,scale,rot,sh_0,sh_rest,opacity

    @torch.no_grad()
    def _cat_tensors_to_optimizer(self, tensors_dict:dict,optimizer:torch.optim.Optimizer):
        point_dimension = -2 if self.bCluster else -1
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and stored_state["exp_avg"].shape==group["params"][0].shape
                stored_state["exp_avg"].data=torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=point_dimension).contiguous()
                stored_state["exp_avg_sq"].data=torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=point_dimension).contiguous()
            new_param=torch.cat((group["params"][0], extension_tensor), dim=point_dimension).contiguous()
            optimizer.state.pop(group['params'][0])#pop param
            group["params"][0]=torch.nn.Parameter(new_param)
            if stored_state is not None:
                optimizer.state[group["params"][0]]=stored_state#assign to new param
                assert stored_state["exp_avg"].shape == stored_state["exp_avg_sq"].shape and stored_state["exp_avg"].shape==group["params"][0].shape
        return
    
    @torch.no_grad()
    def _prune_optimizer(self,valid_mask:torch.Tensor,optimizer:torch.optim.Optimizer):
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if self.bCluster:
                    chunk_size=stored_state["exp_avg"].shape[-1]
                    uncluster_avg,uncluster_avg_sq=cluster.uncluster(stored_state["exp_avg"],stored_state["exp_avg_sq"])
                    uncluster_avg=uncluster_avg[...,valid_mask]
                    uncluster_avg_sq=uncluster_avg_sq[...,valid_mask]
                    new_avg,new_avg_sq=cluster.cluster_points(chunk_size,uncluster_avg,uncluster_avg_sq)
                else:
                    new_avg=stored_state["exp_avg"][...,valid_mask]
                    new_avg_sq=stored_state["exp_avg_sq"][...,valid_mask]
                stored_state["exp_avg"].data=new_avg
                stored_state["exp_avg_sq"].data=new_avg_sq
            
            if self.bCluster:
                chunk_size=group["params"][0].shape[-1]
                uncluster_param,=cluster.uncluster(group["params"][0])
                uncluster_param=uncluster_param[...,valid_mask]
                new_param,=cluster.cluster_points(chunk_size,uncluster_param)
            else:
                new_param=group["params"][0][...,valid_mask]
            optimizer.state.pop(group['params'][0])#pop param
            group["params"][0]=torch.nn.Parameter(new_param)
            optimizer.state[group["params"][0]]=stored_state#assign to new param
        return
    
class DensityControllerOfficial(DensityControllerBase):
    @torch.no_grad()
    def __init__(self,screen_extent:int,densify_params:DensifyParams,bCluster:bool)->None:
        self.grad_threshold=densify_params.densify_grad_threshold
        self.min_opacity=densify_params.opacity_threshold
        self.percent_dense=densify_params.percent_dense
        self.prune_large_point_from=densify_params.prune_large_point_from
        self.screen_extent=screen_extent
        self.max_screen_size=densify_params.screen_size_threshold
        self.max_gaussians=int(densify_params.max_gaussians)
        self._reported_gradient_stats = False
        super(DensityControllerOfficial,self).__init__(densify_params,bCluster)
        return
    
    @torch.no_grad()
    def get_prune_mask(self,actived_opacity:torch.Tensor,actived_scale:torch.Tensor)->torch.Tensor:
        transparent = (actived_opacity < self.min_opacity).squeeze()
        invisible = StatisticsHelperInst.get_global_culling()
        invisible.shape[0]
        prune_mask=transparent
        prune_mask[:invisible.shape[0]]|=invisible
        big_points_vs = StatisticsHelperInst.get_max('radii') > self.max_screen_size
        prune_mask[:invisible.shape[0]]|=big_points_vs
        return prune_mask

    @torch.no_grad()
    def get_clone_mask(self,actived_scale:torch.Tensor)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads >= self.grad_threshold
        tiny_pts_mask = actived_scale.max(dim=0).values <= self.percent_dense*self.screen_extent
        selected_pts_mask = abnormal_mask&tiny_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def get_split_mask(self,actived_scale:torch.Tensor,N=2)->torch.Tensor:
        mean2d_grads=StatisticsHelperInst.get_mean('mean2d_grad')
        abnormal_mask = mean2d_grads >= self.grad_threshold
        large_pts_mask = actived_scale.max(dim=0).values > self.percent_dense*self.screen_extent
        selected_pts_mask=abnormal_mask&large_pts_mask
        return selected_pts_mask
    
    @torch.no_grad()
    def prune(self,optimizer:torch.optim.Optimizer):
        
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        prune_mask=self.get_prune_mask(opacity.sigmoid(),scale.exp())
        total_points = int(prune_mask.numel())
        remaining_points = int((~prune_mask).sum().item())
        # Early opacity estimates are noisy in masked training. A single
        # aggressive prune must never discard most of the usable surface.
        minimum_remaining = min(total_points, max(256, total_points // 2))
        if remaining_points < minimum_remaining:
            print(
                f"[LiteGS] Skipping unstable prune: it would keep only "
                f"{remaining_points}/{total_points} Gaussians "
                f"(minimum safe count: {minimum_remaining})."
            )
            return
        if self.bCluster:
            N=prune_mask.sum()
            chunk_num=int(N/chunk_size)
            del_limit=chunk_num*chunk_size
            del_indices=prune_mask.nonzero()[:del_limit,0]
            prune_mask=torch.zeros_like(prune_mask)
            prune_mask[del_indices]=True
        print(
            f"[LiteGS] Pruned {int(prune_mask.sum().item())} Gaussians; "
            f"{int((~prune_mask).sum().item())} remain."
        )
        self._prune_optimizer(~prune_mask,optimizer)
        optimizer.state.clear()#prune large point damage the img
        return

    @torch.no_grad()
    def split_and_clone(self,optimizer:torch.optim.Optimizer):
        
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        if self.bCluster:
            chunk_size=xyz.shape[-1]
            xyz,scale,rot,sh_0,sh_rest,opacity=cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)

        clone_mask=self.get_clone_mask(scale.exp())
        split_mask=self.get_split_mask(scale.exp())
        current_points = int(xyz.shape[-1])
        if self.max_gaussians > 0:
            remaining_capacity = self.max_gaussians - current_points
            candidate_mask = clone_mask | split_mask
            candidate_count = int(candidate_mask.sum().item())
            if remaining_capacity <= 0:
                print(
                    f"[LiteGS] Skipping densification: reached the "
                    f"{self.max_gaussians} Gaussian safety limit."
                )
                return
            if candidate_count > remaining_capacity:
                mean2d_grads = StatisticsHelperInst.get_mean('mean2d_grad')
                candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten()
                keep_count = min(remaining_capacity, int(candidate_indices.numel()))
                strongest_local = torch.topk(
                    mean2d_grads[candidate_indices],
                    k=keep_count,
                    largest=True,
                    sorted=False,
                ).indices
                keep_mask = torch.zeros_like(candidate_mask)
                keep_mask[candidate_indices[strongest_local]] = True
                clone_mask &= keep_mask
                split_mask &= keep_mask
                print(
                    f"[LiteGS] Capped densification to {keep_count} additions; "
                    f"the model will reach at most {self.max_gaussians} Gaussians."
                )
        if not self._reported_gradient_stats:
            mean2d_grads = StatisticsHelperInst.get_mean('mean2d_grad')
            finite_grads = mean2d_grads[torch.isfinite(mean2d_grads)]
            if finite_grads.numel() > 0:
                quantiles = torch.quantile(
                    finite_grads.float(),
                    torch.tensor([0.5, 0.9, 0.99], device=finite_grads.device),
                )
                print(
                    "[LiteGS] Densification gradient stats: "
                    f"median={float(quantiles[0]):.3e}, "
                    f"p90={float(quantiles[1]):.3e}, "
                    f"p99={float(quantiles[2]):.3e}, "
                    f"threshold={self.grad_threshold:.3e}."
                )
            self._reported_gradient_stats = True

        #split
        stds=scale[...,split_mask].exp()
        means=torch.zeros((3,stds.size(-1)),device="cuda")
        samples = torch.normal(mean=means, std=stds).unsqueeze(0)
        transform_matrix=wrapper.CreateTransformMatrix.call_fused(scale[...,split_mask].exp(),torch.nn.functional.normalize(rot[...,split_mask],dim=0))
        rotation_matrix=transform_matrix[:3,:3]
        shift=(samples.permute(2,0,1))@rotation_matrix.permute(2,0,1)
        shift=shift.permute(1,2,0).squeeze(0)
        
        split_xyz=xyz[...,split_mask]+shift
        clone_xyz=xyz[...,clone_mask]
        append_xyz=torch.cat((split_xyz,clone_xyz),dim=-1)
        xyz.data[...,split_mask]-=shift
        
        split_scale = (scale[...,split_mask].exp() / (0.8*2)).log()
        clone_scale = scale[...,clone_mask]
        append_scale = torch.cat((split_scale,clone_scale),dim=-1)
        scale.data[...,split_mask]=split_scale

        split_rot=rot[...,split_mask]
        clone_rot=rot[...,clone_mask]
        append_rot = torch.cat((split_rot,clone_rot),dim=-1)

        split_sh_0=sh_0[...,split_mask]
        clone_sh_0=sh_0[...,clone_mask]
        append_sh_0 = torch.cat((split_sh_0,clone_sh_0),dim=-1)

        split_sh_rest=sh_rest[...,split_mask]
        clone_sh_rest=sh_rest[...,clone_mask]
        append_sh_rest = torch.cat((split_sh_rest,clone_sh_rest),dim=-1)

        split_opacity=opacity[...,split_mask]
        clone_opacity=opacity[...,clone_mask]
        append_opacity = torch.cat((split_opacity,clone_opacity),dim=-1)

        if self.bCluster:
            N=append_xyz.shape[-1]
            chunk_num=int(N/chunk_size)
            append_limit=chunk_num*chunk_size
            append_xyz,append_scale,append_rot,append_sh_0,append_sh_rest,append_opacity=cluster.cluster_points(
                chunk_size,append_xyz[...,:append_limit],append_scale[...,:append_limit],
                append_rot[...,:append_limit],append_sh_0[...,:append_limit],
                append_sh_rest[...,:append_limit],append_opacity[...,:append_limit])

        dict_clone = {"xyz": append_xyz,
                      "scale": append_scale,
                      "rot" : append_rot,
                      "sh_0": append_sh_0,
                      "sh_rest": append_sh_rest,
                      "opacity" : append_opacity}
        
        #print("\n#clone:{0} #split:{1} #points:{2}".format(clone_mask.sum().cpu(),split_mask.sum().cpu(),xyz.shape[-1]+append_xyz.shape[-1]*append_xyz.shape[-2]))
        self._cat_tensors_to_optimizer(dict_clone,optimizer)
        return
    
    @torch.no_grad()
    def reset_opacity(self,optimizer):
        xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        actived_opacities=opacity.sigmoid()
        decay_rate=0.5
        decay_mask=(actived_opacities>1/(255*decay_rate-1))
        decay_rate=decay_mask*decay_rate+(~decay_mask)*1.0
        opacity.data=inverse_sigmoid(actived_opacities*decay_rate)#(actived_opacities.clamp_max(0.005))
        optimizer.state.clear()
        return
    
    @torch.no_grad()
    def is_densify_actived(self,epoch:int):

        return epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from and (
            epoch%self.densify_params.densification_interval==0 or
            epoch%self.densify_params.prune_interval==0)

    @torch.no_grad()
    def step(self,optimizer:torch.optim.Optimizer,epoch:int):
        if epoch<self.densify_params.densify_until and epoch>=self.densify_params.densify_from:
            bUpdate=False
            if epoch%self.densify_params.densification_interval==0:
                self.split_and_clone(optimizer)
                bUpdate=True
            if epoch%self.densify_params.prune_interval==0:
                self.prune(optimizer)
                bUpdate=True
            if epoch%self.densify_params.opacity_reset_interval==0:
                self.reset_opacity(optimizer)
                bUpdate=True
            if bUpdate:
                xyz,scale,rot,sh_0,sh_rest,opacity=self._get_params_from_optimizer(optimizer)
                statistics_chunk_num = xyz.shape[-2] if self.bCluster else 1
                StatisticsHelperInst.reset(statistics_chunk_num,xyz.shape[-1],self.is_densify_actived)
                torch.cuda.empty_cache()
        return self._get_params_from_optimizer(optimizer)
    
