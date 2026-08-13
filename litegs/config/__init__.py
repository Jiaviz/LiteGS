from .. import arguments

def get_default_arg()->tuple[arguments.ModelParams,arguments.OptimizationParams,arguments.PipelineParams,arguments.DensifyParams]:
    lp=arguments.ModelParams.get_class_default_obj()
    op=arguments.OptimizationParams.get_class_default_obj()
    pp=arguments.PipelineParams.get_class_default_obj()
    dp=arguments.DensifyParams.get_class_default_obj()
    return lp,op,pp,dp

def get_quality_arg()->tuple[arguments.OptimizationParams,arguments.PipelineParams,arguments.DensifyParams]:
    lp=arguments.ModelParams.get_class_default_obj()
    op=arguments.OptimizationParams.get_class_default_obj()
    pp=arguments.PipelineParams.get_class_default_obj()
    dp=arguments.DensifyParams.get_class_default_obj()
    dp.densify_grad_threshold=0.00015
    return lp,op,pp,dp

def get_2dgs_fast_arg()->tuple[arguments.ModelParams,arguments.OptimizationParams,arguments.PipelineParams,arguments.DensifyParams]:
    """Defaults for fast CUDA 2DGS geometry training on object-centric scenes."""
    lp,op,pp,dp=get_default_arg()
    pp.gaussian_dimension=2
    pp.cluster_size=0
    pp.tile_size=16
    pp.sparse_grad=False
    op.iterations=10000
    op.geometry_start_iteration=500
    dp.densify_from=10
    dp.densify_until=80
    dp.densification_interval=10
    dp.prune_interval=20
    dp.opacity_reset_interval=40
    # The official 2DGS screen-space gradient magnitude varies with image
    # resolution. A threshold of 4.5 suppresses every candidate on typical
    # 1.6K object-centric captures (observed median ~3.2), after which pruning
    # collapses the model. This value remains selective while allowing those
    # detailed captures to densify.
    dp.densify_grad_threshold=2.0
    # A 60k cap loses more than 5 dB on detailed DTU/object imagery. Preserve
    # reconstruction quality while retaining a guard against runaway growth;
    # constrained runs can still override this explicitly.
    dp.max_gaussians=2000000
    return lp,op,pp,dp
