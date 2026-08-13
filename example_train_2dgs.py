#!/usr/bin/env python3
"""Fast CUDA 2D Gaussian Splatting training entry point."""

from argparse import ArgumentParser
import sys

import litegs
import litegs.config


def main() -> None:
    parser = ArgumentParser(description="Train geometry-supervised CUDA 2DGS")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=litegs.config.get_2dgs_fast_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    parser.add_argument("--test_epochs",nargs="+",type=int,default=[])
    parser.add_argument("--save_epochs",nargs="+",type=int,default=[])
    parser.add_argument("--checkpoint_epochs",nargs="+",type=int,default=[])
    parser.add_argument("--start_checkpoint",type=str,default=None)
    parser.add_argument("--save_eval_images",action="store_true")
    parser.add_argument("--eval_image_count",type=int,default=8)
    args=parser.parse_args(sys.argv[1:])
    litegs.training.start(
        litegs.arguments.ModelParams.extract(args),
        litegs.arguments.OptimizationParams.extract(args),
        litegs.arguments.PipelineParams.extract(args),
        litegs.arguments.DensifyParams.extract(args),
        args.test_epochs,args.save_epochs,args.checkpoint_epochs,args.start_checkpoint,
        save_eval_images=args.save_eval_images,eval_image_count=args.eval_image_count,
    )


if __name__ == "__main__":
    main()
