"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from pprint import pprint

#wandb
import wandb

debug = False

if debug:
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def main(args) -> None:
    """main"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
                # >>> wandb 관련 argument들은 cfg로 안 넣도록 제외
                "use_wandb",
                "wandb_project",
                "wandb_run_name",
            ]
            and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    rank = safe_get_rank()

    if rank == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    # >>>>>>>>>>>>>>>>>> W&B 초기화 <<<<<<<<<<<<<<<<<<
    wandb_enabled = args.use_wandb and (rank == 0)
    if wandb_enabled:
        # run name은 기본적으로 config 파일 이름 사용
        default_run_name = os.path.splitext(os.path.basename(args.config))[0]
        run_name = args.wandb_run_name or default_run_name

        # cfg.yaml_cfg 는 YAML 전체 설정 dict
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=cfg.yaml_cfg,
        )

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()
    if wandb_enabled:
        wandb.finish()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="device",
    )
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # env
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")

    # wandb
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dfine",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="wandb run name (default: config file name)",
    )


    args = parser.parse_args()

    main(args)
