"""Clean LRG/RF expert wrapper for FELF-SLR pipeline reproduction.

This wrapper trains one B6-derived expert using the old LocalGlobal trainer:
  --expert-kind lrg  -> B6 + fixed025 static-motion residual expert
  --expert-kind rf   -> B6 + fixed025 residual reliability-fusion expert
"""

from __future__ import annotations

import argparse
import sys

import wlasl_train_old_localglobal_sweep_subset as old


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-kind", choices=["lrg", "rf"], required=True)
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    forwarded = [
        "--feature-kind",
        "old",
        "--pool-mode",
        "mean",
        "--left-branch-dim",
        "96",
        "--right-branch-dim",
        "96",
        "--global-branch-dim",
        "96",
        "--conv-kernel",
        "3",
        "--conv-layers",
        "1",
        "--num-layers",
        "2",
        "--num-heads",
        "4",
        "--ff-dim",
        "768",
        "--dropout",
        "0.2",
        "--mixup-alpha",
        "0.2",
    ]
    if wrapper_args.expert_kind == "lrg":
        forwarded += [
            "--use-lrg-residual",
            "--lrg-residual-gamma-init",
            "0.25",
            "--lrg-residual-gamma-mode",
            "fixed",
        ]
    elif wrapper_args.expert_kind == "rf":
        forwarded += [
            "--use-residual-reliability-fusion",
            "--residual-rf-beta-init",
            "0.25",
            "--residual-rf-beta-mode",
            "fixed",
        ]
    else:
        raise ValueError(wrapper_args.expert_kind)

    sys.argv = [sys.argv[0], *forwarded, *remaining]
    old.main()


if __name__ == "__main__":
    main()
