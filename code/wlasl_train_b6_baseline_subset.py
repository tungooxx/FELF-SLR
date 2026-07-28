"""Clean B6 baseline wrapper for FELF-SLR pipeline reproduction.

This wrapper locks the old LocalGlobal sweep script to the B6 baseline
configuration:
  old frame part-aware features, 96/96/96 branches, dropout 0.2,
  mean pooling, Conv1D kernel 3, 2-layer Transformer, ArcFace.
"""

from __future__ import annotations

import sys

import wlasl_train_old_localglobal_sweep_subset as old


def main() -> None:
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
    sys.argv = [sys.argv[0], *forwarded, *sys.argv[1:]]
    old.main()


if __name__ == "__main__":
    main()
