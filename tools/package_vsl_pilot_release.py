"""Package the derived, pose-only VSL pilot without copying source videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CLASSES = [
    "Cam on", "Cam Trai", "Di Dao", "Dia Chi", "Hap dan", "Kha Nang",
    "Khong Dep", "Khong quen", "Ky nang", "Le Halloween", "May Man",
    "Ngay nay", "Nghi Hoc", "Nhan Vien", "Ruc Ro", "San Truong", "Thay",
    "Thuong Xuyen", "Tiep tan", "Toi", "Tu Choi", "Xin chao", "Xin loi",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(CLASSES) != 23:
        raise RuntimeError("Unexpected VSL class count")

    args.output.mkdir(parents=True, exist_ok=True)
    shapes = {}
    artifacts = []
    for split in ("train", "val", "test"):
        paths = {
            "left": args.cache / f"{split}_left.npy",
            "right": args.cache / f"{split}_right.npy",
            "global": args.cache / f"{split}_global.npy",
        }
        labels = args.cache / ("train_global.npy" if split == "train" else f"{split}_global.npy")
        label_path = args.cache / "aug_r10_labels.npy" if split == "train" else None

        arrays = {name: np.load(path) for name, path in paths.items()}
        if split == "train":
            # The non-augmented train cache has no separate label file; labels are
            # class-major with 18 samples per class in the canonical split.
            y = np.repeat(np.arange(len(CLASSES), dtype=np.int64), 18)
        else:
            y = np.repeat(np.arange(len(CLASSES), dtype=np.int64), 6)

        n = len(y)
        if any(len(value) != n for value in arrays.values()):
            raise ValueError(f"Unexpected {split} sample count")
        output = args.output / f"{split}.npz"
        np.savez_compressed(output, left=arrays["left"], right=arrays["right"],
                            global_context=arrays["global"], labels=y)
        shapes[split] = {
            "left": list(arrays["left"].shape),
            "right": list(arrays["right"].shape),
            "global": list(arrays["global"].shape),
            "labels": list(y.shape),
        }
        artifacts.append({"path": output.name, "sha256": sha256(output)})

    metadata = {
        "name": "VSL-Pilot-23 derived pose release candidate",
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "split": {"train": 414, "val": 138, "test": 138},
        "representation": {
            "kind": "derived pose features",
            "left_dim": 165,
            "right_dim": 165,
            "global_dim": 23,
            "frames": 40,
            "source_videos_included": False,
        },
        "shapes": shapes,
        "artifacts": artifacts,
        "release_status": "candidate; obtain documented public-release consent before publication",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in artifacts),
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
