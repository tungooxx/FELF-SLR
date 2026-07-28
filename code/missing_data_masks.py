"""Shared missing-data mask schema for 457-D WLASL frame vectors.

The legacy feature vector remains unchanged. A parallel seven-value mask makes
real normalized zeros distinguishable from missing detections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


MASK_NAMES = (
    "left_hand_valid",
    "right_hand_valid",
    "pose_valid",
    "frame_valid",
    "left_hand_missing_ratio",
    "right_hand_missing_ratio",
    "was_interpolated",
)
MASK_DIM = len(MASK_NAMES)

POSE_SLICE = slice(0, 132)
LEFT_HAND_SLICE = slice(132, 195)
RIGHT_HAND_SLICE = slice(195, 258)


def _joint_missing_ratio(flat_xyz: np.ndarray) -> float:
    joints = np.asarray(flat_xyz, dtype=np.float32).reshape(21, 3)
    valid = np.isfinite(joints).all(axis=1) & (np.abs(joints).sum(axis=1) > 1e-8)
    return float(1.0 - valid.mean())


def mask_from_legacy_frame(
    frame: np.ndarray,
    *,
    frame_valid: bool | None = None,
    was_interpolated: bool = False,
) -> np.ndarray:
    """Infer a mask from one legacy 457-D frame.

    Backfilled caches cannot distinguish decode failure from complete
    MediaPipe failure, so inferred ``frame_valid`` is false for all-zero
    vectors. New extraction code should pass the actual decode status.
    """
    frame = np.asarray(frame)
    pose_valid = bool(np.isfinite(frame[POSE_SLICE]).all() and np.abs(frame[POSE_SLICE]).sum() > 1e-8)
    left_ratio = _joint_missing_ratio(frame[LEFT_HAND_SLICE])
    right_ratio = _joint_missing_ratio(frame[RIGHT_HAND_SLICE])
    if frame_valid is None:
        frame_valid = bool(np.isfinite(frame).all() and np.abs(frame).sum() > 1e-8)
    return np.asarray(
        [
            left_ratio < 1.0,
            right_ratio < 1.0,
            pose_valid,
            frame_valid,
            left_ratio,
            right_ratio,
            was_interpolated,
        ],
        dtype=np.float32,
    )


def mask_from_detection(
    *,
    pose_present: bool,
    left_hand_present: bool,
    right_hand_present: bool,
    frame_valid: bool,
    was_interpolated: bool = False,
) -> np.ndarray:
    """Build an exact extraction-time mask from MediaPipe detection status."""
    return np.asarray(
        [
            left_hand_present,
            right_hand_present,
            pose_present,
            frame_valid,
            0.0 if left_hand_present else 1.0,
            0.0 if right_hand_present else 1.0,
            was_interpolated,
        ],
        dtype=np.float32,
    )


def masks_from_legacy_sequence(sequence: np.ndarray) -> np.ndarray:
    return np.stack([mask_from_legacy_frame(frame) for frame in sequence]).astype(np.float32)


def append_masks_to_global(global_features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Append the seven reliability values for mask-aware retrained models."""
    global_features = np.asarray(global_features, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if global_features.shape[:-1] != masks.shape[:-1] or masks.shape[-1] != MASK_DIM:
        raise ValueError(f"Incompatible global/mask shapes: {global_features.shape} and {masks.shape}")
    return np.concatenate([global_features, masks], axis=-1)


def save_video_masks(video_dir: str | Path, masks: np.ndarray) -> None:
    """Save masks without placing extra .npy files beside legacy frame files."""
    mask_dir = Path(video_dir) / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(np.asarray(masks, dtype=np.float32)):
        np.save(mask_dir / f"{index}.npy", mask)
