"""SPOTER-style geometric landmark augmentations for WLASL-compatible raw frames."""

from __future__ import annotations

import math

import numpy as np

from wlasl_train_local_global_arcface import SEQUENCE_LENGTH


POSE_SLICE = slice(0, 132)
LEFT_SLICE = slice(132, 195)
RIGHT_SLICE = slice(195, 258)


def split_raw(seq: np.ndarray):
    pose = seq[:, POSE_SLICE].reshape(seq.shape[0], 33, 4).copy()
    left = seq[:, LEFT_SLICE].reshape(seq.shape[0], 21, 3).copy()
    right = seq[:, RIGHT_SLICE].reshape(seq.shape[0], 21, 3).copy()
    return pose, left, right


def join_raw(seq: np.ndarray, pose: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = seq.copy()
    out[:, POSE_SLICE] = pose.reshape(seq.shape[0], -1)
    out[:, LEFT_SLICE] = left.reshape(seq.shape[0], -1)
    out[:, RIGHT_SLICE] = right.reshape(seq.shape[0], -1)
    return out


def valid_xyz(points: np.ndarray) -> np.ndarray:
    return np.abs(points[..., :3]).sum(axis=-1) > 1e-6


def valid_pose(points: np.ndarray) -> np.ndarray:
    return (np.abs(points[..., :2]).sum(axis=-1) > 1e-6) | (points[..., 3] > 1e-6)


def rotate_xy(points: np.ndarray, mask: np.ndarray, center: np.ndarray, angle: float) -> None:
    if not np.any(mask):
        return
    c, s = math.cos(angle), math.sin(angle)
    xy = points[mask, :2] - center[None, :]
    rot = np.empty_like(xy)
    rot[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    rot[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    points[mask, :2] = rot + center[None, :]


def scale_xy(points: np.ndarray, mask: np.ndarray, center: np.ndarray, sx: float, sy: float) -> None:
    if not np.any(mask):
        return
    xy = points[mask, :2] - center[None, :]
    xy[:, 0] *= sx
    xy[:, 1] *= sy
    points[mask, :2] = xy + center[None, :]


def perspective_xy(points: np.ndarray, mask: np.ndarray, center: np.ndarray, px: float, py: float) -> None:
    if not np.any(mask):
        return
    xy = points[mask, :2] - center[None, :]
    den = 1.0 + px * xy[:, 0] + py * xy[:, 1]
    den = np.clip(den, 0.65, 1.35)
    points[mask, :2] = xy / den[:, None] + center[None, :]


def frame_center(pose_f: np.ndarray, left_f: np.ndarray, right_f: np.ndarray) -> np.ndarray:
    chunks = []
    pm = valid_pose(pose_f)
    lm = valid_xyz(left_f)
    rm = valid_xyz(right_f)
    if np.any(pm):
        chunks.append(pose_f[pm, :2])
    if np.any(lm):
        chunks.append(left_f[lm, :2])
    if np.any(rm):
        chunks.append(right_f[rm, :2])
    if not chunks:
        return np.array([0.5, 0.5], dtype=np.float32)
    return np.concatenate(chunks, axis=0).mean(axis=0).astype(np.float32)


def apply_rotate(seq: np.ndarray, max_deg: float = 18.0) -> np.ndarray:
    pose, left, right = split_raw(seq)
    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    for t in range(seq.shape[0]):
        center = frame_center(pose[t], left[t], right[t])
        rotate_xy(pose[t], valid_pose(pose[t]), center, angle)
        rotate_xy(left[t], valid_xyz(left[t]), center, angle)
        rotate_xy(right[t], valid_xyz(right[t]), center, angle)
    return join_raw(seq, pose, left, right)


def apply_squeeze(seq: np.ndarray, min_scale: float = 0.82, max_scale: float = 1.18) -> np.ndarray:
    pose, left, right = split_raw(seq)
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)
    for t in range(seq.shape[0]):
        center = frame_center(pose[t], left[t], right[t])
        scale_xy(pose[t], valid_pose(pose[t]), center, sx, sy)
        scale_xy(left[t], valid_xyz(left[t]), center, sx, sy)
        scale_xy(right[t], valid_xyz(right[t]), center, sx, sy)
    return join_raw(seq, pose, left, right)


def apply_perspective(seq: np.ndarray, strength: float = 0.18) -> np.ndarray:
    pose, left, right = split_raw(seq)
    px = np.random.uniform(-strength, strength)
    py = np.random.uniform(-strength, strength)
    for t in range(seq.shape[0]):
        center = frame_center(pose[t], left[t], right[t])
        perspective_xy(pose[t], valid_pose(pose[t]), center, px, py)
        perspective_xy(left[t], valid_xyz(left[t]), center, px, py)
        perspective_xy(right[t], valid_xyz(right[t]), center, px, py)
    return join_raw(seq, pose, left, right)


def apply_arm_joint_rotate(seq: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    pose, left, right = split_raw(seq)
    left_angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    right_angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    for t in range(seq.shape[0]):
        for side, angle, hand in [("left", left_angle, left[t]), ("right", right_angle, right[t])]:
            elbow_idx = 13 if side == "left" else 14
            wrist_idx = 15 if side == "left" else 16
            pose_arm = [15, 17, 19, 21] if side == "left" else [16, 18, 20, 22]
            elbow_ok = valid_pose(pose[t][elbow_idx : elbow_idx + 1])[0]
            if elbow_ok:
                center = pose[t, elbow_idx, :2].copy()
                pose_mask = np.zeros(33, dtype=bool)
                pose_mask[pose_arm] = valid_pose(pose[t][pose_arm])
                rotate_xy(pose[t], pose_mask, center, angle)
                rotate_xy(hand, valid_xyz(hand), center, angle)
            else:
                hand_mask = valid_xyz(hand)
                if np.any(hand_mask):
                    rotate_xy(hand, hand_mask, hand[hand_mask, :2].mean(axis=0), angle)
            wrist_ok = valid_pose(pose[t][wrist_idx : wrist_idx + 1])[0]
            hand_mask = valid_xyz(hand)
            if wrist_ok and np.any(hand_mask):
                hand[hand_mask, :2] += (pose[t, wrist_idx, :2] - hand[0, :2])[None, :]
    return join_raw(seq, pose, left, right)


def apply_gaussian_noise(seq: np.ndarray, std: float = 0.006) -> np.ndarray:
    pose, left, right = split_raw(seq)
    for arr, mask_fn in [(pose, valid_pose), (left, valid_xyz), (right, valid_xyz)]:
        for t in range(seq.shape[0]):
            mask = mask_fn(arr[t])
            if np.any(mask):
                arr[t, mask, :2] += np.random.normal(0.0, std, size=(int(mask.sum()), 2)).astype(np.float32)
    return join_raw(seq, pose, left, right)


def _interp_resample(seq: np.ndarray, new_len: int) -> np.ndarray:
    t, d = seq.shape
    if t == new_len:
        return seq.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, new_len)
    out = np.empty((new_len, d), dtype=np.float32)
    for i in range(d):
        out[:, i] = np.interp(x_new, x_old, seq[:, i])
    return out


def _time_stretch_component(seq: np.ndarray) -> np.ndarray:
    factor = np.random.uniform(0.8, 1.25)
    new_len = max(2, int(round(seq.shape[0] * factor)))
    return _interp_resample(_interp_resample(seq, new_len), SEQUENCE_LENGTH)


def _time_warp_component(seq: np.ndarray) -> np.ndarray:
    t, d = seq.shape
    control = np.linspace(0.0, 1.0, 6)
    jitter = np.random.normal(0.0, 0.08, size=control.shape)
    jitter[0] = 0.0
    jitter[-1] = 0.0
    warped = np.clip(control + jitter, 0.0, 1.0)
    warped = np.maximum.accumulate(warped)
    if warped[-1] <= warped[0]:
        return seq.copy()
    warped = (warped - warped[0]) / (warped[-1] - warped[0])
    y = np.interp(np.linspace(0.0, 1.0, t), control, warped) * (t - 1)
    out = np.empty_like(seq)
    for i in range(d):
        out[:, i] = np.interp(np.arange(t), y, seq[:, i])
    return out.astype(np.float32)


def _frame_drop_component(seq: np.ndarray) -> np.ndarray:
    t, d = seq.shape
    drop_prob = np.random.uniform(0.05, 0.2)
    keep = np.random.rand(t) > drop_prob
    if keep.sum() < 2:
        keep[np.random.randint(0, t, size=2)] = True
    kept = seq[keep]
    src, dst = np.linspace(0.0, 1.0, kept.shape[0]), np.linspace(0.0, 1.0, t)
    out = np.empty_like(seq)
    for i in range(d):
        out[:, i] = np.interp(dst, src, kept[:, i])
    return out.astype(np.float32)


def _shift_component(seq: np.ndarray) -> np.ndarray:
    shift = np.random.randint(-2, 3)
    return np.roll(seq, shift=shift, axis=0).astype(np.float32) if shift != 0 else seq.copy()


def _affine_component(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    theta = np.deg2rad(np.random.uniform(-20, 20))
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    scale = np.random.uniform(0.9, 1.12)
    trans = np.random.normal(0.0, np.random.uniform(0.0, 0.03), (1, 3)).astype(np.float32)
    for t in range(SEQUENCE_LENGTH):
        pose = out[t, 0:132].reshape(33, 4)
        lh = out[t, 132:195].reshape(21, 3)
        rh = out[t, 195:258].reshape(21, 3)
        for arr in [pose[:, :3], lh, rh]:
            center = arr[:, :2].mean(0, keepdims=True)
            arr[:, :2] = (arr[:, :2] - center) @ r.T * scale + center + trans[0, :2]
            arr[:, 2] *= scale
        out[t, 0:132] = pose.flatten()
        out[t, 132:195] = lh.flatten()
        out[t, 195:258] = rh.flatten()
    return out.astype(np.float32)


def _noise_component(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    noise_std = np.random.uniform(0.005, 0.015)
    for t in range(SEQUENCE_LENGTH):
        pose = out[t, 0:132].reshape(33, 4)
        lh = out[t, 132:195].reshape(21, 3)
        rh = out[t, 195:258].reshape(21, 3)
        pose[:, :3] += np.random.normal(0.0, noise_std, (33, 3)).astype(np.float32)
        lh += np.random.normal(0.0, noise_std, (21, 3)).astype(np.float32)
        rh += np.random.normal(0.0, noise_std, (21, 3)).astype(np.float32)
        out[t, 0:132] = pose.flatten()
        out[t, 132:195] = lh.flatten()
        out[t, 195:258] = rh.flatten()
    return out.astype(np.float32)


def _landmark_drop_component(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    dm_p = np.random.rand(33) < np.random.uniform(0.05, 0.15)
    dm_l = np.random.rand(21) < np.random.uniform(0.05, 0.15)
    dm_r = np.random.rand(21) < np.random.uniform(0.05, 0.15)
    for t in range(SEQUENCE_LENGTH):
        pose = out[t, 0:132].reshape(33, 4)
        lh = out[t, 132:195].reshape(21, 3)
        rh = out[t, 195:258].reshape(21, 3)
        pose[dm_p, :3] = 0.0
        lh[dm_l] = 0.0
        rh[dm_r] = 0.0
        out[t, 0:132] = pose.flatten()
        out[t, 132:195] = lh.flatten()
        out[t, 195:258] = rh.flatten()
    return out.astype(np.float32)


def augment_fixed_components(raw: np.ndarray, mode: str) -> np.ndarray:
    components = {
        "stretch": [_time_stretch_component],
        "warp": [_time_warp_component],
        "frame_drop": [_frame_drop_component],
        "shift": [_shift_component],
        "affine": [_affine_component],
        "noise": [_noise_component],
        "recommended": [_noise_component],
        "landmark_drop": [_landmark_drop_component],
        "frame_drop_noise": [_frame_drop_component, _noise_component],
        "stretch_noise": [_time_stretch_component, _noise_component],
        "temporal": [_time_stretch_component, _time_warp_component, _frame_drop_component, _shift_component],
        "affine_noise": [_affine_component, _noise_component],
        "temporal_noise": [_time_stretch_component, _time_warp_component, _frame_drop_component, _shift_component, _noise_component],
        "temporal_affine": [_time_stretch_component, _time_warp_component, _frame_drop_component, _shift_component, _affine_component],
        "temporal_affine_noise": [
            _time_stretch_component,
            _time_warp_component,
            _frame_drop_component,
            _shift_component,
            _affine_component,
            _noise_component,
        ],
        "all_components": [
            _time_stretch_component,
            _time_warp_component,
            _frame_drop_component,
            _shift_component,
            _affine_component,
            _noise_component,
            _landmark_drop_component,
        ],
    }
    key = mode.replace("fixed_", "")
    if key not in components:
        raise ValueError(f"Unknown fixed-component augmentation mode: {mode}")
    out = raw.copy()
    for fn in components[key]:
        out = fn(out)
    if out.shape[0] != SEQUENCE_LENGTH:
        out = _interp_resample(out, SEQUENCE_LENGTH)
    return out.astype(np.float32)


def augment_fixed_clean_default(raw: np.ndarray) -> np.ndarray:
    """Original-style probabilistic augmentation, without harmful affine/dropout."""
    out = raw.copy()
    if np.random.rand() < 0.7:
        out = _time_stretch_component(out)
    if np.random.rand() < 0.5:
        out = _time_warp_component(out)
    if np.random.rand() < 0.5:
        out = _frame_drop_component(out)
    if np.random.rand() < 0.5:
        out = _shift_component(out)
    if np.random.rand() < 0.9:
        out = _noise_component(out)
    if out.shape[0] != SEQUENCE_LENGTH:
        out = _interp_resample(out, SEQUENCE_LENGTH)
    return out.astype(np.float32)


def augment_spoter_style(raw: np.ndarray, mode: str) -> np.ndarray:
    """Apply one SPOTER-style augmentation family to a [T,457] raw sequence."""
    if mode == "fixed_clean_default":
        return augment_fixed_clean_default(raw)
    if mode.startswith("fixed_"):
        return augment_fixed_components(raw, mode)
    if mode == "fixed":
        from wlasl_train_streams_arcface import augment_fixed

        return augment_fixed(raw)
    if mode == "none":
        return raw.copy()
    if mode == "rotate":
        return apply_rotate(raw)
    if mode == "squeeze":
        return apply_squeeze(raw)
    if mode == "perspective":
        return apply_perspective(raw)
    if mode == "arm_joint_rotate":
        return apply_arm_joint_rotate(raw)
    if mode in {"all", "all_noise"}:
        out = apply_rotate(raw)
        out = apply_squeeze(out)
        out = apply_perspective(out)
        out = apply_arm_joint_rotate(out)
        if mode == "all_noise":
            out = apply_gaussian_noise(out)
        return out
    raise ValueError(f"Unknown augmentation mode: {mode}")
