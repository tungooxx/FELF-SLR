"""Shared geometry, rectification, and part-aware feature utilities for WLASL experiments."""
from dataclasses import dataclass
import numpy as np


FACE_INDICES = [0, 2, 5, 9, 10]
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
SPREAD_ANGLE_TRIPLETS = [
    (1, 0, 5),
    (5, 0, 9),
    (9, 0, 13),
    (13, 0, 17),
]
FINGER_CHAINS = [
    ((0, 1, 2, 3, 4), np.deg2rad([45.0, 80.0, 90.0]).astype(np.float32)),
    ((0, 5, 6, 7, 8), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
    ((0, 9, 10, 11, 12), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
    ((0, 13, 14, 15, 16), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
    ((0, 17, 18, 19, 20), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
]


@dataclass(frozen=True)
class GeometryConfig:
    name: str
    relative_coords: bool = True
    bone_vectors: bool = True
    bone_lengths: bool = True
    angles: bool = True
    distances: bool = True
    face_context: bool = True
    palm_normals: bool = True
    rectification: bool = True
    rectify_alpha: float = 0.4


def safe_norm(v):
    return np.linalg.norm(v) + 1e-8


def unit(v):
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def compute_palm_normal(hand_xyz, hand_label):
    wrist, index_mcp, pinky_mcp = hand_xyz[0], hand_xyz[5], hand_xyz[17]
    v1, v2 = (index_mcp - wrist, pinky_mcp - wrist) if hand_label == 'Right' else (pinky_mcp - wrist, index_mcp - wrist)
    normal = np.cross(v1, v2)
    n = np.linalg.norm(normal)
    if n <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (normal / n).astype(np.float32)


def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (safe_norm(ba) * safe_norm(bc))
    return np.arccos(np.clip(cos, -1.0, 1.0)).astype(np.float32)


def orthogonal_unit(v):
    basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(unit(v), basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ortho = np.cross(v, basis)
    if np.linalg.norm(ortho) <= 1e-8:
        ortho = np.cross(v, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return unit(ortho)


def rotate_about_axis(vec, axis, angle):
    axis = unit(axis)
    vec = vec.astype(np.float32)
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    return (vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)).astype(np.float32)


def rotate_towards(base_dir, target_dir, bend):
    base_dir = unit(base_dir)
    target_dir = unit(target_dir)
    cross = np.cross(base_dir, target_dir)
    if np.linalg.norm(cross) <= 1e-8:
        if np.dot(base_dir, target_dir) >= 0:
            return base_dir
        cross = orthogonal_unit(base_dir)
    return unit(rotate_about_axis(base_dir, cross, bend))


def rectify_hand_kinematics(hand_xyz, alpha=0.4):
    if hand_xyz.shape != (21, 3) or not np.any(hand_xyz):
        return hand_xyz.astype(np.float32)

    rect = hand_xyz.copy().astype(np.float32)
    for chain, max_bends in FINGER_CHAINS:
        pts = hand_xyz[list(chain)].astype(np.float32)
        if np.any(np.linalg.norm(pts, axis=1) <= 1e-8):
            continue

        seg_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1).astype(np.float32)
        if np.any(seg_lens <= 1e-8):
            continue

        new_pts = pts.copy()
        prev_dir = unit(pts[1] - pts[0])
        for idx in range(1, len(chain) - 1):
            raw_dir = unit(pts[idx + 1] - pts[idx])
            bend = np.arccos(np.clip(np.dot(prev_dir, raw_dir), -1.0, 1.0)).astype(np.float32)
            bend = min(float(bend), float(max_bends[idx - 1]))
            child_dir = rotate_towards(prev_dir, raw_dir, bend)
            new_pts[idx + 1] = new_pts[idx] + seg_lens[idx] * child_dir
            prev_dir = unit(new_pts[idx + 1] - new_pts[idx])

        blended = pts + alpha * (new_pts - pts)
        for local_i, joint_i in enumerate(chain[1:], start=1):
            rect[joint_i] = blended[local_i]
    return rect.astype(np.float32)


def geometry_feature_dim(config):
    dim = 0
    if config.relative_coords:
        dim += 120
    if config.bone_vectors:
        dim += 120
    if config.bone_lengths:
        dim += 40
    if config.angles:
        dim += 38
    if config.face_context:
        dim += 12
    if config.palm_normals:
        dim += 6
    if config.distances:
        dim += 11
    return dim


def _prepare_parts(vec, config):
    pose = vec[0:132].reshape(33, 4)[:, :3]
    lh = vec[132:195].reshape(21, 3).astype(np.float32)
    rh = vec[195:258].reshape(21, 3).astype(np.float32)
    if config.rectification:
        lh = rectify_hand_kinematics(lh, config.rectify_alpha)
        rh = rectify_hand_kinematics(rh, config.rectify_alpha)
    return pose, lh, rh


def extract_geometry_features(vec, config):
    pose, lh, rh = _prepare_parts(vec, config)
    face = pose[FACE_INDICES]
    nose = face[0:1].copy()
    face_scale = np.linalg.norm(face[1] - face[2])
    if face_scale <= 1e-6:
        face_scale = 1.0
    face_rel = ((face[1:] - nose) / face_scale).astype(np.float32)

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    body_scale = np.linalg.norm(left_shoulder - right_shoulder)
    if body_scale <= 1e-6:
        body_scale = 1.0

    def hand_features(hand_xyz, hand_label):
        rel = hand_xyz - hand_xyz[0:1]
        hand_scale = np.linalg.norm(hand_xyz[0] - hand_xyz[9])
        if hand_scale <= 1e-6:
            hand_scale = 1.0
        rel = rel / hand_scale

        bone_vec = np.array([rel[c] - rel[p] for p, c in HAND_BONES], dtype=np.float32)
        bone_len = np.linalg.norm(bone_vec, axis=1).astype(np.float32)
        flex = np.array([compute_angle(rel[a], rel[b], rel[c]) for a, b, c in FINGER_ANGLE_TRIPLETS], dtype=np.float32)
        spread = np.array([compute_angle(rel[a], rel[b], rel[c]) for a, b, c in SPREAD_ANGLE_TRIPLETS], dtype=np.float32)
        palm = compute_palm_normal(hand_xyz, hand_label) if np.any(hand_xyz != 0) else np.zeros(3, dtype=np.float32)
        pinch = np.linalg.norm(hand_xyz[4] - hand_xyz[8]) / hand_scale
        tip_spread = np.mean(np.linalg.norm(hand_xyz[[8, 12, 16, 20]] - hand_xyz[[5, 9, 13, 17]], axis=1)) / hand_scale
        scale_ratio = hand_scale / body_scale
        return {
            'rel': rel[1:].flatten().astype(np.float32),
            'bone_vec': bone_vec.flatten().astype(np.float32),
            'bone_len': bone_len,
            'flex': flex,
            'spread': spread,
            'palm': palm,
            'pinch': np.array([pinch], dtype=np.float32),
            'tip_spread': np.array([tip_spread], dtype=np.float32),
            'scale_ratio': np.array([scale_ratio], dtype=np.float32),
        }

    left = hand_features(lh, 'Left')
    right = hand_features(rh, 'Right')

    wrist_to_nose = np.array([
        np.linalg.norm(lh[0] - nose[0]) / body_scale,
        np.linalg.norm(rh[0] - nose[0]) / body_scale,
    ], dtype=np.float32)
    wrist_to_mouth = np.array([
        np.linalg.norm(lh[0] - face[3]) / body_scale,
        np.linalg.norm(rh[0] - face[4]) / body_scale,
    ], dtype=np.float32)
    inter_wrist = np.array([np.linalg.norm(lh[0] - rh[0]) / body_scale], dtype=np.float32)

    parts = []
    if config.relative_coords:
        parts.extend([left['rel'], right['rel']])
    if config.bone_vectors:
        parts.extend([left['bone_vec'], right['bone_vec']])
    if config.bone_lengths:
        parts.extend([left['bone_len'], right['bone_len']])
    if config.angles:
        parts.extend([left['flex'], right['flex'], left['spread'], right['spread']])
    if config.face_context:
        parts.append(face_rel.flatten())
    if config.palm_normals:
        parts.extend([left['palm'], right['palm']])
    if config.distances:
        parts.extend([
            wrist_to_nose,
            wrist_to_mouth,
            inter_wrist,
            left['pinch'],
            right['pinch'],
            left['tip_spread'],
            right['tip_spread'],
            left['scale_ratio'],
            right['scale_ratio'],
        ])
    if not parts:
        raise ValueError('GeometryConfig produced no features.')
    return np.concatenate(parts).astype(np.float32)


def build_geometry_extractor(config):
    def extract(vec):
        return extract_geometry_features(vec, config)
    return extract


def extract_part_aware_features(vec, rectify_alpha=0.4, hand_scale_floor=1e-6):
    cfg = GeometryConfig(name='part_aware', rectification=True, rectify_alpha=rectify_alpha)
    pose, lh, rh = _prepare_parts(vec, cfg)
    face = pose[FACE_INDICES]
    nose = face[0:1].copy()
    face_scale = np.linalg.norm(face[1] - face[2])
    if face_scale <= 1e-6:
        face_scale = 1.0
    face_rel = ((face[1:] - nose) / face_scale).astype(np.float32)

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    body_scale = np.linalg.norm(left_shoulder - right_shoulder)
    if body_scale <= 1e-6:
        body_scale = 1.0

    def local_branch(hand_xyz, hand_label):
        rel = hand_xyz - hand_xyz[0:1]
        hand_valid = np.any(hand_xyz != 0)
        hand_scale = np.linalg.norm(hand_xyz[0] - hand_xyz[9])
        if not hand_valid:
            hand_scale = 1.0
        elif hand_scale <= hand_scale_floor:
            hand_scale = 1.0 if hand_scale_floor <= 1e-6 else float(hand_scale_floor)
        rel = rel / hand_scale
        bone_vec = np.array([rel[c] - rel[p] for p, c in HAND_BONES], dtype=np.float32)
        bone_len = np.linalg.norm(bone_vec, axis=1).astype(np.float32)
        flex = np.array([compute_angle(rel[a], rel[b], rel[c]) for a, b, c in FINGER_ANGLE_TRIPLETS], dtype=np.float32)
        spread = np.array([compute_angle(rel[a], rel[b], rel[c]) for a, b, c in SPREAD_ANGLE_TRIPLETS], dtype=np.float32)
        palm = compute_palm_normal(hand_xyz, hand_label) if hand_valid else np.zeros(3, dtype=np.float32)
        pinch = np.linalg.norm(hand_xyz[4] - hand_xyz[8]) / hand_scale
        tip_spread = np.mean(np.linalg.norm(hand_xyz[[8, 12, 16, 20]] - hand_xyz[[5, 9, 13, 17]], axis=1)) / hand_scale
        scale_ratio = hand_scale / body_scale
        return np.concatenate([
            rel[1:].flatten(),
            bone_vec.flatten(),
            bone_len,
            flex,
            spread,
            palm,
            np.array([pinch, tip_spread, scale_ratio], dtype=np.float32),
        ]).astype(np.float32)

    left = local_branch(lh, 'Left')
    right = local_branch(rh, 'Right')
    wrist_global = np.concatenate([lh[0], rh[0]]).astype(np.float32)
    inter_wrist = np.array([np.linalg.norm(lh[0] - rh[0]) / body_scale], dtype=np.float32)
    wrist_to_face = np.array([
        np.linalg.norm(lh[0] - face[0]) / body_scale,
        np.linalg.norm(rh[0] - face[0]) / body_scale,
        np.linalg.norm(lh[0] - face[3]) / body_scale,
        np.linalg.norm(rh[0] - face[4]) / body_scale,
    ], dtype=np.float32)
    global_features = np.concatenate([face_rel.flatten(), wrist_global, inter_wrist, wrist_to_face]).astype(np.float32)
    return left, right, global_features


def _camera_axis():
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _safe_difference(next_vec, curr_vec):
    return (next_vec - curr_vec).astype(np.float32)


def _exposure(normal, direction):
    if np.linalg.norm(normal) <= 1e-8 or np.linalg.norm(direction) <= 1e-8:
        return np.float32(0.0)
    return np.float32(np.dot(unit(normal), unit(direction)))


def _local_branch_base(hand_xyz, hand_label, body_scale):
    rel = hand_xyz - hand_xyz[0:1]
    hand_scale = np.linalg.norm(hand_xyz[0] - hand_xyz[9])
    if hand_scale <= 1e-6:
        hand_scale = 1.0
    rel = rel / hand_scale
    bone_vec = np.array([rel[c] - rel[p] for p, c in HAND_BONES], dtype=np.float32)
    bone_len = np.linalg.norm(bone_vec, axis=1).astype(np.float32)
    flex = np.array([compute_angle(rel[a], rel[b], rel[c]) for a, b, c in FINGER_ANGLE_TRIPLETS], dtype=np.float32)
    spread = np.array([compute_angle(rel[a], rel[b], rel[c]) for a, b, c in SPREAD_ANGLE_TRIPLETS], dtype=np.float32)
    palm = compute_palm_normal(hand_xyz, hand_label) if np.any(hand_xyz != 0) else np.zeros(3, dtype=np.float32)
    pinch = np.linalg.norm(hand_xyz[4] - hand_xyz[8]) / hand_scale
    tip_spread = np.mean(np.linalg.norm(hand_xyz[[8, 12, 16, 20]] - hand_xyz[[5, 9, 13, 17]], axis=1)) / hand_scale
    scale_ratio = hand_scale / body_scale
    base = np.concatenate([
        rel[1:].flatten(),
        bone_vec.flatten(),
        bone_len,
        flex,
        spread,
        palm,
        np.array([pinch, tip_spread, scale_ratio], dtype=np.float32),
    ]).astype(np.float32)
    return base, palm


def extract_part_aware_sequence_features(seq, rectify_alpha=0.4):
    cfg = GeometryConfig(name='part_aware_sequence', rectification=True, rectify_alpha=rectify_alpha)
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected sequence of shape [T, D], got {seq.shape}")

    frames = []
    left_palms = []
    right_palms = []
    left_wrists = []
    right_wrists = []
    face_noses = []
    left_face_dirs = []
    right_face_dirs = []

    for vec in seq:
        pose, lh, rh = _prepare_parts(vec, cfg)
        face = pose[FACE_INDICES]
        nose = face[0].astype(np.float32)
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        body_scale = np.linalg.norm(left_shoulder - right_shoulder)
        if body_scale <= 1e-6:
            body_scale = 1.0

        face_scale = np.linalg.norm(face[1] - face[2])
        if face_scale <= 1e-6:
            face_scale = 1.0
        face_rel = ((face[1:] - face[0:1]) / face_scale).astype(np.float32)

        left_base, left_palm = _local_branch_base(lh, 'Left', body_scale)
        right_base, right_palm = _local_branch_base(rh, 'Right', body_scale)

        wrist_global = np.concatenate([lh[0], rh[0]]).astype(np.float32)
        inter_wrist = np.array([np.linalg.norm(lh[0] - rh[0]) / body_scale], dtype=np.float32)
        wrist_to_face = np.array([
            np.linalg.norm(lh[0] - face[0]) / body_scale,
            np.linalg.norm(rh[0] - face[0]) / body_scale,
            np.linalg.norm(lh[0] - face[3]) / body_scale,
            np.linalg.norm(rh[0] - face[4]) / body_scale,
        ], dtype=np.float32)
        global_base = np.concatenate([face_rel.flatten(), wrist_global, inter_wrist, wrist_to_face]).astype(np.float32)

        frames.append((left_base, right_base, global_base))
        left_palms.append(left_palm)
        right_palms.append(right_palm)
        left_wrists.append(lh[0].astype(np.float32))
        right_wrists.append(rh[0].astype(np.float32))
        face_noses.append(nose)
        left_face_dirs.append((nose - lh[0]).astype(np.float32))
        right_face_dirs.append((nose - rh[0]).astype(np.float32))

    left_palms = np.asarray(left_palms, dtype=np.float32)
    right_palms = np.asarray(right_palms, dtype=np.float32)
    left_wrists = np.asarray(left_wrists, dtype=np.float32)
    right_wrists = np.asarray(right_wrists, dtype=np.float32)
    left_face_dirs = np.asarray(left_face_dirs, dtype=np.float32)
    right_face_dirs = np.asarray(right_face_dirs, dtype=np.float32)
    cam = _camera_axis()

    left_out = []
    right_out = []
    global_out = []
    total = len(frames)

    for t in range(total):
        next_t = min(t + 1, total - 1)
        left_normal = left_palms[t]
        right_normal = right_palms[t]
        left_next = left_palms[next_t]
        right_next = right_palms[next_t]
        left_wrist_vel = _safe_difference(left_wrists[next_t], left_wrists[t])
        right_wrist_vel = _safe_difference(right_wrists[next_t], right_wrists[t])

        left_dyn = np.concatenate([
            left_normal,
            -left_normal,
            _safe_difference(left_next, left_normal),
            np.array([1.0 - float(np.dot(left_normal, left_next))], dtype=np.float32),
            np.cross(left_normal, left_next).astype(np.float32),
            np.array([_exposure(left_normal, left_face_dirs[t])], dtype=np.float32),
            np.array([_exposure(left_normal, cam)], dtype=np.float32),
            np.array([_exposure(left_normal, left_wrist_vel)], dtype=np.float32),
        ]).astype(np.float32)

        right_dyn = np.concatenate([
            right_normal,
            -right_normal,
            _safe_difference(right_next, right_normal),
            np.array([1.0 - float(np.dot(right_normal, right_next))], dtype=np.float32),
            np.cross(right_normal, right_next).astype(np.float32),
            np.array([_exposure(right_normal, right_face_dirs[t])], dtype=np.float32),
            np.array([_exposure(right_normal, cam)], dtype=np.float32),
            np.array([_exposure(right_normal, right_wrist_vel)], dtype=np.float32),
        ]).astype(np.float32)

        symmetry = np.array([_exposure(left_normal, right_normal)], dtype=np.float32)

        left_base, right_base, global_base = frames[t]
        left_out.append(np.concatenate([left_base, left_dyn]).astype(np.float32))
        right_out.append(np.concatenate([right_base, right_dyn]).astype(np.float32))
        global_out.append(np.concatenate([global_base, symmetry]).astype(np.float32))

    return (
        np.asarray(left_out, dtype=np.float32),
        np.asarray(right_out, dtype=np.float32),
        np.asarray(global_out, dtype=np.float32),
    )


def extract_exp13_full_sequence_features(seq, rectify_alpha=0.4):
    cfg = GeometryConfig(name='exp13_full_sequence', rectification=True, rectify_alpha=rectify_alpha)
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected sequence of shape [T, D], got {seq.shape}")

    left_out = []
    right_out = []
    global_out = []
    orientation_out = []

    prev_left_wrist = None
    prev_right_wrist = None
    prev_left_center = None
    prev_right_center = None
    prev_left_wrist_vel = None
    prev_right_wrist_vel = None
    prev_left_palm_vel = None
    prev_right_palm_vel = None
    prev_interaction_rel = None
    prev_palm_distance = None

    for vec in seq:
        pose, lh, rh = _prepare_parts(vec, cfg)
        face = pose[FACE_INDICES]
        face_center = face.mean(axis=0).astype(np.float32)
        face_scale = np.linalg.norm(face[1] - face[2])
        if face_scale <= 1e-6:
            face_scale = 1.0

        left_shoulder = pose[11]
        right_shoulder = pose[12]
        body_scale = np.linalg.norm(left_shoulder - right_shoulder)
        if body_scale <= 1e-6:
            body_scale = 1.0

        pose_vis = vec[0:132].reshape(33, 4)[:, 3].astype(np.float32)
        face_visibility = np.clip(pose_vis[FACE_INDICES], 0.0, 1.0)
        left_wrist_visibility = np.clip(pose_vis[15], 0.0, 1.0)
        right_wrist_visibility = np.clip(pose_vis[16], 0.0, 1.0)

        def hand_features(hand_xyz, hand_label, prev_wrist, prev_center, other_wrist):
            wrist = hand_xyz[0].astype(np.float32)
            palm_center = hand_xyz[[0, 5, 9, 13, 17]].mean(axis=0).astype(np.float32)
            palm_scale = np.linalg.norm(hand_xyz[0] - hand_xyz[9])
            valid = float(np.any(np.abs(hand_xyz) > 1e-8))
            if palm_scale <= 1e-6:
                palm_scale = 1.0

            shape = ((hand_xyz[1:] - wrist[None, :]) / palm_scale).astype(np.float32).reshape(-1)
            wrist_traj = ((wrist - face_center) / face_scale).astype(np.float32)
            palm_traj = ((palm_center - face_center) / face_scale).astype(np.float32)
            wrist_vel = np.zeros(3, dtype=np.float32) if prev_wrist is None else ((wrist - prev_wrist) / face_scale).astype(np.float32)
            palm_vel = np.zeros(3, dtype=np.float32) if prev_center is None else ((palm_center - prev_center) / face_scale).astype(np.float32)
            palm_normal = compute_palm_normal(hand_xyz, hand_label) if valid else np.zeros(3, dtype=np.float32)
            raw_screen = np.array([wrist[0], wrist[1], palm_center[0], palm_center[1]], dtype=np.float32)
            side_flag = np.array([float(wrist[0] >= face_center[0])], dtype=np.float32)
            body_region = np.array(
                [
                    (wrist[0] - face_center[0]) / body_scale,
                    (wrist[1] - face_center[1]) / body_scale,
                    (wrist[2] - face_center[2]) / body_scale,
                ],
                dtype=np.float32,
            )
            other_rel = ((wrist - other_wrist) / face_scale).astype(np.float32)
            features = np.concatenate(
                [
                    shape,
                    wrist_traj,
                    palm_traj,
                    wrist_vel,
                    palm_vel,
                    raw_screen,
                    side_flag,
                    body_region,
                    other_rel,
                    np.array([valid, palm_scale / body_scale], dtype=np.float32),
                ]
            ).astype(np.float32)
            return features, wrist, palm_center, palm_normal, valid, wrist_vel, palm_vel

        left_feat, left_wrist, left_center, left_normal, left_valid, left_wrist_vel, left_palm_vel = hand_features(
            lh, "Left", prev_left_wrist, prev_left_center, rh[0].astype(np.float32)
        )
        right_feat, right_wrist, right_center, right_normal, right_valid, right_wrist_vel, right_palm_vel = hand_features(
            rh, "Right", prev_right_wrist, prev_right_center, lh[0].astype(np.float32)
        )

        left_wrist_acc = (
            np.zeros(3, dtype=np.float32)
            if prev_left_wrist_vel is None
            else (left_wrist_vel - prev_left_wrist_vel).astype(np.float32)
        )
        right_wrist_acc = (
            np.zeros(3, dtype=np.float32)
            if prev_right_wrist_vel is None
            else (right_wrist_vel - prev_right_wrist_vel).astype(np.float32)
        )
        left_palm_acc = (
            np.zeros(3, dtype=np.float32)
            if prev_left_palm_vel is None
            else (left_palm_vel - prev_left_palm_vel).astype(np.float32)
        )
        right_palm_acc = (
            np.zeros(3, dtype=np.float32)
            if prev_right_palm_vel is None
            else (right_palm_vel - prev_right_palm_vel).astype(np.float32)
        )
        left_feat = np.concatenate([left_feat, left_wrist_acc, left_palm_acc]).astype(np.float32)
        right_feat = np.concatenate([right_feat, right_wrist_acc, right_palm_acc]).astype(np.float32)

        interaction_rel = ((left_wrist - right_wrist) / face_scale).astype(np.float32)
        interaction_center = ((left_center - right_center) / face_scale).astype(np.float32)
        palm_distance = np.array([np.linalg.norm(left_center - right_center) / face_scale], dtype=np.float32)
        relative_velocity = (
            np.zeros(3, dtype=np.float32)
            if prev_left_wrist is None or prev_right_wrist is None
            else (((left_wrist - prev_left_wrist) - (right_wrist - prev_right_wrist)) / face_scale).astype(np.float32)
        )
        interaction_velocity = (
            np.zeros(3, dtype=np.float32)
            if prev_interaction_rel is None
            else (interaction_rel - prev_interaction_rel).astype(np.float32)
        )
        palm_distance_velocity = (
            np.zeros(1, dtype=np.float32)
            if prev_palm_distance is None
            else (palm_distance - prev_palm_distance).astype(np.float32)
        )
        crossing = np.array([float(left_wrist[0] > right_wrist[0])], dtype=np.float32)
        face_rel = ((face - face_center[None, :]) / face_scale).astype(np.float32).reshape(-1)
        validity_flags = np.array(
            [
                left_valid,
                right_valid,
                float(np.any(np.abs(face) > 1e-8)),
                float(np.linalg.norm(left_normal) > 1e-8),
                float(np.linalg.norm(right_normal) > 1e-8),
                left_wrist_visibility,
                right_wrist_visibility,
                float(face_visibility.mean()) if face_visibility.size else 0.0,
            ],
            dtype=np.float32,
        )

        global_features = np.concatenate(
            [
                interaction_rel,
                interaction_center,
                palm_distance,
                relative_velocity,
                interaction_velocity,
                palm_distance_velocity,
                crossing,
                face_rel,
                validity_flags,
            ]
        ).astype(np.float32)
        orientation_features = np.concatenate([left_normal, right_normal], axis=0).astype(np.float32)

        left_out.append(left_feat)
        right_out.append(right_feat)
        global_out.append(global_features)
        orientation_out.append(orientation_features)

        prev_left_wrist = left_wrist
        prev_right_wrist = right_wrist
        prev_left_center = left_center
        prev_right_center = right_center
        prev_left_wrist_vel = left_wrist_vel
        prev_right_wrist_vel = right_wrist_vel
        prev_left_palm_vel = left_palm_vel
        prev_right_palm_vel = right_palm_vel
        prev_interaction_rel = interaction_rel
        prev_palm_distance = palm_distance

    return (
        np.asarray(left_out, dtype=np.float32),
        np.asarray(right_out, dtype=np.float32),
        np.asarray(global_out, dtype=np.float32),
        np.asarray(orientation_out, dtype=np.float32),
    )
