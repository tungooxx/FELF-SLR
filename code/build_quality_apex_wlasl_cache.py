"""Build a resumable WLASL quality/apex 64-to-40 dataset cache."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from missing_data_masks import mask_from_detection
from tool import (
    PoseNormalizer,
    calculate_hand_rotation,
    calculate_palm_normal,
    classify_hand_view,
    determine_hand_shape,
    extract_keypoints,
    one_hot_finger,
)


CANDIDATES = 64
SELECTED = 40
FEATURE_DIM = 457


def frame_features(frame: np.ndarray, holistic) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose_present = result.pose_landmarks is not None
    left_present = result.left_hand_landmarks is not None
    right_present = result.right_hand_landmarks is not None
    mask = mask_from_detection(
        pose_present=pose_present,
        left_hand_present=left_present,
        right_hand_present=right_present,
        frame_valid=True,
    )
    raw = np.zeros(225, dtype=np.float32)
    if not pose_present:
        return np.zeros(FEATURE_DIM, dtype=np.float32), mask, raw
    pose = np.asarray([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark], dtype=np.float32)
    left = np.asarray([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark], dtype=np.float32) if left_present else None
    right = np.asarray([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark], dtype=np.float32) if right_present else None
    raw[:99] = pose[:, :3].reshape(-1)
    if left_present:
        raw[99:162] = left.reshape(-1)
    if right_present:
        raw[162:225] = right.reshape(-1)
    pose, left, right = PoseNormalizer.normalize_holistic(pose, left, right)
    kp = extract_keypoints(pose, left, right)
    lho, rho = [0] * 5, [0] * 5
    lr, rr = [0] * 8, [0] * 8
    ls, rs = [0] * 7, [0] * 7
    lf, rf = [0] * 40, [0] * 40
    if left is not None:
        lho = classify_hand_view(calculate_palm_normal(left, "Left"), "Left")
        lr, ls, lf = calculate_hand_rotation(left), determine_hand_shape(left), one_hot_finger(left)
    if right is not None:
        rho = classify_hand_view(calculate_palm_normal(right, "Right"), "Right")
        rr, rs, rf = calculate_hand_rotation(right), determine_hand_shape(right), one_hot_finger(right)
    return np.concatenate([kp, lho, rho, lr, rr, ls, rs, lf, rf]).astype(np.float32), mask, raw


def quality_scores(raw: np.ndarray, masks: np.ndarray) -> np.ndarray:
    left = raw[:, 99:162].reshape(CANDIDATES, 21, 3)
    right = raw[:, 162:225].reshape(CANDIDATES, 21, 3)
    lv, rv = masks[:, 0] > 0.5, masks[:, 1] > 0.5
    motion = np.zeros(CANDIDATES, dtype=np.float32)
    for hand, valid in ((left, lv), (right, rv)):
        center = hand.mean(axis=1)
        delta = np.linalg.norm(center[1:] - center[:-1], axis=-1)
        motion[1:] += np.where(valid[1:] & valid[:-1], delta, 0.0)
    if motion.max() > 0:
        motion /= motion.max()
    middle = np.exp(-(np.linspace(-1, 1, CANDIDATES) / 0.55) ** 2)
    visibility = np.maximum(lv, rv).astype(np.float32) + 0.5 * (lv & rv).astype(np.float32)
    return 1.5 * visibility + 0.75 * motion + 0.35 * middle


def choose_indices(raw: np.ndarray, masks: np.ndarray, selector: str = "quality_apex") -> np.ndarray:
    score = quality_scores(raw, masks)
    if selector == "coverage_20_60_20":
        # Exact 8/24/8 allocation. Half of each region is uniformly anchored;
        # remaining frames are selected by quality score within that region.
        selected: set[int] = set()
        for start, stop, quota in ((0, 16, 8), (16, 48, 24), (48, 64, 8)):
            anchor_count = quota // 2
            anchors = set(np.round(np.linspace(start, stop - 1, anchor_count)).astype(int).tolist())
            ranked = [int(i) for i in np.argsort(-score[start:stop]) + start if int(i) not in anchors]
            selected.update(anchors)
            selected.update(ranked[: quota - len(anchors)])
        return np.asarray(sorted(selected), dtype=np.int64)
    if selector != "quality_apex":
        raise ValueError(f"Unknown selector: {selector}")
    anchors = set(np.round(np.linspace(0, CANDIDATES - 1, 20)).astype(int).tolist())
    ranked = [int(i) for i in np.argsort(-score) if int(i) not in anchors]
    return np.asarray(sorted(anchors | set(ranked[: SELECTED - len(anchors)])), dtype=np.int64)


def process_record(task: tuple[dict, str, str, str], holistic=None) -> dict:
    record, video_dir, output_dir, selector = task
    target = Path(output_dir) / record["split"] / record["gloss"] / record["video_id"]
    if (target / "validity_masks.npy").exists() and all((target / f"{i}.npy").exists() for i in range(SELECTED)):
        return {"status": "cached", "video_id": record["video_id"]}
    cap = cv2.VideoCapture(str(Path(video_dir) / f"{record['video_id']}.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return {"status": "failed", "video_id": record["video_id"]}
    start = max(0, int(record["frame_start"]))
    end = int(record["frame_end"])
    if end < 0 or end >= total:
        end = total - 1
    source_indices = np.round(np.linspace(start, end, CANDIDATES)).astype(np.int64)
    features, masks, raw = [], [], []
    owns_holistic = holistic is None
    if owns_holistic:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    try:
        for source_index in source_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
            ok, frame = cap.read()
            if ok and frame is not None:
                feat, mask, raw_frame = frame_features(frame, holistic)
            else:
                feat = np.zeros(FEATURE_DIM, dtype=np.float32)
                mask = mask_from_detection(pose_present=False, left_hand_present=False, right_hand_present=False, frame_valid=False)
                raw_frame = np.zeros(225, dtype=np.float32)
            features.append(feat)
            masks.append(mask)
            raw.append(raw_frame)
    finally:
        if owns_holistic:
            holistic.close()
    cap.release()
    features, masks, raw = np.asarray(features), np.asarray(masks), np.asarray(raw)
    chosen = choose_indices(raw, masks, selector)
    target.mkdir(parents=True, exist_ok=True)
    for index, feature in enumerate(features[chosen]):
        np.save(target / f"{index}.npy", feature)
    np.save(target / "frame_indices.npy", source_indices[chosen])
    np.save(target / "candidate_indices.npy", source_indices)
    np.save(target / "validity_masks.npy", masks[chosen])
    return {"status": "written", "video_id": record["video_id"]}


def process_chunk(tasks: list[tuple[dict, str, str, str]]) -> dict[str, int]:
    counts = {"written": 0, "cached": 0, "failed": 0}
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        for index, task in enumerate(tasks, start=1):
            result = process_record(task, holistic)
            counts[result["status"]] += 1
            if index % 50 == 0 or index == len(tasks):
                print(f"worker progress {index}/{len(tasks)} {counts}", flush=True)
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-glosses", type=int, required=True)
    p.add_argument("--json-path", default="WLASL_Full/WLASL_v0.3.json")
    p.add_argument("--video-dir", default="WLASL_Complete/WLASL2000")
    p.add_argument(
        "--reference-data-dir",
        default=None,
        help="When set, build only the exact split/gloss/video IDs present in this frozen dataset.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--selector", choices=["quality_apex", "coverage_20_60_20"], default="quality_apex")
    args = p.parse_args()
    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    records = []
    selected = data[: args.num_glosses]
    if args.reference_data_dir:
        by_id = {
            str(instance["video_id"]): {"gloss": entry["gloss"], **instance}
            for entry in selected
            for instance in entry["instances"]
        }
        selected_glosses = sorted(entry["gloss"] for entry in selected)
        reference = Path(args.reference_data_dir)
        for split in ("train", "val", "test"):
            for gloss in selected_glosses:
                gloss_dir = reference / split / gloss
                if not gloss_dir.exists():
                    continue
                for video_dir in sorted(path for path in gloss_dir.iterdir() if path.is_dir()):
                    record = dict(by_id[video_dir.name])
                    record["split"] = split
                    records.append(record)
    else:
        for entry in selected:
            for instance in entry["instances"]:
                records.append({"gloss": entry["gloss"], **instance})
    tasks = [(record, args.video_dir, args.output_dir, args.selector) for record in records]
    counts = {"written": 0, "cached": 0, "failed": 0}
    chunks = [tasks[index:: args.workers] for index in range(args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_chunk, chunk) for chunk in chunks if chunk]
        with tqdm(total=len(tasks), desc=f"Quality-apex WLASL-{args.num_glosses}", mininterval=5) as progress:
            for future in as_completed(futures):
                result = future.result()
                for key, value in result.items():
                    counts[key] += value
                progress.update(sum(result.values()))
    metadata = {
        "protocol": f"Sampling-v1 {args.selector} 64-to-40",
        "num_glosses": args.num_glosses,
        "candidate_frames": CANDIDATES,
        "selected_frames": SELECTED,
        "score": "1.5 visibility + 0.75 normalized hand motion + 0.35 middle bias; 20 uniform anchors",
        "selector": args.selector,
        "mediapipe": {"model_complexity": 1, "det": 0.3, "track": 0.3, "static_image_mode": False},
        "counts": counts,
        "reference_data_dir": args.reference_data_dir,
        "requested_records": len(records),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "sampling_v1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
