"""Build the pair dataset for ReID fine-tuning.

For each training node (default: 2, 3, 5, 8 — GT-perfect nodes):
  1. Load GT instance labels via evaluate.load_gt_for_image()
  2. Match each detection to its GT instance via IoU
  3. Extract and save crops (bbox + PATCH_EXPAND_RATIO context)
  4. Mine positive pairs  (same GT instance_id, different images)
     and negative pairs   (different GT instance_id, same code, different images)
  5. Save balanced pairs to pairs.json

Run standalone:
    python reid_finetuned_resnet50/dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import IMAGES_DIR  # noqa: E402 — after sys.path
from evaluate import (  # noqa: E402
    IOU_THRESH,
    load_gt_for_image,
    load_detections_for_image,
    match_detections_to_gt,
)
from pipeline.node_reid import PATCH_EXPAND_RATIO, MIN_PATCH_SIZE  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
TRAIN_NODES    = ["2", "3", "5", "8"]
CROPS_DIR      = Path(__file__).parent / "crops"
PAIRS_FILE     = Path(__file__).parent / "pairs.json"
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)


# ── patch extraction ──────────────────────────────────────────────────────────

def _expand_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple:
    pad_x = max(int((x2 - x1) * PATCH_EXPAND_RATIO), 0)
    pad_y = max(int((y2 - y1) * PATCH_EXPAND_RATIO), 0)
    ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex2, ey2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if ex2 - ex1 < MIN_PATCH_SIZE:
        cx = (ex1 + ex2) // 2
        ex1, ex2 = max(0, cx - MIN_PATCH_SIZE // 2), min(w, cx + MIN_PATCH_SIZE // 2)
    if ey2 - ey1 < MIN_PATCH_SIZE:
        cy = (ey1 + ey2) // 2
        ey1, ey2 = max(0, cy - MIN_PATCH_SIZE // 2), min(h, cy + MIN_PATCH_SIZE // 2)
    return ex1, ey1, ex2, ey2


def extract_and_save_crop(
    img_bgr: np.ndarray,
    bbox_norm: list[float],
    out_path: Path,
) -> bool:
    """Crop bbox (+ expand) from img_bgr and save to out_path. Returns success."""
    h, w = img_bgr.shape[:2]
    x1n, y1n, x2n, y2n = bbox_norm
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, w, h)
    patch = img_bgr[ey1:ey2, ex1:ex2]
    if patch.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), patch)
    return True


# ── main dataset builder ──────────────────────────────────────────────────────

def build_dataset(train_nodes: list[str] = TRAIN_NODES) -> dict:
    """
    Returns stats dict and writes:
      - crops/<node>/<image_stem>_<det_idx>.jpg
      - pairs.json
    """
    # obs_id → {gt_instance_id, code, crop_path}
    obs_meta: dict[str, dict] = {}

    print("=== Extracting crops ===")
    for node_id in train_nodes:
        node_crops_dir = CROPS_DIR / f"node{node_id}"
        img_paths = {p.stem: p for p in sorted(IMAGES_DIR.glob(f"node{node_id}_*.jpg"))}

        for stem, img_path in img_paths.items():
            gt_instances = load_gt_for_image(stem)
            detections   = load_detections_for_image(stem)
            if not gt_instances or not detections:
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            for det_idx, det in enumerate(detections):
                obs_id = f"{stem}_{det_idx}"
                gt_iid, iou = match_detections_to_gt(
                    det["bbox_norm"], det["code"], gt_instances
                )
                if gt_iid is None or iou < IOU_THRESH:
                    continue  # unmatched detection

                crop_path = node_crops_dir / f"{stem}_{det_idx}.jpg"
                ok = extract_and_save_crop(img_bgr, det["bbox_norm"], crop_path)
                if not ok:
                    continue

                obs_meta[obs_id] = {
                    "gt_instance_id": gt_iid,
                    "code":           det["code"],
                    "image_stem":     stem,
                    "node_id":        node_id,
                    "crop_path":      str(crop_path.relative_to(ROOT)),
                }

        n_crops = sum(1 for v in obs_meta.values() if v["node_id"] == node_id)
        print(f"  node{node_id}: {n_crops} crops")

    print(f"\nTotal observations with GT match: {len(obs_meta)}")

    # ── pair mining ───────────────────────────────────────────────────────────
    # Group by (node_id, code, gt_instance_id)
    from collections import defaultdict
    groups: dict[tuple, list[str]] = defaultdict(list)
    for obs_id, meta in obs_meta.items():
        key = (meta["node_id"], meta["code"], meta["gt_instance_id"])
        groups[key].append(obs_id)

    positives: list[dict] = []
    negatives: list[dict] = []

    for (node_id, code, gt_iid), obs_ids in groups.items():
        # Positive pairs: same instance, different images
        for a, b in combinations(obs_ids, 2):
            if obs_meta[a]["image_stem"] != obs_meta[b]["image_stem"]:
                positives.append({
                    "img_a": obs_meta[a]["crop_path"],
                    "img_b": obs_meta[b]["crop_path"],
                    "label": 1,
                    "code":  code,
                    "node":  node_id,
                })

        # Hard negative pairs: same code, same node, different instance, different images
        same_code_groups = [
            (gt_iid2, ids2)
            for (n2, c2, gt_iid2), ids2 in groups.items()
            if n2 == node_id and c2 == code and gt_iid2 != gt_iid
        ]
        for a in obs_ids:
            for _, neg_ids in same_code_groups:
                for b in neg_ids:
                    if obs_meta[a]["image_stem"] != obs_meta[b]["image_stem"]:
                        negatives.append({
                            "img_a": obs_meta[a]["crop_path"],
                            "img_b": obs_meta[b]["crop_path"],
                            "label": 0,
                            "code":  code,
                            "node":  node_id,
                        })

    # Balance
    random.shuffle(positives)
    random.shuffle(negatives)
    n = min(len(positives), len(negatives))
    pairs = positives[:n] + negatives[:n]
    random.shuffle(pairs)

    PAIRS_FILE.write_text(json.dumps(pairs, indent=2))
    print(f"\nPositive pairs : {len(positives)}  (using {n})")
    print(f"Negative pairs : {len(negatives)}  (using {n})")
    print(f"Total pairs    : {len(pairs)}")
    print(f"Saved → {PAIRS_FILE}")

    return {"n_obs": len(obs_meta), "n_pos": n, "n_neg": n, "n_pairs": len(pairs)}


if __name__ == "__main__":
    build_dataset()
