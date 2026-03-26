"""Diagnose false-positive ReID clusters and explain why they were not merged.

Usage:
    python analyze_reid_false_positives.py        # all nodes with outputs
    python analyze_reid_false_positives.py 4 7 9  # specific nodes
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import COMPONENT_INFO, CONF_THRESHOLD, DIAMETER_CODES, IMAGES_DIR, LABELS_EXT_DIR, WORK_DIR
from core.linear_assignment import linear_sum_assignment

MULTIVIEW_DIR = WORK_DIR / "multiview"
CACHE_DIR = WORK_DIR / "detection_cache"
IOU_THRESH = 0.30
W = 88


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _iou(b1: list[float], b2: list[float]) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _obs_image(obs_id: str) -> str:
    return obs_id.rsplit("_", 1)[0]


def _obs_index(obs_id: str) -> int:
    return int(obs_id.rsplit("_", 1)[1])


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_scene_nodes(node_id: int) -> tuple[list[dict], str]:
    raw_path = MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes_reid_raw.json"
    final_path = MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8")), "raw_reid"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8")), "final_scene_nodes"
    return [], "missing"


def load_reid_debug(node_id: int) -> dict | None:
    path = MULTIVIEW_DIR / f"node{node_id}" / "reid_debug.json"
    return _load_json(path)


def load_final_scene_nodes(node_id: int) -> list[dict]:
    path = MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_detections_for_image(image_stem: str) -> list[dict]:
    path = CACHE_DIR / f"{image_stem}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("detections", [])
    return [
        det for det in raw
        if det.get("confidence", 0.0) >= CONF_THRESHOLD
        and not det.get("is_diameter", det.get("code", -1) in DIAMETER_CODES)
    ]


def load_gt_for_node(node_id: int) -> dict:
    instances_by_id: dict[int, dict] = {}
    instances_by_image: dict[str, list[dict]] = {}
    image_stems = sorted(path.stem for path in IMAGES_DIR.glob(f"node{node_id}_*.jpg"))

    for stem in image_stems:
        label_path = LABELS_EXT_DIR / f"{stem}_extended.txt"
        items = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                code = int(parts[6])
                if code in DIAMETER_CODES:
                    continue
                instance_id = int(parts[5])
                items.append({
                    "instance_id": instance_id,
                    "code": code,
                    "bbox_norm": _xywh_to_xyxy(
                        float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    ),
                    "image_stem": stem,
                })
                if instance_id not in instances_by_id:
                    _, name = COMPONENT_INFO.get(code, (0, f"code_{code}"))
                    instances_by_id[instance_id] = {
                        "instance_id": instance_id,
                        "code": code,
                        "name": name,
                        "images": [],
                    }
                instances_by_id[instance_id]["images"].append(stem)
        instances_by_image[stem] = items

    return {
        "image_stems": image_stems,
        "instances_by_id": instances_by_id,
        "instances_by_image": instances_by_image,
    }


def match_detection_to_gt(det: dict, gt_instances: list[dict]) -> tuple[int | None, float]:
    best_iid = None
    best_iou = 0.0
    for gt in gt_instances:
        if gt["code"] != det["code"]:
            continue
        iou = _iou(det["bbox_norm"], gt["bbox_norm"])
        if iou > best_iou:
            best_iou = iou
            best_iid = gt["instance_id"]
    if best_iou < IOU_THRESH:
        return None, 0.0
    return best_iid, best_iou


def analyze_scene_node(scene_node: dict, gt_data: dict) -> dict:
    votes: Counter = Counter()
    obs_matches = []

    for obs_id in scene_node["supporting_observed_ids"]:
        image_stem = _obs_image(obs_id)
        det_idx = _obs_index(obs_id)
        dets = load_detections_for_image(image_stem)
        det = dets[det_idx] if det_idx < len(dets) else None
        if det is None:
            obs_matches.append({"obs_id": obs_id, "image_id": image_stem, "gt_instance_id": None, "iou": 0.0})
            continue
        gt_instance_id, iou = match_detection_to_gt(det, gt_data["instances_by_image"].get(image_stem, []))
        if gt_instance_id is not None:
            votes[gt_instance_id] += 1
        obs_matches.append({
            "obs_id": obs_id,
            "image_id": image_stem,
            "gt_instance_id": gt_instance_id,
            "iou": iou,
        })

    matched = sum(votes.values())
    majority_gt_id = votes.most_common(1)[0][0] if votes else None
    majority_count = votes[majority_gt_id] if majority_gt_id is not None else 0
    obs_per_image: dict[str, list[str]] = defaultdict(list)
    for obs_id in scene_node["supporting_observed_ids"]:
        obs_per_image[_obs_image(obs_id)].append(obs_id)

    return {
        "scene_node_id": scene_node["scene_node_id"],
        "votes": dict(votes),
        "majority_gt_id": majority_gt_id,
        "majority_count": majority_count,
        "n_matched": matched,
        "purity": (majority_count / matched) if matched else None,
        "obs_matches": obs_matches,
        "obs_per_image": {img: sorted(ids) for img, ids in sorted(obs_per_image.items())},
    }


def build_bijective_mapping(scene_nodes: list[dict], analyses: dict[str, dict], gt_data: dict) -> dict[str, int]:
    by_code: dict[int, list[dict]] = defaultdict(list)
    gt_by_code: dict[int, list[int]] = defaultdict(list)

    for scene_node in scene_nodes:
        by_code[scene_node["fused_code"]].append(scene_node)
    for instance_id, gt in gt_data["instances_by_id"].items():
        gt_by_code[gt["code"]].append(instance_id)

    mapping: dict[str, int] = {}
    for code, pred_nodes in by_code.items():
        gt_ids = sorted(gt_by_code.get(code, []))
        if not gt_ids:
            continue
        cost = np.zeros((len(pred_nodes), len(gt_ids)), dtype=float)
        for row, scene_node in enumerate(pred_nodes):
            votes = analyses[scene_node["scene_node_id"]]["votes"]
            for col, gt_id in enumerate(gt_ids):
                cost[row, col] = votes.get(gt_id, 0)
        row_ind, col_ind = linear_sum_assignment(-cost)
        for row, col in zip(row_ind, col_ind):
            if cost[row, col] > 0:
                mapping[pred_nodes[row]["scene_node_id"]] = gt_ids[col]
    return mapping


def find_cross_group_entry(reid_debug: dict | None, group_a: str, group_b: str) -> dict | None:
    if not reid_debug:
        return None
    for entry in reid_debug.get("cross_group_analysis", []):
        if {entry["group_a"], entry["group_b"]} == {group_a, group_b}:
            return entry
    return None


def explain_false_positive(
    scene_node: dict,
    analysis: dict,
    scene_node_by_id: dict[str, dict],
    analyses: dict[str, dict],
    sn_to_gt: dict[str, int],
    reid_debug: dict | None,
) -> dict:
    target_id = None
    majority_gt_id = analysis["majority_gt_id"]
    if majority_gt_id is not None:
        for scene_node_id, gt_id in sn_to_gt.items():
            if gt_id == majority_gt_id and scene_node_id != scene_node["scene_node_id"]:
                target_id = scene_node_id
                break

    if target_id is None:
        same_code = [
            other for other in scene_node_by_id.values()
            if other["fused_code"] == scene_node["fused_code"]
            and other["scene_node_id"] != scene_node["scene_node_id"]
        ]
        if same_code:
            target_id = max(
                same_code,
                key=lambda item: (
                    analyses[item["scene_node_id"]]["majority_count"],
                    item["support_count"],
                    item["scene_node_id"],
                ),
            )["scene_node_id"]

    target = scene_node_by_id.get(target_id) if target_id else None
    target_analysis = analyses.get(target_id) if target_id else None
    cross_entry = find_cross_group_entry(reid_debug, scene_node["scene_node_id"], target_id) if target_id else None

    root_causes = []
    if analysis["support_count"] > len(scene_node["support_images"]):
        root_causes.append("il cluster viola già il vincolo una sola osservazione per immagine")

    duplicate_gt_images = defaultdict(list)
    for obs in analysis["obs_matches"]:
        if obs["gt_instance_id"] is None:
            continue
        duplicate_gt_images[(obs["image_id"], obs["gt_instance_id"])].append(obs["obs_id"])
    same_image_gt_dups = [
        {"image_id": image_id, "gt_instance_id": gt_id, "observed_ids": sorted(obs_ids)}
        for (image_id, gt_id), obs_ids in sorted(duplicate_gt_images.items())
        if len(obs_ids) > 1
    ]
    if same_image_gt_dups:
        root_causes.append("nello stesso cluster ci sono più detection che matchano lo stesso GT nella stessa immagine")

    if cross_entry:
        merged_dups = cross_entry.get("merged_duplicate_images", [])
        if merged_dups:
            root_causes.append(
                "fondere col cluster candidato violerebbe il vincolo una sola osservazione per immagine"
            )
        cross_summary = cross_entry.get("cross_pair_summary", {})
        if cross_summary.get("n_effective_affinity", 0) == 0:
            root_causes.append("non esiste evidenza di affinità sufficiente tra i due cluster")
        elif cross_summary.get("mean_ratio", 0.0) < reid_debug["thresholds"]["final_group_min_affinity"]:
            root_causes.append("l'affinità media tra i due cluster resta sotto la soglia di grouping")
        soft = cross_entry.get("soft_merge", {})
        if soft.get("blocked_by_image_overlap"):
            root_causes.append("il soft-merge è bloccato perché i due cluster condividono almeno un'immagine")
        else:
            if not soft.get("pass_proto", True):
                root_causes.append("il soft-merge fallisce per similarità prototipale troppo bassa")
            if not soft.get("pass_cross_coassoc", True):
                root_causes.append("il soft-merge fallisce per co-associazione cross-cluster troppo bassa")
            if not soft.get("pass_score", True):
                root_causes.append("il punteggio combinato del soft-merge resta sotto soglia")

    return {
        "scene_node_id": scene_node["scene_node_id"],
        "code": scene_node["fused_code"],
        "name": scene_node["fused_name"],
        "support_count": scene_node["support_count"],
        "support_images": list(scene_node["support_images"]),
        "majority_gt_id": majority_gt_id,
        "majority_count": analysis["majority_count"],
        "purity": analysis["purity"],
        "votes": analysis["votes"],
        "obs_per_image": analysis["obs_per_image"],
        "same_image_gt_duplicates": same_image_gt_dups,
        "target_scene_node_id": target_id,
        "target_support_count": target["support_count"] if target else None,
        "target_support_images": list(target["support_images"]) if target else [],
        "target_majority_gt_id": target_analysis["majority_gt_id"] if target_analysis else None,
        "cross_group_analysis": cross_entry,
        "root_causes": root_causes,
    }


def analyze_node(node_id: int) -> dict | None:
    scene_nodes, scene_source = load_scene_nodes(node_id)
    if not scene_nodes:
        print(f"[node {node_id}] nessun scene node disponibile")
        return None

    gt_data = load_gt_for_node(node_id)
    reid_debug = load_reid_debug(node_id)
    analyses = {scene_node["scene_node_id"]: analyze_scene_node(scene_node, gt_data) for scene_node in scene_nodes}
    for scene_node in scene_nodes:
        analyses[scene_node["scene_node_id"]]["support_count"] = scene_node["support_count"]

    sn_to_gt = build_bijective_mapping(scene_nodes, analyses, gt_data)
    scene_node_by_id = {scene_node["scene_node_id"]: scene_node for scene_node in scene_nodes}
    fp_scene_nodes = [scene_node for scene_node in scene_nodes if scene_node["scene_node_id"] not in sn_to_gt]
    fp_reports = [
        explain_false_positive(
            scene_node,
            analyses[scene_node["scene_node_id"]],
            scene_node_by_id,
            analyses,
            sn_to_gt,
            reid_debug,
        )
        for scene_node in sorted(fp_scene_nodes, key=lambda item: (item["fused_code"], item["scene_node_id"]))
    ]

    final_scene_nodes = load_final_scene_nodes(node_id)
    report = {
        "node_id": node_id,
        "scene_source": scene_source,
        "n_scene_nodes": len(scene_nodes),
        "n_final_scene_nodes": len(final_scene_nodes),
        "n_gt_instances": len(gt_data["instances_by_id"]),
        "n_false_positive_scene_nodes": len(fp_reports),
        "reid_debug_available": reid_debug is not None,
        "observation_budget_violations": reid_debug.get("observation_budget_violations", []) if reid_debug else [],
        "false_positive_scene_nodes": fp_reports,
    }

    out_path = MULTIVIEW_DIR / f"node{node_id}" / "reid_false_positive_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * W}")
    print(f"  REID FALSE POSITIVE ANALYSIS - node {node_id}")
    print(f"{'=' * W}")
    print(
        f"  scene_nodes={len(scene_nodes)} from={scene_source}  final_scene_nodes={len(final_scene_nodes)}  "
        f"GT={len(gt_data['instances_by_id'])}  FP={len(fp_reports)}"
    )
    print(f"  reid_debug={'yes' if reid_debug else 'no'}  report={out_path}")

    if reid_debug and reid_debug.get("observation_budget_violations"):
        print("\n  Observation budget violations")
        for entry in reid_debug["observation_budget_violations"]:
            images = ", ".join(
                f"{item['image_id']} x{item['count']}" for item in entry.get("violations", [])
            )
            print(f"    {entry['scene_node_id']} code={entry['code']} -> {images}")

    if not fp_reports:
        print("\n  Nessun falso positivo rilevato nel mapping pred->GT.")
        return report

    print("\n  False positives")
    for fp in fp_reports:
        print(
            f"    {fp['scene_node_id']} code={fp['code']} supp={fp['support_count']} "
            f"purity={_fmt_float(fp['purity'])} majority_gt={fp['majority_gt_id']}"
        )
        print(f"      images={fp['support_images']}")
        print(f"      target={fp['target_scene_node_id']} target_images={fp['target_support_images']}")
        print(f"      votes={fp['votes']}")
        if fp["root_causes"]:
            print(f"      causes={' | '.join(fp['root_causes'])}")
        if fp["same_image_gt_duplicates"]:
            for dup in fp["same_image_gt_duplicates"]:
                print(
                    f"      same-image GT duplicate: image={dup['image_id']} gt={dup['gt_instance_id']} "
                    f"obs={dup['observed_ids']}"
                )
        cross = fp.get("cross_group_analysis")
        if cross:
            cps = cross["cross_pair_summary"]
            soft = cross["soft_merge"]
            print(
                f"      cross_pairs={cps['n_cross_pairs']} with_stats={cps['n_with_pair_stats']} "
                f"effective={cps['n_effective_affinity']} mean_ratio={_fmt_float(cps['mean_ratio'])} "
                f"max_ratio={_fmt_float(cps['max_ratio'])}"
            )
            print(
                f"      soft_merge overlap={soft['overlap_images']} proto={_fmt_float(soft['proto_sim'])} "
                f"cross={_fmt_float(soft['cross_coassoc'])} score={_fmt_float(soft['score'])}"
            )

    return report


def main() -> None:
    if len(sys.argv) > 1:
        node_ids = [int(arg) for arg in sys.argv[1:]]
    else:
        node_ids = sorted(
            int(path.name.replace("node", ""))
            for path in MULTIVIEW_DIR.glob("node*")
            if path.is_dir() and path.name.replace("node", "").isdigit()
        )
    for node_id in node_ids:
        analyze_node(node_id)


if __name__ == "__main__":
    main()
