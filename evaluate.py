"""
Confronto scene nodes ricostruiti vs Ground Truth per ogni nodo idraulico.

Uso:
    python evaluate_reid.py           # tutti i nodi disponibili
    python evaluate_reid.py 2 7 9     # solo i nodi specificati

Output terminale per ogni nodo:
  - tabella GT instances (code, nome, n istanze)
  - tabella scene nodes  (code, nome, n predetti, support, conf)
  - confronto TP / FP / FN per codice
  - purity per ogni scene node (frazione di sue detection che matchano un unico GT)

GT format (labels_extended):
    <class> <cx_n> <cy_n> <w_n> <h_n> <instance_id> <hydraulic_code>
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import (
    COMPONENT_INFO,
    CONF_THRESHOLD,
    IMAGES_DIR,
    LABELS_EXT_DIR,
    WORK_DIR,
)

MULTIVIEW_DIR = WORK_DIR / "multiview"
CACHE_DIR     = WORK_DIR / "detection_cache"

# IoU minima per considerare un match detection→GT
IOU_THRESH = 0.30

W = 72  # larghezza separatori


# ── GT loading ────────────────────────────────────────────────────────────────

def _iou(b1: list, b2: list) -> float:
    """bbox normalizzate [x1,y1,x2,y2]."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def _xywh_to_xyxy(cx, cy, w, h) -> list:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def load_gt_for_image(image_stem: str) -> list[dict]:
    """
    Carica GT instances per una singola immagine, escludendo diameter codes.
    Ritorna lista di {instance_id, code, name, bbox_norm [x1y1x2y2]}.
    """
    from config import DIAMETER_CODES
    p = LABELS_EXT_DIR / f"{image_stem}_extended.txt"
    if not p.exists():
        return []
    instances = []
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        code = int(parts[6])
        if code in DIAMETER_CODES:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        instance_id  = int(parts[5])
        _, name      = COMPONENT_INFO.get(code, (0, f"code_{code}"))
        instances.append({
            "instance_id": instance_id,
            "code":        code,
            "name":        name,
            "bbox_norm":   _xywh_to_xyxy(cx, cy, w, h),
            "image_stem":  image_stem,
        })
    return instances


def load_gt_for_node(node_id: int) -> dict:
    """
    Aggrega GT per tutte le immagini del nodo.
    Ritorna:
      instances_by_id:   {instance_id -> {code, name, bbox per immagine}}
      instances_by_image: {image_stem -> [gt_instance_dicts]}
    """
    image_stems = sorted(
        p.stem for p in IMAGES_DIR.glob(f"node{node_id}_*.jpg")
    )
    instances_by_id: dict[int, dict] = {}
    instances_by_image: dict[str, list] = {}

    for stem in image_stems:
        gt = load_gt_for_image(stem)
        instances_by_image[stem] = gt
        for item in gt:
            iid  = item["instance_id"]
            code = item["code"]
            if iid not in instances_by_id:
                instances_by_id[iid] = {
                    "instance_id": iid,
                    "code":        code,
                    "name":        item["name"],
                    "images":      [],
                    "bboxes":      {},
                }
            instances_by_id[iid]["images"].append(stem)
            instances_by_id[iid]["bboxes"][stem] = item["bbox_norm"]

    return {
        "image_stems":       image_stems,
        "instances_by_id":   instances_by_id,
        "instances_by_image": instances_by_image,
    }


# ── Scene node loading ────────────────────────────────────────────────────────

def load_scene_nodes(node_id: int) -> list[dict]:
    p = MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def load_detections_for_image(image_stem: str) -> list[dict]:
    """
    Ritorna le detection FILTRATE nello stesso modo di build_observed_graph:
    solo quelle con confidence >= CONF_THRESHOLD e non diameter.
    L'indice nell'observed_node_id si riferisce a questa lista filtrata.
    """
    from config import DIAMETER_CODES
    p = CACHE_DIR / f"{image_stem}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    raw  = data.get("detections", [])
    return [
        d for d in raw
        if d.get("confidence", 0.0) >= CONF_THRESHOLD
        and not d.get("is_diameter", d.get("code", -1) in DIAMETER_CODES)
    ]


# ── Matching detection → GT ───────────────────────────────────────────────────

def match_detections_to_gt(
    det_bbox: list,
    det_code: int,
    gt_instances: list[dict],
) -> tuple[int | None, float]:
    """
    Trova il GT instance che meglio matcha la detection (stesso code, IoU massimo).
    Ritorna (instance_id, iou) oppure (None, 0.0).
    """
    best_iid, best_iou = None, 0.0
    for gt in gt_instances:
        if gt["code"] != det_code:
            continue
        iou = _iou(det_bbox, gt["bbox_norm"])
        if iou > best_iou:
            best_iou = iou
            best_iid = gt["instance_id"]
    if best_iou < IOU_THRESH:
        return None, 0.0
    return best_iid, best_iou


# ── Per-scene-node purity ─────────────────────────────────────────────────────

def compute_scene_node_purity(
    scene_node: dict,
    gt_data: dict,
) -> dict:
    """
    Per ogni detection del scene node, trova il GT instance match.
    Ritorna:
      gt_instance_votes: Counter {instance_id -> n_matches}
      n_matched:  detection matchate a un GT
      n_total:    detection totali nel scene node
      purity:     majority_count / n_matched (o None se 0 matches)
      majority_gt_id: GT instance più votato
    """
    gt_by_image = gt_data["instances_by_image"]
    votes: Counter = Counter()
    n_total   = 0
    n_matched = 0

    for obs_id in scene_node["supporting_observed_ids"]:
        # obs_id format: node{nid}_{imgsfx}_{det_idx}
        # image_stem = tutto tranne l'ultimo segmento
        parts = obs_id.rsplit("_", 1)
        if len(parts) != 2:
            continue
        image_stem = parts[0]
        det_idx    = int(parts[1])

        dets = load_detections_for_image(image_stem)
        if det_idx >= len(dets):
            continue
        det = dets[det_idx]
        if det.get("confidence", 0) < CONF_THRESHOLD:
            continue
        n_total += 1

        gt_instances = gt_by_image.get(image_stem, [])
        iid, iou = match_detections_to_gt(det["bbox_norm"], det["code"], gt_instances)
        if iid is not None:
            votes[iid] += 1
            n_matched += 1

    majority_gt_id = votes.most_common(1)[0][0] if votes else None
    majority_count = votes[majority_gt_id] if majority_gt_id is not None else 0
    purity         = majority_count / n_matched if n_matched > 0 else None

    return {
        "gt_instance_votes": dict(votes),
        "n_total":           n_total,
        "n_matched":         n_matched,
        "majority_gt_id":    majority_gt_id,
        "majority_count":    majority_count,
        "purity":            purity,
    }


# ── Per-node evaluation ───────────────────────────────────────────────────────

def evaluate_node(node_id: int) -> dict | None:
    scene_nodes = load_scene_nodes(node_id)
    if not scene_nodes:
        print(f"  [node {node_id}] scene_nodes.json non trovato — skip")
        return None

    gt_data = load_gt_for_node(node_id)
    if not gt_data["instances_by_id"]:
        print(f"  [node {node_id}] nessun GT trovato in {LABELS_EXT_DIR} — skip")
        return None

    gt_instances     = gt_data["instances_by_id"]
    gt_by_code       = defaultdict(list)
    for inst in gt_instances.values():
        gt_by_code[inst["code"]].append(inst)

    pred_by_code: dict[int, list] = defaultdict(list)
    for sn in scene_nodes:
        pred_by_code[sn["fused_code"]].append(sn)

    # ── Purity per scene node ─────────────────────────────────────────────────
    sn_purities = []
    for sn in scene_nodes:
        pur = compute_scene_node_purity(sn, gt_data)
        sn_purities.append((sn, pur))

    # ── Code-level TP/FP/FN ──────────────────────────────────────────────────
    all_codes = sorted(set(gt_by_code) | set(pred_by_code))
    code_rows = []
    for code in all_codes:
        gt_n    = len(gt_by_code[code])
        pred_n  = len(pred_by_code[code])
        _, name = COMPONENT_INFO.get(code, (0, f"code_{code}"))
        tp = min(gt_n, pred_n)
        fp = max(0, pred_n - gt_n)
        fn = max(0, gt_n - pred_n)
        code_rows.append({
            "code": code, "name": name,
            "gt": gt_n, "pred": pred_n,
            "tp": tp, "fp": fp, "fn": fn,
        })

    total_gt   = sum(r["gt"]   for r in code_rows)
    total_pred = sum(r["pred"] for r in code_rows)
    total_tp   = sum(r["tp"]   for r in code_rows)
    total_fp   = sum(r["fp"]   for r in code_rows)
    total_fn   = sum(r["fn"]   for r in code_rows)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    f1        = (2 * precision * recall / (precision + recall)
                 if precision is not None and recall is not None
                    and (precision + recall) > 0 else None)

    valid_purities = [p["purity"] for _, p in sn_purities if p["purity"] is not None]
    avg_purity     = float(np.mean(valid_purities)) if valid_purities else None

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'═' * (W + 4)}")
    print(f"  NODE {node_id}  —  GT instances: {total_gt}  |  scene nodes: {total_pred}  "
          f"|  {len(gt_data['image_stems'])} viste")
    print(f"{'═' * (W + 4)}")

    # GT table
    print(f"\n  {'── GT instances ':─<{W}}")
    print(f"  {'code':<6}{'name':<18}{'n_instances':>12}{'n_images':>10}")
    print(f"  {'─' * W}")
    for code in sorted(gt_by_code):
        insts = gt_by_code[code]
        _, name = COMPONENT_INFO.get(code, (0, f"code_{code}"))
        imgs  = len({i for inst in insts for i in inst["images"]})
        print(f"  {code:<6}{name:<18}{len(insts):>12}{imgs:>10}")

    # Scene nodes vs GT comparison
    print(f"\n  {'── Code-level comparison (GT vs Pred) ':─<{W}}")
    print(f"  {'code':<6}{'name':<18}{'GT':>5}{'pred':>6}{'TP':>5}{'FP':>5}{'FN':>5}  status")
    print(f"  {'─' * W}")
    for r in code_rows:
        if r["gt"] == 0:
            status = "? spurio"
        elif r["fp"] > 0:
            status = "⚠ over-segm."
        elif r["fn"] > 0:
            status = "✗ mancante"
        else:
            status = "✓"
        print(f"  {r['code']:<6}{r['name']:<18}{r['gt']:>5}{r['pred']:>6}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}  {status}")
    print(f"  {'─' * W}")
    print(f"  {'TOTALE':<24}{total_gt:>5}{total_pred:>6}{total_tp:>5}{total_fp:>5}{total_fn:>5}")
    prec_s = f"{precision:.3f}" if precision is not None else "n/a"
    rec_s  = f"{recall:.3f}"    if recall    is not None else "n/a"
    f1_s   = f"{f1:.3f}"        if f1        is not None else "n/a"
    pur_s  = f"{avg_purity:.3f}" if avg_purity is not None else "n/a"
    print(f"\n  precision={prec_s}  recall={rec_s}  F1={f1_s}  avg_purity={pur_s}")

    # Per-scene-node purity detail
    print(f"\n  {'── Scene node purity ':─<{W}}")
    print(f"  {'scene_node_id':<18}{'name':<18}{'code':>6}{'supp':>6}"
          f"{'matched':>9}{'purity':>8}  majority_gt_id")
    print(f"  {'─' * W}")
    for sn, pur in sorted(sn_purities,
                          key=lambda x: (x[0]["fused_code"], x[0]["scene_node_id"])):
        p_str  = f"{pur['purity']:.3f}" if pur["purity"] is not None else "  n/a"
        mid    = str(pur["majority_gt_id"]) if pur["majority_gt_id"] is not None else "—"
        flag   = "" if pur["purity"] is None or pur["purity"] >= 0.8 else " ⚠"
        print(f"  {sn['scene_node_id']:<18}{sn['fused_name']:<18}"
              f"{sn['fused_code']:>6}{sn['support_count']:>6}"
              f"{pur['n_matched']:>9}{p_str:>8}  {mid}{flag}")
    print(f"  {'─' * W}")

    return {
        "node_id":    node_id,
        "gt_total":   total_gt,
        "pred_total": total_pred,
        "tp":         total_tp,
        "fp":         total_fp,
        "fn":         total_fn,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "avg_purity": avg_purity,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        node_ids = [int(a) for a in sys.argv[1:]]
    else:
        node_ids = sorted(
            int(p.name.replace("node", ""))
            for p in MULTIVIEW_DIR.iterdir()
            if p.is_dir() and p.name.startswith("node")
        )

    print(f"Nodes to evaluate: {node_ids}")

    rows = []
    for nid in node_ids:
        r = evaluate_node(nid)
        if r:
            rows.append(r)

    if not rows:
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═' * (W + 4)}")
    print(f"  SUMMARY  —  {len(rows)} nodes")
    print(f"{'═' * (W + 4)}")
    print(f"  {'node':>5}{'gt':>6}{'pred':>6}{'tp':>5}{'fp':>5}{'fn':>5}"
          f"{'prec':>8}{'rec':>8}{'F1':>8}{'purity':>8}")
    print(f"  {'─' * (W + 2)}")
    for r in sorted(rows, key=lambda x: x["node_id"]):
        fmt = lambda v: f"{v:.3f}" if v is not None else " n/a"
        print(f"  {r['node_id']:>5}{r['gt_total']:>6}{r['pred_total']:>6}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
              f"{fmt(r['precision']):>8}{fmt(r['recall']):>8}"
              f"{fmt(r['f1']):>8}{fmt(r['avg_purity']):>8}")
    print(f"  {'─' * (W + 2)}")

    # Medie
    def _mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    fmt = lambda v: f"{v:.3f}" if v is not None else " n/a"
    print(f"  {'mean':>5}{'':>6}{'':>6}{'':>5}{'':>5}{'':>5}"
          f"{fmt(_mean('precision')):>8}{fmt(_mean('recall')):>8}"
          f"{fmt(_mean('f1')):>8}{fmt(_mean('avg_purity')):>8}")
    print(f"  {'─' * (W + 2)}")


if __name__ == "__main__":
    main()
