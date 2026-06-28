"""
Diagnostica over-segmentation per un nodo idraulico.

Uso:
    python diagnose_node.py          # default: NODE_ID = 9
    python diagnose_node.py 7

Risponde a:
  - Quali pred node sono spurii (non matchati a nessun GT)?
  - Quale GT istanza corrispondono le loro detection?
  - Qual è la copertura (n_immagini) di ogni GT istanza?
  - Dove si trova il "buco" di copertura?
  - Perché il re-id non li ha uniti al cluster corretto?
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from graph_reconstruction.shared.linear_assignment import linear_sum_assignment
from graph_reconstruction.config import ANNOTATIONS_DIR, COMPONENT_INFO, CONF_THRESHOLD, DIAMETER_CODES, IMAGES_DIR, LABELS_EXT_DIR, WORK_DIR

NODE_ID     = int(sys.argv[1]) if len(sys.argv) > 1 else 9
ANNOTATIONS = ANNOTATIONS_DIR
CACHE_DIR   = WORK_DIR / "detection_cache"
MULTIVIEW   = WORK_DIR / "multiview"
IOU_THRESH  = 0.30
W = 76


# ── utils ─────────────────────────────────────────────────────────────────────

def iou(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def xywh(cx, cy, w, h) -> list:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def load_dets(stem: str) -> list:
    p = CACHE_DIR / f"{stem}.json"
    if not p.exists():
        return []
    raw = json.loads(p.read_text()).get("detections", [])
    return [
        d for d in raw
        if d.get("confidence", 0) >= CONF_THRESHOLD
        and not d.get("is_diameter", d.get("code", -1) in DIAMETER_CODES)
    ]


def load_gt_by_image(node_id: int) -> dict[str, list]:
    result = {}
    for p in sorted(IMAGES_DIR.glob(f"node{node_id}_*.jpg")):
        stem = p.stem
        lp = LABELS_EXT_DIR / f"{stem}_extended.txt"
        if not lp.exists():
            continue
        insts = []
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            code = int(parts[6])
            if code in DIAMETER_CODES:
                continue
            insts.append({
                "nid":  f"{code}#{int(parts[5])}",
                "code": code,
                "bbox": xywh(float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])),
            })
        result[stem] = insts
    return result


def best_gt_match(det, gt_list) -> tuple[str | None, float]:
    best_nid, best_v = None, 0.0
    for g in gt_list:
        if g["code"] != det["code"]:
            continue
        v = iou(det["bbox_norm"], g["bbox"])
        if v > best_v:
            best_v, best_nid = v, g["nid"]
    return (best_nid if best_v >= IOU_THRESH else None), best_v


# ── purity + Hungarian (same logic as evaluate.py) ───────────────────────────

def purity_for_sn(sn: dict, gt_by_image: dict) -> dict:
    votes: Counter = Counter()
    n_matched = 0
    for obs_id in sn["supporting_observed_ids"]:
        parts = obs_id.rsplit("_", 1)
        if len(parts) != 2:
            continue
        stem, idx = parts[0], int(parts[1])
        dets = load_dets(stem)
        if idx >= len(dets):
            continue
        gt_nid, v = best_gt_match(dets[idx], gt_by_image.get(stem, []))
        if gt_nid:
            votes[gt_nid] += 1
            n_matched += 1
    maj = votes.most_common(1)[0] if votes else (None, 0)
    return {
        "votes":   dict(votes),
        "n_matched": n_matched,
        "majority": maj[0],
        "majority_count": maj[1],
        "purity": maj[1] / n_matched if n_matched > 0 else None,
    }


def build_bijective_mapping(sn_purities, gt_nodes) -> dict[str, str]:
    by_code: dict[int, list] = defaultdict(list)
    for sn, pur in sn_purities:
        by_code[sn["fused_code"]].append((sn, pur))

    gt_by_code: dict[int, list] = defaultdict(list)
    for nid, n in gt_nodes.items():
        gt_by_code[n["code"]].append(nid)

    sn_to_gt: dict[str, str] = {}
    for code, items in by_code.items():
        gt_iids = gt_by_code.get(code, [])
        if not gt_iids:
            continue
        cost = np.zeros((len(items), len(gt_iids)))
        for i, (sn, pur) in enumerate(items):
            for j, gt_nid in enumerate(gt_iids):
                cost[i, j] = pur["votes"].get(gt_nid, 0)
        row_ind, col_ind = linear_sum_assignment(-cost)
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] > 0:
                sn_to_gt[items[i][0]["scene_node_id"]] = gt_iids[j]
    return sn_to_gt


# ── main diagnostic ───────────────────────────────────────────────────────────

def main():
    gt_data     = json.loads((ANNOTATIONS / f"node{NODE_ID}_gt.json").read_text("utf-8"))
    gt_nodes    = {n["nid"]: n for n in gt_data["nodes"]}
    gt_by_image = load_gt_by_image(NODE_ID)
    sn_list     = json.loads((MULTIVIEW / f"node{NODE_ID}" / "scene_nodes.json").read_text("utf-8"))
    images      = sorted(gt_by_image.keys())

    sn_purities = [(sn, purity_for_sn(sn, gt_by_image)) for sn in sn_list]
    sn_to_gt    = build_bijective_mapping(sn_purities, gt_nodes)
    gt_to_sn    = {v: k for k, v in sn_to_gt.items()}

    unmatched_pred = [sn for sn in sn_list if sn["scene_node_id"] not in sn_to_gt]
    unmatched_gt   = [nid for nid in gt_nodes if nid not in gt_to_sn]

    # ── group by code ─────────────────────────────────────────────────────────
    all_codes = sorted({sn["fused_code"] for sn in sn_list} | {gt_nodes[n]["code"] for n in gt_nodes})

    print(f"\n{'═' * W}")
    print(f"  DIAGNOSTICA OVER-SEGMENTATION  —  nodo {NODE_ID}")
    print(f"{'═' * W}")

    print(f"\n  pred totali: {len(sn_list)}   GT totali: {len(gt_nodes)}")
    print(f"  pred unmatched: {len(unmatched_pred)}   GT uncovered: {len(unmatched_gt)}")

    # ── per-code summary ──────────────────────────────────────────────────────
    print(f"\n  {'── Code-level':─<{W}}")
    print(f"  {'code':<6}{'name':<18}{'GT':>4}{'pred':>6}  diff")
    print(f"  {'─' * (W-2)}")
    for code in all_codes:
        gt_n   = sum(1 for n in gt_nodes.values() if n["code"] == code)
        pred_n = sum(1 for sn in sn_list if sn["fused_code"] == code)
        _, name = COMPONENT_INFO.get(code, (0, f"code_{code}"))
        diff = pred_n - gt_n
        flag = f"  +{diff} OVER" if diff > 0 else (f"  {diff} UNDER" if diff < 0 else "  OK")
        print(f"  {code:<6}{name:<18}{gt_n:>4}{pred_n:>6}{flag}")

    # ── spurious pred nodes ───────────────────────────────────────────────────
    if unmatched_pred:
        print(f"\n  {'── Pred nodes non mappati (FP) ':─<{W}}")
        for sn in sorted(unmatched_pred, key=lambda s: (s["fused_code"], s["scene_node_id"])):
            sid   = sn["scene_node_id"]
            _, pur = next(p for p in sn_purities if p[0]["scene_node_id"] == sid)
            print(f"\n  ▶ {sid}  supp={sn['support_count']}  "
                  f"conf={sn['fused_confidence']:.3f}  images={sn['support_images']}")
            print(f"    votes={pur['votes']}  majority={pur['majority']}  purity={pur['purity']}")

            # Show the single detection
            for obs_id in sn["supporting_observed_ids"]:
                stem, idx = obs_id.rsplit("_", 1)
                dets = load_dets(stem)
                det  = dets[int(idx)] if int(idx) < len(dets) else None
                if not det:
                    print(f"    obs {obs_id}: NOT FOUND in cache")
                    continue
                gt_nid, v = best_gt_match(det, gt_by_image.get(stem, []))
                print(f"    obs {obs_id}  code={det['code']}  conf={det['confidence']:.3f}  "
                      f"GT→{gt_nid}  IoU={v:.3f}  bbox={[round(x, 3) for x in det['bbox_norm']]}")

            # Which pred node already covers its majority GT?
            if pur["majority"]:
                covering = [
                    sn2["scene_node_id"]
                    for sn2, pur2 in sn_purities
                    if sn2["scene_node_id"] != sid and pur2["majority"] == pur["majority"]
                ]
                alt = sn_to_gt.get(next(
                    (s["scene_node_id"] for s, p in sn_purities
                     if s["scene_node_id"] != sid and pur["majority"] in p["votes"]), ""), "?")
                print(f"    GT {pur['majority']} è già coperto dal pred node "
                      f"assegnato: {sn_to_gt.get(next((s['scene_node_id'] for s,p in sn_purities if sn_to_gt.get(s['scene_node_id'])==pur['majority']),None),'—')}")
                if covering:
                    print(f"    Altri pred con stesso majority: {covering}")

    # ── GT nodes without pred coverage ───────────────────────────────────────
    if unmatched_gt:
        print(f"\n  {'── GT instances senza pred corrispondente (FN) ':─<{W}}")
        for nid in sorted(unmatched_gt):
            n    = gt_nodes[nid]
            _, name = COMPONENT_INFO.get(n["code"], (0, f"code_{n['code']}"))
            # How many images show this GT instance?
            gt_imgs = [stem for stem, gl in gt_by_image.items()
                       if any(g["nid"] == nid for g in gl)]
            # How many pred detections match it (cross all scene nodes)?
            total_det_matches = sum(
                pur["votes"].get(nid, 0) for _, pur in sn_purities
            )
            print(f"  ▶ GT {nid} ({name})  visible in {len(gt_imgs)} images: "
                  f"{[s.replace(f'node{NODE_ID}_','') for s in gt_imgs]}  "
                  f"total_det_matches={total_det_matches}")

    # ── Per-image detection count vs GT (focus: codes with over-segmentation) ─
    over_codes = [c for c in all_codes
                  if sum(1 for sn in sn_list if sn["fused_code"] == c) >
                     sum(1 for n in gt_nodes.values() if n["code"] == c)]

    for code in over_codes:
        _, name = COMPONENT_INFO.get(code, (0, f"code_{code}"))
        gt_n   = sum(1 for n in gt_nodes.values() if n["code"] == code)
        pred_n = sum(1 for sn in sn_list if sn["fused_code"] == code)
        print(f"\n  {'─' * W}")
        print(f"  OVER-SEGMENTATION: code {code} ({name})  GT={gt_n}  pred={pred_n}")
        print(f"  {'─' * W}")

        # Per-image table
        print(f"\n  {'image':<18} {'det':>4} {'GT':>4}  detections → GT_nid (IoU)")
        print(f"  {'─' * (W-2)}")
        for stem in images:
            dets_c = [(i, d) for i, d in enumerate(load_dets(stem)) if d["code"] == code]
            gt_c   = [g for g in gt_by_image.get(stem, []) if g["code"] == code]
            pairs  = []
            for i, d in dets_c:
                gnid, v = best_gt_match(d, gt_c)
                pairs.append(f"{i}→{gnid or '?'}({v:.2f})")
            flag = " ⚠" if len(dets_c) != len(gt_c) else ""
            print(f"  {stem:<18} {len(dets_c):>4} {len(gt_c):>4}  {' | '.join(pairs)}{flag}")

        # GT instance visibility
        print(f"\n  {'GT instance':<14} {'n_imgs':>7}  images_seen")
        print(f"  {'─' * (W-2)}")
        gt_nids_code = [nid for nid, n in gt_nodes.items() if n["code"] == code]
        gt_coverage  = {}
        for nid in sorted(gt_nids_code):
            imgs = [s for s, gl in gt_by_image.items()
                    if any(g["nid"] == nid for g in gl)]
            gt_coverage[nid] = set(imgs)
            detected_in = []
            for stem in imgs:
                dets_c = [(i, d) for i, d in enumerate(load_dets(stem)) if d["code"] == code]
                best_v = max(
                    (iou(d["bbox_norm"], g["bbox"])
                     for g in gt_by_image.get(stem, []) if g["nid"] == nid
                     for i2, d in dets_c),
                    default=0.0,
                )
                if best_v >= IOU_THRESH:
                    detected_in.append(stem.replace(f"node{NODE_ID}_", ""))
            print(f"  {nid:<14} {len(imgs):>7}  "
                  f"visible={[s.replace(f'node{NODE_ID}_','') for s in imgs]}  "
                  f"detected={detected_in}")

        # Pred cluster coverage for this code
        print(f"\n  {'Pred cluster':<14} {'supp':>6} {'n_imgs':>7}  majority→GT  purity  images")
        print(f"  {'─' * (W-2)}")
        for sn, pur in sorted(sn_purities,
                               key=lambda x: (x[0]["fused_code"], -x[0]["support_count"])):
            if sn["fused_code"] != code:
                continue
            assigned = sn_to_gt.get(sn["scene_node_id"], "UNMATCHED")
            flag = " *** SPURIOUS" if assigned == "UNMATCHED" else ""
            pur_s = f"{pur['purity']:.3f}" if pur["purity"] is not None else "  n/a"
            imgs_short = [s.replace(f"node{NODE_ID}_", "") for s in sn["support_images"]]
            print(f"  {sn['scene_node_id']:<14} {sn['support_count']:>6} {len(sn['support_images']):>7}  "
                  f"{pur['majority'] or '—'}  {pur_s}  {imgs_short}{flag}")

        # Coverage gap: which images are NOT represented in the matching pred cluster?
        print(f"\n  {'── Copertura gap per GT instance ':─<{W}}")
        for nid in sorted(gt_nids_code):
            matched_pred = gt_to_sn.get(nid)
            if matched_pred is None:
                print(f"  GT {nid}: NESSUN pred assegnato")
                continue
            pred_sn = next((s for s, p in sn_purities if s["scene_node_id"] == matched_pred), None)
            if pred_sn is None:
                continue
            pred_imgs = set(pred_sn["support_images"])
            missing = gt_coverage[nid] - pred_imgs
            if missing:
                miss_short = [s.replace(f"node{NODE_ID}_", "") for s in sorted(missing)]
                print(f"  GT {nid} → pred {matched_pred}  |  immagini GT non coperte: {miss_short}")
            else:
                print(f"  GT {nid} → pred {matched_pred}  |  copertura completa")

    # ── Root-cause summary ────────────────────────────────────────────────────
    print(f"\n  {'── Causa radice ':─<{W}}")
    all_images_counts = {}
    for stem in images:
        for code in over_codes:
            dets_c = sum(1 for d in load_dets(stem) if d["code"] == code)
            gt_c   = sum(1 for g in gt_by_image.get(stem, []) if g["code"] == code)
            if code not in all_images_counts:
                all_images_counts[code] = []
            all_images_counts[code].append((stem, dets_c, gt_c))

    for code in over_codes:
        _, name = COMPONENT_INFO.get(code, (0, f"code_{code}"))
        rows = all_images_counts[code]
        max_dets = max(r[1] for r in rows)
        crowded = [r[0].replace(f"node{NODE_ID}_", "") for r in rows if r[1] == max_dets]
        under   = [r[0].replace(f"node{NODE_ID}_", "") for r in rows if r[1] < r[2]]
        print(f"\n  {name} (code {code}):")
        print(f"    max det/image = {max_dets} (immagine/i più affollata: {crowded})")
        print(f"    immagini con meno det che GT: {under}")
        print(f"    → le detection 'in eccesso' nell'immagine affollata non trovano")
        print(f"      partner cross-image → COASSOC_MIN_RUNS=2 non soddisfatto → singleton")


if __name__ == "__main__":
    main()
