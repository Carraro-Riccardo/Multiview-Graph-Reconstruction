"""Graph projection debug — overlays the fused graph onto each original image.

Per ogni vista mostra:
  • bbox di ogni detection colorato per scene node
  • archi del grafo finale proiettati tra le detection corrispondenti
  • colore arco = stato di consistenza nella vista:
      verde  — entrambi visibili E spazialmente vicini   (supported)
      arancio — entrambi visibili MA distanti             (violated)
      grigio  — almeno un endpoint non visibile in questa vista (invisible)

Output per nodo: DEBUG_OUTPUT_DIR/nodeX/projections/proj_<image>.png
Output summary:  DEBUG_OUTPUT_DIR/nodeX/projections/consistency_report.json

Uso:
    python debug_graph_projection.py 3
    python debug_graph_projection.py 2 7 9
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_reconstruction.config import DEBUG_OUTPUT_DIR, DIAMETER_CODES, IMAGES_DIR, WORK_DIR as OUTPUT_ROOT

NODE_ID   = 3
WORK_DIR  = OUTPUT_ROOT / "multiview"
CACHE_DIR = OUTPUT_ROOT / "detection_cache"

# Fattore moltiplicativo del cutoff KNN per considerare due nodi "vicini"
# (stesso valore usato in instance_spatial_edges.py)
SPATIAL_CLOSE_FACTOR = 1.8
# ─────────────────────────────────────────────────────────────────────────────

# Palette colori per scene node (per codice)
_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#cddc39",
    "#ff5722", "#607d8b", "#795548", "#8bc34a", "#ff9800",
]


# ── Utility ───────────────────────────────────────────────────────────────────

def _load_filtered_dets(image_stem: str) -> list[dict]:
    p = CACHE_DIR / f"{image_stem}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    filtered = [
        d for d in data.get("detections", [])
        if d.get("accepted", True)
        and not d.get("is_diameter", d.get("code") in DIAMETER_CODES)
    ]
    for i, d in enumerate(filtered):
        d["_filtered_idx"] = i
    return filtered


def _find_image(image_stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        p = Path(IMAGES_DIR) / f"{image_stem}{ext}"
        if p.exists():
            return p
    return None


def _knn_cutoff(dets: list[dict]) -> float:
    if len(dets) < 2:
        return 0.2
    from scipy.spatial.distance import cdist
    try:
        from graph_reconstruction.config import DIST_MULTIPLIER
    except Exception:
        DIST_MULTIPLIER = 2.5
    pts = np.array([[d["cx"], d["cy"]] for d in dets])
    D   = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(axis=1))) * DIST_MULTIPLIER


def _dist2d(a: dict, b: dict) -> float:
    return float(np.sqrt((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2))


# ── Core ──────────────────────────────────────────────────────────────────────

def _load_data(node_id: int) -> tuple[dict, list, list]:
    fused_path  = WORK_DIR / f"node{node_id}" / "fused_graph.json"
    scene_path  = WORK_DIR / f"node{node_id}" / "scene_nodes.json"
    fused  = json.loads(fused_path.read_text(encoding="utf-8"))
    scene_nodes = json.loads(scene_path.read_text(encoding="utf-8"))
    return fused, scene_nodes, fused["edges"]


def project_graph_onto_view(
    node_id:     int,
    image_stem:  str,
    fused:       dict,
    scene_nodes: list,
) -> dict:
    """
    Proietta il grafo finale sulla vista `image_stem`.

    Restituisce un dict con:
        supported_edges, violated_edges, invisible_edges,
        consistency_score, edge_details
    """
    # Mappa scene_node_id → supporting_observed_ids per questa vista
    sn_to_obs: dict[str, dict | None] = {}
    for sn in scene_nodes:
        sn_id = sn["scene_node_id"]
        # Trova la detection di questa scena node in questa vista
        obs_in_view = [
            oid for oid in sn.get("supporting_observed_ids", [])
            if oid.startswith(image_stem + "_")
        ]
        if obs_in_view:
            obs_id  = obs_in_view[0]              # al massimo 1 per vista
            det_idx = int(obs_id.rsplit("_", 1)[1])
            dets    = _load_filtered_dets(image_stem)
            if det_idx < len(dets):
                sn_to_obs[sn_id] = dets[det_idx]
            else:
                sn_to_obs[sn_id] = None
        else:
            sn_to_obs[sn_id] = None

    # Cutoff spaziale per questa vista
    dets_all = _load_filtered_dets(image_stem)
    cutoff   = _knn_cutoff(dets_all) if dets_all else 0.15

    # Analizza ogni arco
    supported, violated, invisible = [], [], []
    edge_details = []

    for edge in fused["edges"]:
        a, b    = edge["node_a"], edge["node_b"]
        det_a   = sn_to_obs.get(a)
        det_b   = sn_to_obs.get(b)

        if det_a is None or det_b is None:
            invisible.append((a, b))
            edge_details.append({
                "a": a, "b": b, "status": "invisible",
                "dist": None, "cutoff": cutoff,
            })
        else:
            dist = _dist2d(det_a, det_b)
            if dist <= cutoff * SPATIAL_CLOSE_FACTOR:
                supported.append((a, b))
                status = "supported"
            else:
                violated.append((a, b))
                status = "violated"
            edge_details.append({
                "a": a, "b": b, "status": status,
                "dist": round(dist, 4),
                "cutoff_threshold": round(cutoff * SPATIAL_CLOSE_FACTOR, 4),
            })

    visible_edges = len(supported) + len(violated)
    consistency   = len(supported) / visible_edges if visible_edges > 0 else 1.0

    return {
        "image":            image_stem,
        "supported_edges":  supported,
        "violated_edges":   violated,
        "invisible_edges":  invisible,
        "consistency_score": consistency,
        "visible_edges":    visible_edges,
        "edge_details":     edge_details,
        "cutoff":           cutoff,
        "sn_to_obs":        sn_to_obs,
        "dets_all":         dets_all,
    }


def draw_projection(
    node_id:    int,
    image_stem: str,
    result:     dict,
    fused:      dict,
    out_path:   Path,
) -> None:
    img_path = _find_image(image_stem)
    if img_path is None:
        print(f"  [proj] image not found: {image_stem}")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]

    # Palette per scene node
    all_sn_ids = sorted(result["sn_to_obs"].keys())
    codes = {n["id"]: n.get("code", 0) for n in fused["nodes"]}
    unique_codes = sorted(set(codes.values()))
    code_color   = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(unique_codes)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor="#0d1117")
    ax_img, ax_info = axes
    ax_img.set_facecolor("#0d1117")
    ax_info.set_facecolor("#0d1117")

    # ── Immagine con proiezione ───────────────────────────────────────────────
    ax_img.imshow(img)
    ax_img.set_xlim(0, W); ax_img.set_ylim(H, 0)
    ax_img.axis("off")

    sn_to_obs = result["sn_to_obs"]
    cutoff_px  = result["cutoff"] * SPATIAL_CLOSE_FACTOR

    # Disegna bbox detection
    for sn_id, det in sn_to_obs.items():
        if det is None:
            continue
        code  = codes.get(sn_id, 0)
        color = code_color.get(code, "#888888")
        bbox  = det["bbox_norm"]
        x1, y1, x2, y2 = bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0", linewidth=2,
            edgecolor=color, facecolor="none",
        )
        ax_img.add_patch(rect)
        ax_img.text(
            x1, y1 - 4, sn_id,
            color=color, fontsize=6, fontweight="bold",
            bbox=dict(facecolor="#0d1117", alpha=0.6, pad=1, edgecolor="none"),
        )

    # Disegna archi
    edge_style = {
        "supported": dict(color="#2ecc71", lw=2.5, alpha=0.9, ls="-"),
        "violated":  dict(color="#e74c3c", lw=2.5, alpha=0.9, ls="--"),
        "invisible": dict(color="#555555", lw=1.2, alpha=0.4, ls=":"),
    }
    for detail in result["edge_details"]:
        a, b   = detail["a"], detail["b"]
        det_a  = sn_to_obs.get(a)
        det_b  = sn_to_obs.get(b)
        status = detail["status"]

        if det_a is None or det_b is None:
            continue  # nessuno dei due visibile → skip arco

        xa, ya = det_a["cx"] * W, det_a["cy"] * H
        xb, yb = det_b["cx"] * W, det_b["cy"] * H
        st = edge_style[status]
        ax_img.plot([xa, xb], [ya, yb], color=st["color"],
                    lw=st["lw"], alpha=st["alpha"], ls=st["ls"], zorder=3)

        # Etichetta distanza su archi violated
        if status == "violated":
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            ax_img.text(
                mx, my, f"d={detail['dist']:.3f}\nth={detail['cutoff_threshold']:.3f}",
                color="#e74c3c", fontsize=5.5, ha="center", va="center",
                bbox=dict(facecolor="#1a1a1a", alpha=0.75, pad=1, edgecolor="none"),
            )

    cs  = result["consistency_score"]
    n_s = len(result["supported_edges"])
    n_v = len(result["violated_edges"])
    n_i = len(result["invisible_edges"])
    bar_color = "#2ecc71" if cs >= 0.8 else "#f39c12" if cs >= 0.5 else "#e74c3c"
    ax_img.set_title(
        f"{image_stem}  —  consistency {cs:.0%}  "
        f"({n_s} supported  {n_v} violated  {n_i} invisible)",
        color=bar_color, fontsize=10, pad=6,
    )

    # ── Pannello info ─────────────────────────────────────────────────────────
    ax_info.axis("off")
    lines = [
        ("EDGE CONSISTENCY REPORT", "white", 11, True),
        ("", "white", 8, False),
        (f"View:  {image_stem}", "#aaaaaa", 8, False),
        (f"Node:  {node_id}", "#aaaaaa", 8, False),
        (f"Score: {cs:.1%}  ({n_s}✓ / {n_v}✗ / {n_i} invis.)", bar_color, 10, True),
        ("", "white", 8, False),
        ("── Supported ─────────────────", "#2ecc71", 8, True),
    ]
    for a, b in result["supported_edges"]:
        d = next(e for e in result["edge_details"] if e["a"] == a and e["b"] == b)
        lines.append((f"  {a} — {b}   d={d['dist']:.3f}", "#2ecc71", 7.5, False))
    lines.append(("── Violated ──────────────────", "#e74c3c", 8, True))
    for a, b in result["violated_edges"]:
        d = next(e for e in result["edge_details"] if e["a"] == a and e["b"] == b)
        lines.append((
            f"  {a} — {b}\n"
            f"    dist={d['dist']:.3f}  thresh={d['cutoff_threshold']:.3f}",
            "#e74c3c", 7.5, False,
        ))
    lines.append(("── Invisible ─────────────────", "#777777", 8, True))
    for a, b in result["invisible_edges"]:
        lines.append((f"  {a} — {b}", "#555555", 7, False))

    y_cursor = 0.97
    for text, color, fs, bold in lines:
        ax_info.text(
            0.02, y_cursor, text,
            color=color, fontsize=fs,
            fontweight="bold" if bold else "normal",
            transform=ax_info.transAxes, va="top",
            fontfamily="monospace",
        )
        n_newlines = text.count("\n") + 1
        y_cursor -= 0.025 * n_newlines * (fs / 8.0)
        if y_cursor < 0.02:
            break

    # Legend
    legend_handles = [
        mpatches.Patch(color="#2ecc71", label="Supported (visible + close)"),
        mpatches.Patch(color="#e74c3c", label="Violated (visible but far)"),
        mpatches.Patch(color="#555555", label="Invisible (≥1 endpoint absent)"),
    ]
    ax_img.legend(handles=legend_handles, loc="lower right",
                  fontsize=8, framealpha=0.7,
                  facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout(pad=0.5)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_global_consistency(view_results: list[dict]) -> dict:
    """
    Aggrega i risultati per-vista in un report di consistenza globale.

    Per ogni arco del grafo calcola:
      - n_views_where_both_visible
      - n_views_where_supported
      - global_consistency  = supported / visible
    """
    edge_stats: dict = {}

    for vr in view_results:
        for detail in vr["edge_details"]:
            key = (detail["a"], detail["b"])
            if key not in edge_stats:
                edge_stats[key] = {
                    "a": detail["a"], "b": detail["b"],
                    "n_visible": 0, "n_supported": 0, "n_violated": 0,
                    "violated_in": [], "supported_in": [],
                }
            st = detail["status"]
            if st == "supported":
                edge_stats[key]["n_visible"]   += 1
                edge_stats[key]["n_supported"]  += 1
                edge_stats[key]["supported_in"].append(vr["image"])
            elif st == "violated":
                edge_stats[key]["n_visible"]  += 1
                edge_stats[key]["n_violated"] += 1
                edge_stats[key]["violated_in"].append(vr["image"])

    for k, s in edge_stats.items():
        nv = s["n_visible"]
        s["global_consistency"] = s["n_supported"] / nv if nv > 0 else None
        s["verdict"] = (
            "never_visible"       if nv == 0 else
            "always_consistent"   if s["n_violated"] == 0 else
            "always_inconsistent" if s["n_supported"] == 0 else
            "partially_consistent"
        )

    # Ordina per consistenza crescente (problemi prima)
    sorted_stats = sorted(
        edge_stats.values(),
        key=lambda s: (
            s["global_consistency"] if s["global_consistency"] is not None else 2.0,
        ),
    )

    mean_cs = np.mean([
        vr["consistency_score"] for vr in view_results
    ]) if view_results else 0.0

    return {
        "node_mean_consistency": float(mean_cs),
        "n_views": len(view_results),
        "edges": sorted_stats,
        "problem_edges": [
            s for s in sorted_stats
            if s["verdict"] in ("always_inconsistent", "partially_consistent")
        ],
    }


def _print_consistency_report(report: dict, node_id: int) -> None:
    cs = report["node_mean_consistency"]
    bar = "#" * int(cs * 20)
    color_tag = "OK" if cs >= 0.8 else "!!" if cs >= 0.5 else "XX"
    print(f"\n  {'=' * 70}")
    print(f"  GLOBAL CONSISTENCY - node {node_id}")
    print(f"  {'=' * 70}")
    print(f"  Mean consistency : {cs:.1%}  {color_tag}  [{bar:<20}]")
    print(f"  Views analysed   : {report['n_views']}")
    print()

    problems = report["problem_edges"]
    if not problems:
        print("  All visible edges are globally consistent. OK")
    else:
        print(f"  Problem edges ({len(problems)}):")
        print(f"  {'edge':<30} {'vis':>4} {'sup':>4} {'vio':>4} {'cons':>7}  verdict")
        print(f"  {'-' * 70}")
        for s in problems:
            cs_e = f"{s['global_consistency']:.0%}" if s["global_consistency"] is not None else "  n/a"
            print(
                f"  {s['a']:<14} - {s['b']:<14} "
                f"{s['n_visible']:>4} {s['n_supported']:>4} {s['n_violated']:>4} "
                f"{cs_e:>7}  {s['verdict']}"
            )
            if s["violated_in"]:
                print(f"    violated in: {', '.join(s['violated_in'])}")
        print(f"  {'-' * 70}")

    print(f"  {'=' * 70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_node(node_id: int) -> dict:
    fused, scene_nodes, _ = _load_data(node_id)
    all_images = fused.get("images", [])

    # Ricava le immagini dai support_images dei scene nodes se non nel grafo
    if not all_images:
        imgs_set = set()
        for sn in scene_nodes:
            imgs_set.update(sn.get("support_images", []))
        all_images = sorted(imgs_set)

    out_dir = DEBUG_OUTPUT_DIR / f"node{node_id}" / "projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    view_results = []

    for image_stem in all_images:
        result = project_graph_onto_view(node_id, image_stem, fused, scene_nodes)
        view_results.append(result)

        out_path = out_dir / f"proj_{image_stem}.png"
        draw_projection(node_id, image_stem, result, fused, out_path)
        cs = result["consistency_score"]
        tag = "OK" if cs >= 0.8 else "!!" if cs >= 0.5 else "XX"
        print(
            f"  [proj] {image_stem}  consistency={cs:.0%} {tag}  "
            f"({len(result['supported_edges'])} sup / "
            f"{len(result['violated_edges'])} viol / "
            f"{len(result['invisible_edges'])} invis)"
        )

    report = compute_global_consistency(view_results)
    _print_consistency_report(report, node_id)

    # Salva report JSON
    report_serializable = {
        "node_id": node_id,
        "node_mean_consistency": report["node_mean_consistency"],
        "n_views": report["n_views"],
        "edges": [
            {**e, "violated_in": e["violated_in"], "supported_in": e["supported_in"]}
            for e in report["edges"]
        ],
        "problem_edges": [e["a"] + " — " + e["b"] for e in report["problem_edges"]],
    }
    out_json = out_dir / "consistency_report.json"
    out_json.write_text(json.dumps(report_serializable, indent=2, ensure_ascii=False))
    print(f"  [proj] report saved: {out_json.resolve()}")

    return report


if __name__ == "__main__":
    node_ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [NODE_ID]
    for nid in node_ids:
        run_node(nid)
