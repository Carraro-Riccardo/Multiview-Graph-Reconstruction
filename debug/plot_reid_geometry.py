"""Visualize per-code 3D centroid clustering for a given node.

Usage:
    python plot_reid_geometry.py <node_id>           # all codes
    python plot_reid_geometry.py <node_id> 15        # only code 15
    python plot_reid_geometry.py <node_id> 15 95     # codes 15 and 95

Output: one PNG per code in output/multiview/node<id>/reid_geometry/
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


WORK_DIR = Path("output")


def _load(node_id: str) -> tuple[dict, dict]:
    base = WORK_DIR / "multiview" / f"node{node_id}"
    geom_path = base / "reid_geometry.json"
    sn_path   = base / "scene_nodes.json"
    if not geom_path.exists():
        raise FileNotFoundError(f"reid_geometry.json not found for node {node_id} — run the pipeline first")
    obs_geom  = json.loads(geom_path.read_text(encoding="utf-8"))
    scene_nodes = json.loads(sn_path.read_text(encoding="utf-8"))
    return obs_geom, scene_nodes


def _plot_code(
    code: int,
    detections: list[dict],
    sn_names: dict[str, str],
    out_dir: Path,
) -> None:
    """One figure per code: XZ and YZ side-by-side projections."""
    scene_ids = sorted({d["scene_node"] for d in detections})
    palette   = cm.tab10.colors
    color_of  = {sid: palette[i % len(palette)] for i, sid in enumerate(scene_ids)}

    image_ids  = sorted({d["image_id"] for d in detections})
    markers    = ["o", "s", "^", "D", "v", "P", "*", "X"]
    marker_of  = {img: markers[i % len(markers)] for i, img in enumerate(image_ids)}

    pts = np.array([d["centroid"] for d in detections])  # (N, 3)
    # VGGT: X=right, Y=down, Z=depth  → use X, Z for top-down; Y, Z for front
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    code_name = sn_names.get(str(code), str(code))
    fig.suptitle(
        f"Node re-ID  |  code {code} — {code_name}  |  "
        f"{len(scene_ids)} cluster(s),  {len(detections)} detection(s)",
        fontsize=12,
    )

    for ax, (hdata, vdata, xlabel, ylabel) in zip(
        axes,
        [
            (x, z, "X  (lateral)", "Z  (depth)"),
            (z, y, "Z  (depth)",   "Y  (vertical, down+)"),
        ],
    ):
        for d, hv, vv in zip(detections, hdata, vdata):
            ax.scatter(
                hv, vv,
                s=120,
                color=color_of[d["scene_node"]],
                marker=marker_of[d["image_id"]],
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                d["image_id"].split("_")[-1],   # last part of image id
                (hv, vv),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
                color=color_of[d["scene_node"]],
            )

        # Draw lines connecting detections in the same cluster
        by_cluster: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(detections):
            by_cluster[d["scene_node"]].append(i)
        for sid, idxs in by_cluster.items():
            if len(idxs) < 2:
                continue
            cx = hdata[idxs]
            cy = vdata[idxs]
            hull_order = np.argsort(np.arctan2(cy - cy.mean(), cx - cx.mean()))
            closed = np.append(hull_order, hull_order[0])
            ax.plot(cx[closed], cy[closed], "-", color=color_of[sid], alpha=0.35, lw=1)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Legend: clusters
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_of[sid],
               markersize=9, markeredgecolor="k", markeredgewidth=0.5,
               label=f"{sid}  ({sn_names.get(sid.split('#')[0] if '#' in sid else sid, sid)})")
        for sid in scene_ids
    ]
    legend_els += [
        Line2D([0], [0], marker=marker_of[img], color="grey", markersize=8, linestyle="None",
               label=f"view {img}")
        for img in image_ids
    ]
    axes[1].legend(handles=legend_els, fontsize=8, loc="upper right", framealpha=0.8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"code_{code:03d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main(node_id: str, filter_codes: list[int] | None = None) -> None:
    obs_geom, scene_nodes_list = _load(node_id)

    # Build code→name map from scene_nodes
    sn_names: dict[str, str] = {}
    for sn in scene_nodes_list:
        sn_names[str(sn["fused_code"])] = sn["fused_name"]
        sn_names[sn["scene_node_id"]]   = sn["fused_name"]

    # Group detections by code
    by_code: dict[int, list[dict]] = defaultdict(list)
    for oid, info in obs_geom.items():
        code = int(info["code"])
        if filter_codes and code not in filter_codes:
            continue
        by_code[code].append({
            "oid":        oid,
            "centroid":   info["centroid"],
            "image_id":   info["image_id"],
            "scene_node": info["scene_node"],
            "code":       code,
        })

    if not by_code:
        print(f"No geometry data for node {node_id}" + (f" codes {filter_codes}" if filter_codes else ""))
        return

    out_dir = WORK_DIR / "multiview" / f"node{node_id}" / "reid_geometry"
    print(f"\nNode {node_id}: plotting {len(by_code)} code(s)  ->  {out_dir}")

    for code in sorted(by_code):
        detections = by_code[code]
        name = sn_names.get(str(code), f"code_{code}")
        n_clusters = len({d["scene_node"] for d in detections})
        print(f"  code {code:4d}  {name:<20}  {len(detections):3d} det  {n_clusters} cluster(s)")
        _plot_code(code, detections, sn_names, out_dir)

    print(f"\nAll plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_reid_geometry.py <node_id> [code1 code2 ...]")
        sys.exit(1)
    _node_id      = sys.argv[1]
    _filter_codes = [int(c) for c in sys.argv[2:]] if len(sys.argv) > 2 else None
    main(_node_id, _filter_codes)
