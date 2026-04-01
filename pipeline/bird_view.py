"""Bird's-eye view layout of a reconstructed hydraulic scene.

Uses the VGGT 3D centroids of scene nodes, projects them onto the dominant
plane via PCA, then draws the fused graph on top.

Entry point: render_bird_view(output_dir) → saves bird_view.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import COMPONENT_INFO


# ── Colour palette (one colour per component code, auto-generated) ─────────────
_CMAP = plt.get_cmap("tab20")

def _code_color(code: int) -> tuple:
    return _CMAP(code % 20)


# ── PCA projection ──────────────────────────────────────────────────────────────

def _project_to_plane(centroids: np.ndarray) -> np.ndarray:
    """Project Nx3 centroids onto their dominant 2D plane via PCA.

    Returns Nx2 coordinates in the plane frame (metres, origin = centroid mean).
    """
    mean = centroids.mean(axis=0)
    X = centroids - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # First two right-singular vectors span the dominant plane
    return X @ Vt[:2].T          # Nx2


# ── Main renderer ───────────────────────────────────────────────────────────────

def render_bird_view(output_dir: Path) -> Path | None:
    """Load saved JSON artifacts and render a bird's-eye PNG.

    Returns the path to the saved PNG, or None if data is missing.
    """
    output_dir = Path(output_dir)

    scene_nodes_path   = output_dir / "scene_nodes.json"
    fused_graph_path   = output_dir / "fused_graph.json"
    vggt_debug_path    = output_dir / "vggt_edge_debug.json"

    if not (scene_nodes_path.exists() and fused_graph_path.exists() and vggt_debug_path.exists()):
        return None

    scene_nodes  = json.load(scene_nodes_path.open())
    fused_graph  = json.load(fused_graph_path.open())
    vggt_debug   = json.load(vggt_debug_path.open())
    scene_geom   = vggt_debug.get("scene_geometry", {})

    if not scene_geom:
        return None

    # ── Collect node positions ─────────────────────────────────────────────────
    node_ids    = list(scene_geom.keys())
    centroids3d = np.array([scene_geom[n]["centroid"][:3] for n in node_ids])

    if len(centroids3d) < 2:
        return None

    coords2d = _project_to_plane(centroids3d)   # Nx2 in metres
    pos = {nid: coords2d[i] for i, nid in enumerate(node_ids)}

    # Build lookup: scene_node_id → fused_code, fused_name
    meta: dict[str, dict] = {}
    for sn in scene_nodes:
        meta[sn["scene_node_id"]] = {
            "code": sn["fused_code"],
            "name": sn["fused_name"],
            "support": sn["support_count"],
        }

    # ── Draw ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f0")
    fig.patch.set_facecolor("#f5f5f0")

    # Edges
    edges = fused_graph.get("edges", [])
    for edge in edges:
        a, b = edge["node_a"], edge["node_b"]
        if a in pos and b in pos:
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.plot([xa, xb], [ya, yb],
                    color="#888888", linewidth=1.5, zorder=1, solid_capstyle="round")

    # Nodes
    for nid, (x, y) in pos.items():
        m = meta.get(nid, {})
        code  = m.get("code", 0)
        name  = m.get("name", nid)
        sup   = m.get("support", 1)
        color = _code_color(code)

        radius = 0.012 + 0.003 * min(sup, 8)   # larger = more views
        circle = mpatches.Circle((x, y), radius, color=color,
                                  ec="white", linewidth=1.2, zorder=3)
        ax.add_patch(circle)

        # Short label: code#instance  (e.g. "202#1")
        short = f"{code}#{nid.split('#')[-1]}" if "#" in nid else nid
        ax.text(x, y + radius + 0.008, short,
                ha="center", va="bottom", fontsize=6.5,
                color="#222222", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    # ── Legend (unique codes present) ─────────────────────────────────────────
    seen_codes: dict[int, str] = {}
    for m in meta.values():
        c = m["code"]
        if c not in seen_codes:
            seen_codes[c] = m["name"]

    legend_handles = [
        mpatches.Patch(color=_code_color(c), label=f"{c} – {n}")
        for c, n in sorted(seen_codes.items())
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=7, framealpha=0.85, ncol=2,
              title="Component codes", title_fontsize=8)

    # Axis labels and title
    node_id = output_dir.name   # e.g. "node4"
    ax.set_xlabel("PCA axis 1  [m]", fontsize=9)
    ax.set_ylabel("PCA axis 2  [m]", fontsize=9)
    ax.set_title(f"Bird's-eye layout — {node_id}   "
                 f"({len(node_ids)} components, {len(edges)} connections)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / "bird_view.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
