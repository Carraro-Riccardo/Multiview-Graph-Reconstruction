"""Debug visualization: crops per scene node.

Per ogni scene node mostra i crop delle detection da ogni immagine di supporto,
affiancati con il loro GT match (se fornito da evaluate.py).

Uso:
    python debug_crops.py 3            # solo node 3
    python debug_crops.py 2 3 9        # più nodi

Output: output/multiview/nodeX/debug_crops.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
NODE_ID      = 2
WORK_DIR     = Path("./output/multiview")
CACHE_DIR    = Path("./output/detection_cache")
IMAGES_DIR   = Path("C:/Users/ricca/Desktop/images/val")
PADDING      = 0.04          # bbox padding fraction
MAX_COLS     = 8             # max views per row
THUMB_SIZE   = 96            # pixel size of each crop thumbnail
# ─────────────────────────────────────────────────────────────────────────────

try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from config import DIAMETER_CODES, IMAGES_DIR as _CFG_IMAGES_DIR
    IMAGES_DIR = _CFG_IMAGES_DIR
except Exception:
    DIAMETER_CODES = set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_filtered_dets(image_stem: str) -> list[dict]:
    """Load accepted non-diameter detections (same filter as pipeline)."""
    p = CACHE_DIR / f"{image_stem}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        d for d in data.get("detections", [])
        if d.get("accepted", True)
        and not d.get("is_diameter", d.get("code") in DIAMETER_CODES)
    ]


def _find_image(image_stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = Path(IMAGES_DIR) / f"{image_stem}{ext}"
        if p.exists():
            return p
    return None


def _crop_det(image_stem: str, det_idx: int) -> np.ndarray | None:
    """Return an RGBA crop for the detection, or None if not found."""
    img_path = _find_image(image_stem)
    if img_path is None:
        return None
    dets = _load_filtered_dets(image_stem)
    if det_idx >= len(dets):
        return None
    det  = dets[det_idx]
    bbox = det["bbox_norm"]   # [x1, y1, x2, y2] normalised
    img  = Image.open(img_path).convert("RGBA")
    W, H = img.size
    pad  = PADDING
    x1   = max(0, (bbox[0] - pad) * W)
    y1   = max(0, (bbox[1] - pad) * H)
    x2   = min(W, (bbox[2] + pad) * W)
    y2   = min(H, (bbox[3] + pad) * H)
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    # Resize to fixed thumbnail
    size = THUMB_SIZE
    crop.thumbnail((size, size), Image.LANCZOS)
    arr  = np.array(crop)
    return arr


def _parse_obs_id(obs_id: str) -> tuple[str, int]:
    """'node3_0005_3' → ('node3_0005', 3)"""
    parts = obs_id.rsplit("_", 1)
    return parts[0], int(parts[1])


# ── Per-code color palette ────────────────────────────────────────────────────

def _code_color(code: int) -> str:
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
    ]
    return palette[code % len(palette)]


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_debug_crops(node_id: int) -> Path:
    scene_path = WORK_DIR / f"node{node_id}" / "scene_nodes.json"
    if not scene_path.exists():
        print(f"[node{node_id}] scene_nodes.json not found — skip")
        return None

    scene_nodes = json.loads(scene_path.read_text(encoding="utf-8"))

    # Sort by code then scene_node_id for readability
    scene_nodes.sort(key=lambda n: (n["fused_code"], n["scene_node_id"]))

    n_nodes = len(scene_nodes)
    # Each scene node occupies one row; columns = support views (up to MAX_COLS)
    max_views_per_node = max(
        (len(sn["supporting_observed_ids"]) for sn in scene_nodes), default=1
    )
    n_cols = min(max_views_per_node, MAX_COLS) + 1  # +1 for label column

    cell_h  = THUMB_SIZE / 96 * 1.3   # inches
    cell_w  = THUMB_SIZE / 96 * 1.3
    label_w = 2.8

    fig_w = label_w + (n_cols - 1) * cell_w + 0.5
    fig_h = n_nodes * cell_h + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")
    fig.suptitle(
        f"Debug crops — node {node_id}  ({n_nodes} scene nodes)",
        color="white", fontsize=11, y=0.99,
    )

    # Build a grid: n_nodes rows × n_cols cols
    gs = GridSpec(
        n_nodes, n_cols,
        figure=fig,
        left=label_w / fig_w,
        right=0.99,
        top=0.96,
        bottom=0.02,
        hspace=0.08,
        wspace=0.04,
    )

    legend_patches: dict[int, mpatches.Patch] = {}

    for row, sn in enumerate(scene_nodes):
        code  = sn["fused_code"]
        name  = sn["fused_name"]
        snid  = sn["scene_node_id"]
        obs   = sn["supporting_observed_ids"]
        color = _code_color(code)

        # Label on the left (outside the GridSpec cells — use fig.text)
        label_x = 0.01
        label_y_frac = (n_nodes - row - 0.5) / n_nodes
        label_y = 0.02 + label_y_frac * (fig_h - 1.0) / fig_h
        fig.text(
            label_x, label_y,
            f"{snid}\n{name}  [{code}]\n{len(obs)} views",
            color=color, fontsize=6.5, va="center",
            fontweight="bold",
        )

        for col_i, obs_id in enumerate(obs[:MAX_COLS]):
            img_stem, det_idx = _parse_obs_id(obs_id)
            crop = _crop_det(img_stem, det_idx)

            ax = fig.add_subplot(gs[row, col_i])
            ax.set_facecolor("#0d1117")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

            if crop is not None:
                ax.imshow(crop)
            else:
                ax.text(0.5, 0.5, "?", color="gray", ha="center", va="center",
                        fontsize=10, transform=ax.transAxes)

            # Caption: image stem (last part)
            short_stem = img_stem.split("_")[-1]
            ax.set_xlabel(short_stem, fontsize=4.5, color="#aaaaaa", labelpad=1)

        if code not in legend_patches:
            legend_patches[code] = mpatches.Patch(color=color, label=f"[{code}] {name}")

    # Legend at the bottom
    fig.legend(
        handles=list(legend_patches.values()),
        loc="lower center",
        ncol=min(len(legend_patches), 6),
        fontsize=7,
        framealpha=0.5,
        facecolor="#1a1a2e",
        labelcolor="white",
        bbox_to_anchor=(0.5 + label_w / fig_w / 2, 0.0),
    )

    out = WORK_DIR / f"node{node_id}" / "debug_crops.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[node{node_id}] saved: {out.resolve()}")
    return out


if __name__ == "__main__":
    node_ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [NODE_ID]
    for nid in node_ids:
        plot_debug_crops(nid)
