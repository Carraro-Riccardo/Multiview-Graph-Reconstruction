"""Visualize the fused graph for a specific node.

Layout: kamada-kawai weighted by edge support_count, so nodes with strong
evidence cluster naturally. Forced / structural-fallback edges are clearly
de-emphasised (dashed, faded, thin).

Nodes are coloured by component code (same code → same hue), with opacity
scaled by support_count / n_views so well-supported nodes pop.

Usage:
    python visualize.py            # uses NODE_ID defined below
    python visualize.py 9          # specific node
    python visualize.py 2 7 9      # multiple nodes
    python visualize.py --all      # all reconstructed nodes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_reconstruction.config import WORK_DIR as _WORK_DIR  # noqa: E402
from evaluation.standard_comparison.visualizza_best_match import generate_best_match_report  # noqa: E402
from json_adapter_to_qgis import convert_fused_graph_to_qgis  # noqa: E402

# ── CONFIG ───────────────────────────────────────────────────────────────────
NODE_ID     = 2
WORK_DIR    = _WORK_DIR / "multiview"
SHOW_WINDOW = False
# ─────────────────────────────────────────────────────────────────────────────

# Palette for component codes (up to 20 distinct codes per scene)
_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#cddc39",
    "#ff5722", "#607d8b", "#795548", "#8bc34a", "#ff9800",
    "#673ab7", "#009688", "#f44336", "#2196f3", "#4caf50",
]


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_fused_graph(node_id: str) -> tuple[nx.Graph, dict]:
    json_path = WORK_DIR / f"node{node_id}" / "fused_graph.json"
    if not json_path.exists():
        raise FileNotFoundError(f"fused_graph.json not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    G = nx.Graph()
    for n in data["nodes"]:
        G.add_node(n["id"], **n)
    for e in data["edges"]:
        G.add_edge(e["node_a"], e["node_b"], **e)
    return G, data


# ── Layout ────────────────────────────────────────────────────────────────────

def _weighted_layout(G: nx.Graph) -> dict[str, tuple[float, float]]:
    """
    Kamada-Kawai layout on a copy of G where each edge has a distance
    inversely proportional to its support_count (strong evidence → closer).
    Forced / structural-fallback edges get a very long distance so they
    spread apart in the layout.

    Isolated nodes (degree 0) are placed in a row at the bottom.
    """
    if G.number_of_nodes() == 0:
        return {}

    # Build weighted graph for layout
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, a in G.edges(data=True):
        sc = a.get("support_count", 0)
        forced = a.get("forced", False)
        reason = a.get("forced_reason", "") or ""
        if forced and "structural_fallback" in reason:
            dist = 8.0
        elif forced:
            dist = 4.0
        elif sc <= 0:
            dist = 3.0
        else:
            dist = 1.0 / (sc + 0.5)
        H.add_edge(u, v, weight=dist)

    # Separate isolated nodes
    isolated  = [n for n in G.nodes() if G.degree(n) == 0]
    connected = [n for n in G.nodes() if G.degree(n) > 0]

    pos: dict[str, tuple[float, float]] = {}

    if connected:
        sub = H.subgraph(connected)
        n_conn = len(connected)
        scale  = max(3.0, 0.8 * n_conn)
        try:
            sub_pos = nx.kamada_kawai_layout(sub, weight="weight", scale=scale)
        except Exception:
            sub_pos = nx.spring_layout(sub, weight="weight",
                                       k=2.5 / max(n_conn ** 0.5, 1), seed=42,
                                       iterations=200, scale=scale)
        # Nudge overlapping nodes apart with a short spring pass
        sub_pos = nx.spring_layout(sub, pos=sub_pos, fixed=None,
                                   k=1.5 / max(n_conn ** 0.5, 1),
                                   iterations=30, scale=scale, seed=0)
        pos.update(sub_pos)

    # Place isolated nodes in a row below
    if isolated:
        ys = [p[1] for p in pos.values()] if pos else [0.0]
        y_bottom = min(ys) - 1.5
        x_start  = 0.0
        for i, n in enumerate(isolated):
            pos[n] = (x_start + i * 1.5, y_bottom)

    return pos


# ── Color helpers ─────────────────────────────────────────────────────────────

def _build_code_color_map(G: nx.Graph) -> dict[int, str]:
    codes = sorted({G.nodes[n].get("code", 0) for n in G.nodes()})
    return {code: _PALETTE[i % len(_PALETTE)] for i, code in enumerate(codes)}


def _node_face(G: nx.Graph, nid: str, code_colors: dict[int, str],
               n_views: int) -> tuple[str, float]:
    """Return (hex_color, alpha) for a node face."""
    code  = G.nodes[nid].get("code", 0)
    color = code_colors.get(code, "#888888")
    supp  = G.nodes[nid].get("support_count", 0)
    alpha = 0.45 + 0.55 * min(supp / max(n_views, 1), 1.0)
    return color, alpha


def _edge_style(attrs: dict) -> tuple[str, float, float, str]:
    """Return (color, width, alpha, linestyle) for an edge."""
    sc     = attrs.get("support_count", 0)
    forced = attrs.get("forced", False)
    reason = attrs.get("forced_reason", "") or ""
    if not forced:
        width = 1.5 + 2.0 * min(sc / 7.0, 1.0)  # thicker = more support
        return "#ecf0f1", width, 0.85, "solid"
    if "structural_fallback" in reason:
        return "#555555", 1.0, 0.35, "dashed"
    return "#e67e22", 1.2, 0.55, "dashed"


def _edge_label(attrs: dict, n_views: int) -> str:
    sc = attrs.get("support_count", 0)
    if attrs.get("forced", False):
        return ""   # don't clutter with forced labels
    if sc == 0:
        return ""
    cs = attrs.get("consistency_score")
    if cs is not None:
        return f"{sc}/{n_views} {cs:.0%}"
    return f"{sc}/{n_views}"


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_node_graph(node_id: str) -> Path:
    G, data   = load_fused_graph(node_id)
    n_views   = data.get("total_images", 1)
    n_nodes   = G.number_of_nodes()
    n_edges   = G.number_of_edges()
    n_forced  = sum(1 for _, _, a in G.edges(data=True) if a.get("forced"))
    n_fallback = sum(1 for _, _, a in G.edges(data=True)
                     if a.get("forced") and "structural_fallback" in (a.get("forced_reason") or ""))
    n_observed = n_edges - n_forced
    connected = nx.is_connected(G) if n_nodes > 0 else True

    code_colors = _build_code_color_map(G)
    pos = _weighted_layout(G)

    # ── Draw edges by category ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    for u, v, a in G.edges(data=True):
        color, width, alpha, style = _edge_style(a)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=color, width=width, alpha=alpha, style=style,
        )

    # Edge labels (only observed edges with support)
    edge_labels = {
        (u, v): _edge_label(a, n_views)
        for u, v, a in G.edges(data=True)
        if _edge_label(a, n_views)
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=6, font_color="#cccccc",
        bbox={"alpha": 0},
    )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for nid in G.nodes():
        color, alpha = _node_face(G, nid, code_colors, n_views)
        supp  = G.nodes[nid].get("support_count", 0)
        size  = 600 + 150 * G.degree(nid) + 60 * supp

        # Border: gold if full support, grey otherwise
        border = "#f1c40f" if supp >= n_views else "#444444"
        border_w = 2.5 if supp >= n_views else 1.2

        # Convert hex+alpha to RGBA for draw
        rgb  = mcolors.to_rgb(color)
        rgba = [*rgb, alpha]

        nx.draw_networkx_nodes(
            G, pos, nodelist=[nid], ax=ax,
            node_color=[rgba], node_size=[size],
            linewidths=border_w, edgecolors=border,
        )

    # ── Node labels ───────────────────────────────────────────────────────────
    node_labels = {}
    for nid in G.nodes():
        name = G.nodes[nid].get("name", "?")
        supp = G.nodes[nid].get("support_count", 0)
        deg  = G.degree(nid)
        node_labels[nid] = f"{nid}\n{name}\n{supp}/{n_views}  d={deg}"

    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=6.5, font_color="white")

    # ── Legend ────────────────────────────────────────────────────────────────
    code_patches = [
        mpatches.Patch(color=c, label=f"[{code}] {_code_name(G, code)}")
        for code, c in sorted(code_colors.items())
    ]
    edge_patches = [
        mpatches.Patch(color="#ecf0f1", label=f"Observed edge (thickness ∝ support)"),
        mpatches.Patch(color="#e67e22", label="Forced bridge"),
        mpatches.Patch(color="#555555", label="Structural fallback (spanning tree)"),
        mpatches.Patch(color="#f1c40f", label="Full-support node border"),
    ]
    leg1 = ax.legend(handles=code_patches, loc="upper left",
                     framealpha=0.7, facecolor="#1a1a2e",
                     labelcolor="white", fontsize=7.5, title="Components",
                     title_fontsize=8)
    leg1.get_title().set_color("white")
    ax.add_artist(leg1)
    ax.legend(handles=edge_patches, loc="lower left",
              framealpha=0.7, facecolor="#1a1a2e",
              labelcolor="white", fontsize=7.5)

    # ── Title ─────────────────────────────────────────────────────────────────
    conn_str = "CONNECTED" if connected else "DISCONNECTED"
    ax.set_title(
        f"Fused Graph — node {node_id}  |  "
        f"{n_nodes} nodes  {n_edges} edges  "
        f"({n_observed} observed · {n_forced - n_fallback} forced · {n_fallback} fallback)  "
        f"{conn_str}  ({n_views} views)",
        color="white", fontsize=10, pad=10,
    )
    ax.axis("off")
    plt.tight_layout()

    out = WORK_DIR / f"node{node_id}" / "fused_graph_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if SHOW_WINDOW:
        plt.show()
    plt.close()
    print(f"[node{node_id}] saved: {out.resolve()}")
    return out


def _code_name(G: nx.Graph, code: int) -> str:
    for nid in G.nodes():
        if G.nodes[nid].get("code") == code:
            return G.nodes[nid].get("name", f"code_{code}")
    return f"code_{code}"


def _available_node_ids() -> list[str]:
    def sort_key(node_id: str):
        return [int(part) if part.isdigit() else part for part in node_id.split("_")]

    return sorted(
        (
            path.name[len("node"):]
            for path in WORK_DIR.glob("node*")
            if path.is_dir() and (path / "fused_graph.json").exists()
        ),
        key=sort_key,
    )


def main() -> None:
    args = sys.argv[1:]
    node_ids = _available_node_ids() if args == ["--all"] else (args or [str(NODE_ID)])
    if not node_ids:
        raise SystemExit(f"No fused_graph.json found under {WORK_DIR}")

    for nid in node_ids:
        plot_node_graph(nid)
        node_dir = WORK_DIR / f"node{nid}"
        convert_fused_graph_to_qgis(
            node_dir / "fused_graph.json",
            node_dir,
            geometry_source="fake",
        )

    report_path = generate_best_match_report(
        pipeline_dir=WORK_DIR,
        output_dir=_WORK_DIR / "standard_comparison",
        node_ids=node_ids,
    )
    if report_path:
        print(f"[best-match] saved: {report_path.resolve()}")
    else:
        print("[best-match] report not generated: no comparable standard graph found")


if __name__ == "__main__":
    main()
