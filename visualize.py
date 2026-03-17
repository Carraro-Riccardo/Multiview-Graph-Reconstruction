"""Visualize the fused graph for a specific node.

Usage:
    python visualize.py            # uses NODE_ID defined below
    python visualize.py 9          # specific node
    python visualize.py 2 7 9      # multiple nodes
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — remove if you have a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import json

# ── CONFIG ───────────────────────────────────────────────────────────────────
NODE_ID     = 9          # ← change here
WORK_DIR    = Path("./output/multiview")
SHOW_WINDOW = False      # True if a display is available
# ─────────────────────────────────────────────────────────────────────────────


def load_fused_graph(node_id: int) -> tuple[nx.Graph, dict]:
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


def _node_color(G: nx.Graph, nid: str, n_views: int) -> str:
    if G.degree(nid) == 0:
        return "#e74c3c"
    if G.nodes[nid].get("is_anchor"):
        return "#27ae60"
    return "#2980b9"


def _node_border(G: nx.Graph, nid: str, n_views: int) -> str:
    if G.nodes[nid].get("support_count", 0) >= n_views:
        return "#f1c40f"
    return "#dddddd"


def _edge_color(attrs: dict) -> str:
    if not attrs.get("forced", False):
        return "#ecf0f1"
    reason = attrs.get("forced_reason", "")
    if "structural_fallback" in reason:
        return "#e74c3c"
    return "#e67e22"


def _edge_label(attrs: dict) -> str:
    if attrs.get("forced", False):
        sc = attrs.get("support_count", 0)
        return f"F({sc})" if sc else "F"
    return str(attrs.get("support_count", ""))


def plot_node_graph(node_id: int) -> Path:
    G, data = load_fused_graph(node_id)
    n_views  = data.get("total_images", 1)
    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    n_forced = sum(1 for _, _, a in G.edges(data=True) if a.get("forced"))
    n_iso    = sum(1 for n in G.nodes() if G.degree(n) == 0)
    connected = nx.is_connected(G) if n_nodes > 0 else True

    pos = nx.spring_layout(G, k=2.2, iterations=80, seed=42)

    node_colors   = [_node_color(G, n, n_views)  for n in G.nodes()]
    border_colors = [_node_border(G, n, n_views) for n in G.nodes()]
    node_sizes    = [500 + 120 * G.degree(n) for n in G.nodes()]

    normal_edges   = [(u, v) for u, v, a in G.edges(data=True) if not a.get("forced")]
    bridge_edges   = [(u, v) for u, v, a in G.edges(data=True)
                      if a.get("forced") and "structural_fallback" not in a.get("forced_reason", "")]
    fallback_edges = [(u, v) for u, v, a in G.edges(data=True)
                      if a.get("forced") and "structural_fallback" in a.get("forced_reason", "")]

    node_labels = {}
    for nid in G.nodes():
        nd   = G.nodes[nid]
        name = nd.get("name", "?")
        sc   = nd.get("support_count", 0)
        node_labels[nid] = f"{nid}\n{name}\n[{sc}/{n_views}]"

    edge_labels = {
        (u, v): _edge_label(a)
        for u, v, a in G.edges(data=True)
        if _edge_label(a)
    }

    fig, ax = plt.subplots(figsize=(16, 11), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges,   ax=ax,
                           edge_color="#ecf0f1", width=2.2, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edgelist=bridge_edges,   ax=ax,
                           edge_color="#e67e22", width=1.8, alpha=0.9, style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=fallback_edges, ax=ax,
                           edge_color="#e74c3c", width=1.5, alpha=0.9, style="dashed")

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           linewidths=2.5, edgecolors=border_colors)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=6.5, font_color="white")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=6, font_color="#bbbbbb",
                                 bbox={"alpha": 0})

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="Anchor (valence ≥ 3)"),
        mpatches.Patch(color="#2980b9", label="Normal node"),
        mpatches.Patch(color="#e74c3c", label="Isolated node (degree 0)"),
        mpatches.Patch(color="#f1c40f", label="Supported by all views"),
        mpatches.Patch(color="#ecf0f1", label="Observed edge"),
        mpatches.Patch(color="#e67e22", label="Forced — candidate bridge"),
        mpatches.Patch(color="#e74c3c", label="Forced — structural fallback"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              framealpha=0.7, facecolor="#2a2a4e",
              labelcolor="white", fontsize=8)

    conn_str = "CONNECTED" if connected else "DISCONNECTED"
    ax.set_title(
        f"Fused Graph — node {node_id}  |  "
        f"nodes {n_nodes}  edges {n_edges}  "
        f"forced {n_forced}  isolated {n_iso}  "
        f"{conn_str}  ({n_views} views)",
        color="white", fontsize=11, pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    out = WORK_DIR / f"node{node_id}" / "fused_graph_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if SHOW_WINDOW:
        plt.show()
    plt.close()
    print(f"Saved: {out.resolve()}")
    return out


if __name__ == "__main__":
    node_ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [NODE_ID]
    for nid in node_ids:
        plot_node_graph(nid)
