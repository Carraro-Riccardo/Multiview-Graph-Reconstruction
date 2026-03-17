"""Image-only multi-view hydraulic reconstruction with structural reconciliation."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from .branch_fragments import (
    serialize_anchor_nodes,
    serialize_branch_fragments,
    serialize_scene_branches,
)
from .candidate_scene_graph import build_candidate_graph, serialize_candidate_edges
from config import IMAGES_DIR, MIN_IMAGES, SKIP_NODES, WORK_DIR
from core.debug_utils import save_json
from core.detection import discover_nodes
from .graph_refinement import force_spanning_tree, refine_candidate_graph, serialize_fused_graph
from .multiview_fusion import fuse_multiview_scene
from .node_reid import identify_scene_nodes, serialize_scene_nodes
from .observed_graphs import build_all_observed_graphs, serialize_observed_graphs
from .structural_reconciliation import reconcile_shortcut_edges

_W = 72   # separator width


def _sep(char="─"):
    print(f"  {'─' * _W}" if char == "─" else f"  {'═' * _W}")


def _output_dir(node_id: int) -> Path:
    return WORK_DIR / "multiview" / f"node{node_id}"


def run_multiview_pipeline(node_id: int, image_paths: list, output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = _output_dir(node_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * (_W + 4)}")
    print(f"  NODE {node_id}  —  {len(image_paths)} views  —  {output_dir}")
    print(f"{'═' * (_W + 4)}")

    # ── Step 1: per-view observed graphs ─────────────────────────────────────
    print(f"\n  [Step 1] Per-view detection & local graph hypotheses")
    _sep()
    observed_graphs = build_all_observed_graphs(image_paths)
    if not observed_graphs:
        print(f"  [node {node_id}] ERROR: no valid detections in any view — skipping")
        return {"status": "error", "reason": "no_detections", "node_id": node_id}

    total_obs = sum(len(og.nodes) for og in observed_graphs)
    total_frg = sum(len(og.branch_fragments) for og in observed_graphs)
    print(f"  views accepted : {len(observed_graphs)}/{len(image_paths)}")
    print(f"  total detections (post-filter) : {total_obs}")
    print(f"  total branch fragments         : {total_frg}")

    observed_json = serialize_observed_graphs(observed_graphs)
    save_json(output_dir / "observed_graphs.json", observed_json)
    save_json(output_dir / "local_observed_graphs.json", observed_json)

    all_branch_fragments = [
        fragment
        for og in observed_graphs
        for fragment in og.branch_fragments
    ]
    save_json(output_dir / "branch_fragments.json", {
        "n_branch_fragments": len(all_branch_fragments),
        "branch_fragments": serialize_branch_fragments(all_branch_fragments),
    })

    # ── Step 2: node re-identification ───────────────────────────────────────
    print(f"\n  [Step 2] Node re-identification (multi-seed + co-association + soft-merge)")
    _sep()
    scene_nodes, reid_debug = identify_scene_nodes(observed_graphs, image_paths=image_paths)
    save_json(output_dir / "scene_nodes.json", serialize_scene_nodes(scene_nodes))
    save_json(output_dir / "reid_debug.json", reid_debug)

    # Print scene node table
    col = 16
    print(f"  {'scene_node_id':<18}{'name':<{col}}{'code':>6}{'supp':>6}{'views':>7}{'conf':>7}{'anchor':>8}")
    _sep()
    for sn in sorted(scene_nodes, key=lambda s: (s.fused_code, s.scene_node_id)):
        print(
            f"  {sn.scene_node_id:<18}{sn.fused_name:<{col}}"
            f"{sn.fused_code:>6}{sn.support_count:>6}{len(sn.support_images):>7}"
            f"{sn.fused_confidence:>7.2f}{sn.anchor_hint_score:>8.2f}"
        )
    _sep()
    print(f"  {len(scene_nodes)} scene nodes from {total_obs} observations")

    # ── Step 3: fragment fusion & anchor identification ───────────────────────
    print(f"\n  [Step 3] Fragment fusion and anchor identification")
    _sep()
    fusion_state, fusion_debug = fuse_multiview_scene(scene_nodes, observed_graphs)
    anchor_nodes   = fusion_state["anchor_nodes"]
    scene_branches = fusion_state["scene_branches"]
    print(f"  anchor nodes   : {len(anchor_nodes)}")
    print(f"  scene branches : {len(scene_branches)}")

    save_json(output_dir / "anchor_nodes.json", {
        "n_anchor_nodes": len(anchor_nodes),
        "anchor_nodes":   serialize_anchor_nodes(anchor_nodes),
    })
    save_json(output_dir / "scene_branches.json", {
        "n_scene_branches":            len(scene_branches),
        "scene_branches":              serialize_scene_branches(scene_branches),
        "branch_reid_debug":           fusion_debug["branch_reid"],
        "anchor_attachment_hypotheses": fusion_debug["attachment_hypotheses"],
    })

    # ── Step 4: candidate scene graph ────────────────────────────────────────
    print(f"\n  [Step 4] Candidate scene graph")
    _sep()
    candidate_edges, candidate_debug = build_candidate_graph(
        scene_nodes, observed_graphs, anchor_nodes, scene_branches,
    )
    print(f"  candidate edges : {len(candidate_edges)}")
    save_json(output_dir / "candidate_scene_graph.json", {
        "scene_nodes":     serialize_scene_nodes(scene_nodes),
        "anchor_nodes":    serialize_anchor_nodes(anchor_nodes),
        "scene_branches":  serialize_scene_branches(scene_branches),
        "candidate_edges": serialize_candidate_edges(candidate_edges),
        "debug":           candidate_debug,
    })

    # ── Step 5: structural reconciliation ────────────────────────────────────
    print(f"\n  [Step 5] Structural reconciliation of shortcut edges")
    _sep()
    candidate_edges, reconciliation_debug = reconcile_shortcut_edges(scene_nodes, candidate_edges)
    n_weakened = sum(1 for e in candidate_edges if getattr(e, "weakened_by_reconciliation", False))
    print(f"  edges after reconciliation : {len(candidate_edges)}  (weakened: {n_weakened})")
    save_json(output_dir / "structural_reconciliation.json", reconciliation_debug)

    # ── Step 6: global refinement ────────────────────────────────────────────
    print(f"\n  [Step 6] Global refinement and pruning")
    _sep()
    fused_graph, refinement_debug = refine_candidate_graph(
        scene_nodes, anchor_nodes, scene_branches,
        candidate_edges, len(observed_graphs),
    )
    save_json(output_dir / "graph_refinement_debug.json", refinement_debug)

    # ── Step 7: force spanning tree ───────────────────────────────────────────
    print(f"\n  [Step 7] Spanning-tree forcing")
    _sep()
    fused_graph, span_debug = force_spanning_tree(fused_graph, candidate_edges)
    save_json(output_dir / "spanning_tree_debug.json", span_debug)
    save_json(output_dir / "fused_graph.json", serialize_fused_graph(fused_graph, len(observed_graphs)))

    n_forced   = span_debug["n_forced"]
    print(f"  forced edges added : {n_forced}")
    for fe in span_debug["forced_edges"]:
        via = fe["via"]
        print(f"    {fe['a']}  ←forced({via})→  {fe['b']}")

    # ── Final graph summary ───────────────────────────────────────────────────
    n_isolated = sum(1 for nid in fused_graph.nodes() if fused_graph.degree(nid) == 0)
    connected  = nx.is_connected(fused_graph) if fused_graph.number_of_nodes() > 0 else True

    print(f"\n  {'─' * _W}")
    print(f"  FINAL GRAPH  —  node {node_id}")
    print(f"  {'─' * _W}")
    print(f"  nodes: {fused_graph.number_of_nodes()}   edges: {fused_graph.number_of_edges()}   "
          f"isolated: {n_isolated}   {'CONNECTED' if connected else 'DISCONNECTED'}")
    print(f"  {'─' * _W}")
    print(f"  {'node_id':<18}{'name':<16}{'code':>6}{'deg':>5}{'val':>5}  support_images")
    print(f"  {'─' * _W}")
    for nid in sorted(fused_graph.nodes(), key=lambda x: (fused_graph.nodes[x].get("code", 0), x)):
        nd       = fused_graph.nodes[nid]
        deg      = fused_graph.degree(nid)
        val      = nd.get("valence", nd.get("max_degree", "?"))
        name     = nd.get("name", nd.get("fused_name", "?"))
        code     = nd.get("code", nd.get("fused_code", "?"))
        imgs     = ", ".join(nd.get("support_images", []))
        flag     = " !" if deg == 0 else ""
        print(f"  {nid:<18}{name:<16}{str(code):>6}{deg:>5}{str(val):>5}  {imgs}{flag}")
    print(f"  {'─' * _W}")
    print(f"  output: {output_dir}")

    return {
        "status":            "ok",
        "node_id":           node_id,
        "n_views":           len(observed_graphs),
        "n_scene_nodes":     len(scene_nodes),
        "n_anchor_nodes":    len(anchor_nodes),
        "n_branch_fragments": len(all_branch_fragments),
        "n_scene_branches":  len(scene_branches),
        "n_candidate_edges": len(candidate_edges),
        "n_final_nodes":     fused_graph.number_of_nodes(),
        "n_final_edges":     fused_graph.number_of_edges(),
        "n_forced_edges":    n_forced,
        "n_isolated":        n_isolated,
        "connected":         connected,
        "output_dir":        str(output_dir),
    }


def run_multiview_all(images_dir: Path | None = None) -> dict:
    if images_dir is None:
        images_dir = IMAGES_DIR

    eligible = {
        node_id: image_paths
        for node_id, image_paths in discover_nodes(images_dir).items()
        if node_id not in SKIP_NODES and len(image_paths) >= MIN_IMAGES
    }
    node_ids = sorted(eligible)
    print(f"\nNodes to process: {node_ids}")

    results = {}
    for node_id, image_paths in sorted(eligible.items()):
        results[node_id] = run_multiview_pipeline(node_id, image_paths)

    # ── Cross-node summary ────────────────────────────────────────────────────
    ok_results = [r for r in results.values() if r.get("status") == "ok"]
    if ok_results:
        print(f"\n\n{'═' * (_W + 4)}")
        print(f"  OVERALL SUMMARY  —  {len(ok_results)}/{len(results)} nodes completed")
        print(f"{'═' * (_W + 4)}")
        hdr = f"  {'node':>5}{'views':>7}{'scene_n':>9}{'cand_e':>8}{'final_n':>9}{'final_e':>9}{'forced':>8}{'iso':>5}  conn"
        print(hdr)
        print(f"  {'─' * (_W + 2)}")
        for r in sorted(ok_results, key=lambda x: x["node_id"]):
            conn = "Y" if r["connected"] else "N"
            print(
                f"  {r['node_id']:>5}"
                f"{r['n_views']:>7}"
                f"{r['n_scene_nodes']:>9}"
                f"{r['n_candidate_edges']:>8}"
                f"{r['n_final_nodes']:>9}"
                f"{r['n_final_edges']:>9}"
                f"{r.get('n_forced_edges', 0):>8}"
                f"{r['n_isolated']:>5}  {conn}"
            )
        print(f"  {'─' * (_W + 2)}")

    return results


if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    run_multiview_all()
