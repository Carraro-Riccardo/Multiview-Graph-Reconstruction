"""Image-only multi-view hydraulic reconstruction - direct adjacency pipeline."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from config import IMAGES_DIR, MIN_IMAGES, SKIP_NODES, VGGT_EDGE_ENABLED, WORK_DIR
from core.debug_utils import save_json
from core.detection import discover_nodes
from .direct_graph_builder import build_direct_graph
from .graph_refinement import force_spanning_tree, serialize_fused_graph
from .node_reid import identify_scene_nodes, serialize_scene_nodes
from .observed_graphs import build_all_observed_graphs
from .spatial_consistency import run_consistency_pass
from .vggt_3d_edges import build_vggt_edge_hints, build_vggt_search_roots

_W = 72


def _sep(char="-"):
    fill = "-" if char == "-" else "="
    print(f"  {fill * _W}")


def _output_dir(node_id: int) -> Path:
    return WORK_DIR / "multiview" / f"node{node_id}"


def run_multiview_pipeline(node_id: int, image_paths: list, output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = _output_dir(node_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * (_W + 4)}")
    print(f"  NODE {node_id}  -  {len(image_paths)} views  -  {output_dir}")
    print(f"{'=' * (_W + 4)}")

    print("\n  [Step 1] Per-view detection & local graph hypotheses")
    _sep()
    observed_graphs = build_all_observed_graphs(image_paths)
    if not observed_graphs:
        print(f"  [node {node_id}] ERROR: no valid detections in any view - skipping")
        return {"status": "error", "reason": "no_detections", "node_id": node_id}

    total_obs = sum(len(og.nodes) for og in observed_graphs)
    print(f"  views accepted : {len(observed_graphs)}/{len(image_paths)}")
    print(f"  total detections (post-filter) : {total_obs}")

    print("\n  [Step 2] Node re-identification (multi-seed + co-association + soft-merge)")
    _sep()
    scene_nodes, _ = identify_scene_nodes(observed_graphs, image_paths=image_paths)
    save_json(output_dir / "scene_nodes.json", serialize_scene_nodes(scene_nodes))

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

    vggt_edge_payload = {"enabled": False, "pair_hints": {}}
    if VGGT_EDGE_ENABLED:
        search_roots = build_vggt_search_roots(node_id, output_dir, WORK_DIR)
        vggt_edge_payload = build_vggt_edge_hints(scene_nodes, observed_graphs, search_roots)
        save_json(
            output_dir / "vggt_edge_debug.json",
            {
                "enabled": vggt_edge_payload.get("enabled", False),
                "search_roots": vggt_edge_payload.get("search_roots", []),
                "n_loaded_images": vggt_edge_payload.get("n_loaded_images", 0),
                "n_scene_centroids": vggt_edge_payload.get("n_scene_centroids", 0),
                "scene_cutoff": vggt_edge_payload.get("scene_cutoff"),
                "loaded_images": vggt_edge_payload.get("loaded_images", {}),
                "skipped_images": vggt_edge_payload.get("skipped_images", []),
                "pair_hints": vggt_edge_payload.get("pair_hints_serialized", []),
            },
        )
        print(
            f"  [VGGT] loaded_views={vggt_edge_payload.get('n_loaded_images', 0)}/"
            f"{len(observed_graphs)}  scene_centroids={vggt_edge_payload.get('n_scene_centroids', 0)}  "
            f"edge_hints={len(vggt_edge_payload.get('pair_hints', {}))}"
        )

    print("\n  [Step 3] Direct per-image adjacency graph")
    _sep()
    fused_graph, bridge_edges, direct_debug = build_direct_graph(
        scene_nodes,
        observed_graphs,
        len(observed_graphs),
        vggt_edge_hints=vggt_edge_payload.get("pair_hints", {}),
    )
    n_voted = direct_debug["n_pairs_voted"]
    n_shortcuts = direct_debug["n_shortcut_suppressed"]
    print(f"  voted pairs={n_voted}  shortcuts suppressed={n_shortcuts}  edges={fused_graph.number_of_edges()}")

    print("\n  [Step 4] Spanning-tree forcing")
    _sep()
    fused_graph, span_debug = force_spanning_tree(fused_graph, bridge_edges)
    n_forced = span_debug["n_forced"]
    print(f"  forced edges added : {n_forced}")
    for fe in span_debug["forced_edges"]:
        print(f"    {fe['a']}  --({fe['via']})->  {fe['b']}")

    print("\n  [Step 5] Spatial consistency pruning")
    _sep()
    fused_graph, consistency_debug = run_consistency_pass(
        scene_nodes, observed_graphs, fused_graph, bridge_edges
    )
    n_pruned = consistency_debug.get("n_pruned", 0)
    print(f"  pruned={n_pruned}  swapped={consistency_debug.get('n_swapped', 0)}")

    save_json(output_dir / "fused_graph.json", serialize_fused_graph(fused_graph, len(observed_graphs)))

    n_isolated = sum(1 for nid in fused_graph.nodes() if fused_graph.degree(nid) == 0)
    connected = nx.is_connected(fused_graph) if fused_graph.number_of_nodes() > 0 else True

    print(f"\n  {'-' * _W}")
    print(f"  FINAL GRAPH  -  node {node_id}")
    print(f"  {'-' * _W}")
    print(
        f"  nodes: {fused_graph.number_of_nodes()}   edges: {fused_graph.number_of_edges()}   "
        f"isolated: {n_isolated}   {'CONNECTED' if connected else 'DISCONNECTED'}"
    )
    print(f"  {'-' * _W}")
    print(f"  {'node_id':<18}{'name':<16}{'code':>6}{'deg':>5}{'val':>5}  support_images")
    print(f"  {'-' * _W}")
    for nid in sorted(fused_graph.nodes(), key=lambda x: (fused_graph.nodes[x].get('code', 0), x)):
        nd = fused_graph.nodes[nid]
        deg = fused_graph.degree(nid)
        val = nd.get("valence", nd.get("max_degree", "?"))
        name = nd.get("name", nd.get("fused_name", "?"))
        code = nd.get("code", nd.get("fused_code", "?"))
        imgs = ", ".join(nd.get("support_images", []))
        flag = " !" if deg == 0 else ""
        print(f"  {nid:<18}{name:<16}{str(code):>6}{deg:>5}{str(val):>5}  {imgs}{flag}")
    print(f"  {'-' * _W}")
    print(f"  output: {output_dir}")

    return {
        "status": "ok",
        "node_id": node_id,
        "n_views": len(observed_graphs),
        "n_scene_nodes": len(scene_nodes),
        "n_voted_pairs": n_voted,
        "n_final_nodes": fused_graph.number_of_nodes(),
        "n_final_edges": fused_graph.number_of_edges(),
        "n_forced_edges": n_forced,
        "n_isolated": n_isolated,
        "connected": connected,
        "output_dir": str(output_dir),
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

    ok_results = [r for r in results.values() if r.get("status") == "ok"]
    if ok_results:
        print(f"\n\n{'=' * (_W + 4)}")
        print(f"  OVERALL SUMMARY  -  {len(ok_results)}/{len(results)} nodes completed")
        print(f"{'=' * (_W + 4)}")
        hdr = f"  {'node':>5}{'views':>7}{'scene_n':>9}{'voted_p':>9}{'final_n':>9}{'final_e':>9}{'forced':>8}{'iso':>5}  conn"
        print(hdr)
        print(f"  {'-' * (_W + 2)}")
        for r in sorted(ok_results, key=lambda x: x["node_id"]):
            conn = "Y" if r["connected"] else "N"
            print(
                f"  {r['node_id']:>5}"
                f"{r['n_views']:>7}"
                f"{r['n_scene_nodes']:>9}"
                f"{r.get('n_voted_pairs', 0):>9}"
                f"{r['n_final_nodes']:>9}"
                f"{r['n_final_edges']:>9}"
                f"{r.get('n_forced_edges', 0):>8}"
                f"{r['n_isolated']:>5}  {conn}"
            )
        print(f"  {'-' * (_W + 2)}")

    return results


if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    run_multiview_all()
