"""
graph_fusion.py — Multi-view graph fusion for hydraulic node reconstruction.

This module implements a constraint-driven fusion pipeline that combines:
  - A reference graph extracted from a PDF hydraulic diagram (prior)
  - Observed graphs reconstructed from individual camera images

Key design principles:
  1. The reference graph is a *prior*, not ground truth — it can be wrong.
  2. Observed graphs are *partial and noisy* — missing nodes are expected.
  3. Matching is *rule-based* (compatibility constraints), not score-optimised.
  4. All matching decisions are logged for inspection (fusion_debug.json).
  5. Coordinates from the PDF diagram are *never* used for matching.

Pipeline steps:
  1. load_reference_graph   — parse PDF-derived JSON into typed objects
  2. load_observed_graphs   — parse per-image reconstruction JSONs
  3. Compatibility filters  — type, DN, degree (all binary pass/fail)
  4. find_anchor_nodes      — high-distinctiveness nodes to seed matching
  5. candidate_matches      — enumerate compatible reference nodes per observed node
  6. validate_local_structure — check neighborhoods with domain priors
  7. expand_match           — BFS-expand an anchor match into a partial hypothesis
  8. fuse_observations      — accumulate evidence across images into FusionState
  9. update_reference_graph_with_observations — correct the prior
  10. save_fused_graph / save_fusion_debug    — JSON output
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import (
    COOC_COUNTS,
    COMPONENT_INFO,
    DEGREE_SATURATION,
    DIAMETER_CODES,
    FORBIDDEN_PAIRS,
    TRIPLET_COUNTS,
)

logger = logging.getLogger(__name__)

# ── Type mapping: reference-graph string type → set of compatible numeric codes ──
#
# The LLM that generates the reference graph uses a coarse 8-word vocabulary.
# Each entry lists the component codes that can plausibly represent that
# semantic type.  "unknown" has an empty set and is always treated as
# compatible (the LLM was uncertain — do not reject on that basis).

_TYPE_TO_CODES: dict[str, frozenset[int]] = {
    "valve":   frozenset({15, 119}),                           # Saracinesca, Rubinetto
    "tee":     frozenset({116, 255, 201, 83, 85}),             # Tee FL / rid. / BIC, Rampa ab./ind.
    "cross":   frozenset({192, 61}),                           # Croce, Rampa cond.
    "reducer": frozenset({75, 255}),                           # Riduzione, Tee rid. (used as reducer)
    "pipe":    frozenset({53}),                                # Tubo
    "flange":  frozenset({27, 95, 207, 15}),                  # Gibault FL, Bout bic./norm., Saracinesca
    "elbow":   frozenset({54, 111, 175, 190, 195, 235, 245}), # all Curva types
    "unknown": frozenset(),  # empty → skip type check (permissive)
}

# Reverse mapping: numeric code → best-match semantic type.
# Used to annotate observed nodes loaded from image-graph JSONs.
_CODE_TO_SEMANTIC: dict[int, str] = {}
for _sem_type, _codes in _TYPE_TO_CODES.items():
    if _sem_type == "unknown":
        continue
    for _c in _codes:
        if _c not in _CODE_TO_SEMANTIC:
            _CODE_TO_SEMANTIC[_c] = _sem_type

# Standard hydraulic nominal-diameter values (mm).
# Used by is_dn_compatible to define "within one step" tolerance.
_DN_STANDARD = [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600]

# Type pairs considered structurally close (minor mismatch allowed).
# Symmetric: (a, b) covers (b, a) too.
_CLOSE_TYPE_PAIRS: frozenset[frozenset] = frozenset({
    frozenset({"tee", "cross"}),
    frozenset({"tee", "reducer"}),
    frozenset({"valve", "flange"}),
    frozenset({"reducer", "cross"}),
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReferenceNode:
    """A node in the reference graph derived from a PDF hydraulic diagram."""
    id: str
    type: str                  # semantic type: valve | tee | cross | reducer |
                               #                pipe  | elbow | flange | unknown
    dn_main: float | None      # main nominal diameter (mm), from PDF label
    dn_branch: float | None    # branch nominal diameter (mm), from PDF label
    neighbors: list[str]       # IDs of directly connected reference nodes
    degree: int                # len(neighbors)
    shurikode: int | None = None  # numeric component code, assigned from type mapping


@dataclass
class ReferenceEdge:
    """An edge in the reference graph."""
    node_a: str
    node_b: str
    confidence: float          # starts at the LLM-assigned value; reduced by Step 9


@dataclass
class ObservedNode:
    """A node in a per-image observed graph."""
    id: str              # format "CODE#INSTANCE", e.g. "15#0"
    code: int            # numeric component code from YOLO detection
    type: str            # semantic type inferred from code via _CODE_TO_SEMANTIC
    dn_estimate: float | None  # DN estimate — None for standard image-only detection
    neighbors: list[str]
    confidence: float
    image_id: str        # source image filename (stem)


@dataclass
class ObservedEdge:
    """An edge in a per-image observed graph."""
    node_a: str
    node_b: str
    confidence: float


@dataclass
class FusionState:
    """Accumulates multi-view evidence during the fusion process."""

    node_support: dict[str, int] = field(default_factory=dict)
    # reference_node_id → number of images that produced a match

    edge_support: dict[tuple[str, str], int] = field(default_factory=dict)
    # (node_a, node_b) sorted tuple of reference IDs → support count

    supporting_images: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # reference_node_id → set of image_ids that observed a match

    contradictions: list[dict[str, Any]] = field(default_factory=list)
    # logged contradiction events


# Internal graph containers for convenient lookup

@dataclass
class ReferenceGraph:
    nodes: dict[str, ReferenceNode]   # id → node
    edges: list[ReferenceEdge]


@dataclass
class ObservedGraph:
    nodes: dict[str, ObservedNode]    # id → node
    edges: list[ObservedEdge]
    image_id: str


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load reference graph
# ─────────────────────────────────────────────────────────────────────────────

def load_reference_graph(path: str | Path) -> ReferenceGraph:
    """
    Parse the PDF-derived reference graph JSON into typed objects.

    Expected JSON format::

        {
          "components": [
            { "id": str, "type": str, "dn_main": float|null,
              "dn_branch": float|null, ... },
            ...
          ],
          "connections": [
            { "from": str, "to": str, "confidence": float },
            ...
          ]
        }

    Neighbor lists and degree are computed from the connection list.
    Coordinate fields ("center", "bbox") are intentionally ignored.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    # ── Parse edges, build adjacency ──
    raw_edges: list[dict] = raw.get("connections", [])
    adjacency: dict[str, list[str]] = defaultdict(list)
    edges: list[ReferenceEdge] = []

    for conn in raw_edges:
        a = str(conn["from"])
        b = str(conn["to"])
        conf = float(conn.get("confidence", 1.0))
        edges.append(ReferenceEdge(node_a=a, node_b=b, confidence=conf))
        adjacency[a].append(b)
        adjacency[b].append(a)

    # ── Parse nodes ──
    nodes: dict[str, ReferenceNode] = {}
    for comp in raw.get("components", []):
        nid = str(comp["id"])
        raw_type = comp.get("type", "unknown").lower().strip()

        # Normalise to known vocabulary; anything else → "unknown"
        ntype = raw_type if raw_type in _TYPE_TO_CODES else "unknown"

        dn_main   = _as_float(comp.get("dn_main"))
        dn_branch = _as_float(comp.get("dn_branch"))

        nbrs = list(adjacency.get(nid, []))

        # Assign a representative numeric code from the type mapping
        codes_for_type = _TYPE_TO_CODES.get(ntype, frozenset())
        shurikode = next(iter(codes_for_type), None) if codes_for_type else None

        nodes[nid] = ReferenceNode(
            id=nid,
            type=ntype,
            dn_main=dn_main,
            dn_branch=dn_branch,
            neighbors=nbrs,
            degree=len(nbrs),
            shurikode=shurikode,
        )

    logger.info(
        "[ref_graph] loaded %d nodes, %d edges from %s",
        len(nodes), len(edges), path,
    )
    return ReferenceGraph(nodes=nodes, edges=edges)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load observed graphs
# ─────────────────────────────────────────────────────────────────────────────

def load_observed_graphs(folder: str | Path) -> list[ObservedGraph]:
    """
    Load all per-image graph JSONs from *folder*.

    Accepts two JSON formats:

    **Format A** — graph_final.json style::

        { "nodes": [{"nid": str, "code": int, ...}],
          "edges": [{"u": str, "v": str, ...}] }

    **Format B** — raw detection list::

        [ {"code": int, "conf": float, ...}, ... ]

    Nodes whose code belongs to DIAMETER_CODES (label-only markers) are
    excluded — they carry no physical structure information.
    Each resulting graph is labelled with its filename stem as image_id.
    """
    folder = Path(folder)
    graphs: list[ObservedGraph] = []

    for json_path in sorted(folder.glob("*.json")):
        try:
            obs_graph = _parse_observed_json(json_path)
            if obs_graph is not None:
                graphs.append(obs_graph)
        except Exception as exc:
            logger.warning("[obs_graph] skipping %s — %s", json_path.name, exc)

    logger.info(
        "[obs_graph] loaded %d observed graphs from %s", len(graphs), folder
    )
    return graphs


def _parse_observed_json(path: Path) -> ObservedGraph | None:
    """
    Parse a single observed-graph JSON file into an ObservedGraph.
    Returns None if the file does not match either expected format.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    image_id = path.stem

    # ── Format A: dict with "nodes" key ──
    if isinstance(raw, dict) and "nodes" in raw:
        raw_nodes: list[dict] = raw["nodes"]
        raw_edges: list[dict] = raw.get("edges", [])

        # Build adjacency from edge list
        adjacency: dict[str, list[str]] = defaultdict(list)
        edges: list[ObservedEdge] = []

        for e in raw_edges:
            u = str(e.get("u", e.get("from", "")))
            v = str(e.get("v", e.get("to", "")))
            conf = float(e.get("conf", e.get("confidence", 1.0)))
            if u and v:
                edges.append(ObservedEdge(node_a=u, node_b=v, confidence=conf))
                adjacency[u].append(v)
                adjacency[v].append(u)

        nodes: dict[str, ObservedNode] = {}
        for n in raw_nodes:
            nid = str(n.get("nid", n.get("id", "")))
            code = int(n.get("code", 0))

            if code in DIAMETER_CODES:
                continue  # diameter markers are annotations, not physical nodes

            conf = float(n.get("conf", n.get("confidence", 1.0)))
            sem_type = _CODE_TO_SEMANTIC.get(code, "unknown")
            nbrs = list(adjacency.get(nid, []))

            nodes[nid] = ObservedNode(
                id=nid,
                code=code,
                type=sem_type,
                dn_estimate=None,   # image detection does not provide DN
                neighbors=nbrs,
                confidence=conf,
                image_id=image_id,
            )

        if not nodes:
            return None
        return ObservedGraph(nodes=nodes, edges=edges, image_id=image_id)

    # ── Format B: raw detection list ──
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "code" in raw[0]:
        nodes = {}
        for i, det in enumerate(raw):
            code = int(det["code"])
            if code in DIAMETER_CODES:
                continue
            nid = f"{code}#{i}"
            conf = float(det.get("conf", 1.0))
            sem_type = _CODE_TO_SEMANTIC.get(code, "unknown")
            nodes[nid] = ObservedNode(
                id=nid,
                code=code,
                type=sem_type,
                dn_estimate=None,
                neighbors=[],     # raw detections carry no edges
                confidence=conf,
                image_id=image_id,
            )
        return ObservedGraph(nodes=nodes, edges=[], image_id=image_id) if nodes else None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Compatibility filters  (all binary, no numeric weights)
# ─────────────────────────────────────────────────────────────────────────────

def is_type_compatible(observed: ObservedNode, reference: ReferenceNode) -> bool:
    """
    Return True if the observed node's component type is compatible with the
    reference node's semantic type.

    Rules (in order):
      1. If the reference type is "unknown", always pass — the LLM was uncertain.
      2. If the observed type is "unknown", always pass — detection was uncertain.
      3. Exact match: the observed code belongs to the reference-type code set.
      4. Close-type match: the two semantic types belong to a known close pair
         (e.g. "tee" ↔ "cross") — allow minor mismatches from noisy detection.
    """
    ref_type = reference.type

    if ref_type == "unknown":
        return True

    if observed.type == "unknown":
        return True

    # Exact code-level match
    compatible_codes = _TYPE_TO_CODES.get(ref_type, frozenset())
    if observed.code in compatible_codes:
        return True

    # Close semantic type
    pair = frozenset({observed.type, ref_type})
    return pair in _CLOSE_TYPE_PAIRS


def is_dn_compatible(observed: ObservedNode, reference: ReferenceNode) -> bool:
    """
    Return True if the observed DN estimate is compatible with the reference DN.

    Partial information is tolerated: if either side lacks a DN value, the
    check passes.  Standard image-based detection does not produce DN estimates,
    so this check will usually be a pass-through.

    When both sides carry DN information, the values must lie within one step
    on the standard DN scale (e.g. DN100 and DN125 are one step apart).
    """
    obs_dn = observed.dn_estimate
    ref_dn = reference.dn_main   # compare against main diameter

    if obs_dn is None or ref_dn is None:
        return True

    return _dn_within_one_step(obs_dn, ref_dn)


def is_degree_plausible(observed: ObservedNode, reference: ReferenceNode) -> bool:
    """
    Return True if the observed node's degree is physically plausible given
    the reference node's position.

    The only hard rejection is when the observed degree exceeds the physical
    valence of the component type (e.g. a Saracinesca with 3 neighbours is
    impossible — its valence is 2).

    We intentionally do *not* reject low-degree observations: a Tee with only
    1 visible neighbour is fine — the others may be off-screen.
    """
    valence, _ = COMPONENT_INFO.get(observed.code, (None, ""))
    if valence is None or valence == 0:
        return True  # unknown or label-only component

    obs_deg = len(observed.neighbors)

    # Reject if observed degree exceeds the physical maximum
    if obs_deg > valence:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Anchor detection
# ─────────────────────────────────────────────────────────────────────────────

def find_anchor_nodes(reference_graph: ReferenceGraph) -> list[ReferenceNode]:
    """
    Return reference nodes sorted by structural distinctiveness (descending).

    An anchor is a node that is easiest to *uniquely* identify in an image:
      - High degree (≥ 3): junctions are visually salient.
      - Rare type in this specific graph: fewer ambiguous candidates.
      - Unique DN combination: distinctive label signature.

    Nodes with degree < 3 are excluded — they offer too little structural
    context to be reliable seeds for BFS expansion.

    Distinctiveness is a lexicographic tuple — no arbitrary numeric weights.
    """
    type_counts: dict[str, int] = defaultdict(int)
    for node in reference_graph.nodes.values():
        type_counts[node.type] += 1

    dn_pair_counts: dict[tuple, int] = defaultdict(int)
    for node in reference_graph.nodes.values():
        dn_pair_counts[(node.dn_main, node.dn_branch)] += 1

    ranked: list[tuple[tuple, ReferenceNode]] = []

    for node in reference_graph.nodes.values():
        if node.degree < 3:
            continue  # not a junction — too ambiguous as anchor

        degree_score = node.degree
        # More negative → rarer type → sorts higher
        type_rarity = -type_counts[node.type]
        # 1 if this DN pair is unique in the reference graph, else 0
        dn_unique = 1 if dn_pair_counts[(node.dn_main, node.dn_branch)] == 1 else 0

        # Lexicographic: compare degree first, then DN uniqueness, then rarity
        distinctiveness = (degree_score, dn_unique, type_rarity)
        ranked.append((distinctiveness, node))

    ranked.sort(key=lambda x: x[0], reverse=True)
    anchors = [node for _, node in ranked]

    logger.info(
        "[anchors] %d anchor nodes: %s",
        len(anchors), [a.id for a in anchors[:5]],
    )
    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Node matching candidates
# ─────────────────────────────────────────────────────────────────────────────

def candidate_matches(
    observed_node: ObservedNode,
    reference_graph: ReferenceGraph,
) -> list[ReferenceNode]:
    """
    Return all reference nodes that pass every compatibility filter for the
    given observed node (type, DN, degree).

    The returned list may contain multiple candidates.  Callers should use
    validate_local_structure to further discriminate between them.
    """
    return [
        ref_node
        for ref_node in reference_graph.nodes.values()
        if (is_type_compatible(observed_node, ref_node)
            and is_dn_compatible(observed_node, ref_node)
            and is_degree_plausible(observed_node, ref_node))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Local structural validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_local_structure(
    observed_node: ObservedNode,
    reference_node: ReferenceNode,
    observed_graph: ObservedGraph,
    reference_graph: ReferenceGraph,
    debug_log: list[dict] | None = None,
) -> bool:
    """
    Check whether placing *observed_node* at the position of *reference_node*
    is consistent with the local neighbourhood constraints from the domain priors.

    Checks (all hard rules — no numeric scores):

    1. **FORBIDDEN_PAIRS**: none of the observed node's neighbours may form a
       forbidden pair with the observed node itself.

    2. **DEGREE_SATURATION** (terminal components): if DEGREE_SATURATION for the
       observed code is 0.0 (always terminal in the dataset) and the node already
       has ≥ 2 neighbours with a valence of ≤ 2, reject — it cannot be a junction.

    3. **TRIPLET_COUNTS** (soft): if the code appears in the triplet table *and*
       no observed neighbour pair forms any known triplet, log a warning but do
       NOT hard-reject (partial observations are normal).

    Returns True if all hard checks pass.
    """
    obs_code = observed_node.code

    # Collect numeric codes of observed neighbours
    obs_nbr_codes: list[int] = []
    for nbr_id in observed_node.neighbors:
        nbr = observed_graph.nodes.get(nbr_id)
        if nbr:
            obs_nbr_codes.append(nbr.code)

    # ── Check 1: FORBIDDEN_PAIRS ──
    for nbr_code in obs_nbr_codes:
        pair = (min(obs_code, nbr_code), max(obs_code, nbr_code))
        if pair in FORBIDDEN_PAIRS:
            _log(debug_log, {
                "check": "forbidden_pair",
                "observed": observed_node.id,
                "reference": reference_node.id,
                "pair": list(pair),
                "result": "REJECT",
            })
            return False

    # ── Check 2: DEGREE_SATURATION — terminal component anomaly ──
    valence, _ = COMPONENT_INFO.get(obs_code, (None, ""))
    if valence is not None and valence <= 2:
        saturation = DEGREE_SATURATION.get(obs_code, 0.5)
        obs_degree = len(observed_node.neighbors)
        # A component that is *always* terminal in the dataset should not
        # appear as a hub (≥ 2 neighbours) when its valence is 2.
        if saturation == 0.0 and obs_degree >= 2:
            _log(debug_log, {
                "check": "degree_saturation",
                "observed": observed_node.id,
                "code": obs_code,
                "saturation": saturation,
                "actual_degree": obs_degree,
                "result": "REJECT",
            })
            return False

    # ── Check 3: TRIPLET_COUNTS (soft warning) ──
    if len(obs_nbr_codes) >= 2:
        triplet_ok = _any_triplet_supported(obs_code, obs_nbr_codes)

        if not triplet_ok:
            code_in_table = any(obs_code == t[1] for t in TRIPLET_COUNTS)
            if code_in_table:
                # Known code, but no matching triplet — log for inspection
                _log(debug_log, {
                    "check": "triplet_support",
                    "observed": observed_node.id,
                    "code": obs_code,
                    "neighbor_codes": obs_nbr_codes,
                    "result": "WARN",   # warn only — partial observation is expected
                })

    _log(debug_log, {
        "check": "local_structure",
        "observed": observed_node.id,
        "reference": reference_node.id,
        "result": "PASS",
    })
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Hypothesis expansion
# ─────────────────────────────────────────────────────────────────────────────

def expand_match(
    anchor_observed: ObservedNode,
    anchor_reference: ReferenceNode,
    observed_graph: ObservedGraph,
    reference_graph: ReferenceGraph,
    debug_log: list[dict] | None = None,
) -> dict[str, str]:
    """
    BFS-expand a single anchor match into a partial node-to-node hypothesis.

    Starting from the confirmed pair (anchor_observed → anchor_reference),
    iteratively attempts to match unmatched neighbours of each confirmed pair.

    For each unmatched observed neighbour:
      - Iterate over unmatched reference neighbours of the current reference node.
      - Accept the first reference neighbour that passes all compatibility and
        structural validation checks.
      - If no compatible reference neighbour is found, the observed node is left
        unmatched (partial match — not an error).

    Returns:
        A dict mapping observed_node_id → reference_node_id for all matched nodes.
    """
    # Confirmed matches (observed → reference) and their inverse
    match: dict[str, str] = {anchor_observed.id: anchor_reference.id}
    reverse: dict[str, str] = {anchor_reference.id: anchor_observed.id}

    queue: deque[tuple[str, str]] = deque([(anchor_observed.id, anchor_reference.id)])

    _log(debug_log, {
        "step": "expand_match",
        "anchor_observed": anchor_observed.id,
        "anchor_reference": anchor_reference.id,
    })

    while queue:
        obs_id, ref_id = queue.popleft()
        obs_node = observed_graph.nodes[obs_id]
        ref_node = reference_graph.nodes[ref_id]

        # Neighbours not yet assigned on each side
        unmatched_obs = [n for n in obs_node.neighbors if n not in match]
        unmatched_ref = [n for n in ref_node.neighbors if n not in reverse]

        for obs_nbr_id in unmatched_obs:
            obs_nbr = observed_graph.nodes.get(obs_nbr_id)
            if obs_nbr is None:
                continue

            best_ref_id: str | None = None

            for ref_nbr_id in list(unmatched_ref):
                ref_nbr = reference_graph.nodes.get(ref_nbr_id)
                if ref_nbr is None:
                    continue

                if (is_type_compatible(obs_nbr, ref_nbr)
                        and is_dn_compatible(obs_nbr, ref_nbr)
                        and is_degree_plausible(obs_nbr, ref_nbr)
                        and validate_local_structure(
                            obs_nbr, ref_nbr, observed_graph, reference_graph, debug_log
                        )):
                    best_ref_id = ref_nbr_id
                    break  # first compatible neighbour wins

            if best_ref_id is not None:
                match[obs_nbr_id] = best_ref_id
                reverse[best_ref_id] = obs_nbr_id
                unmatched_ref.remove(best_ref_id)
                queue.append((obs_nbr_id, best_ref_id))

                _log(debug_log, {
                    "step": "expand_match",
                    "matched_observed": obs_nbr_id,
                    "to_reference": best_ref_id,
                })
            else:
                _log(debug_log, {
                    "step": "expand_match",
                    "unmatched_observed": obs_nbr_id,
                    "reason": "no compatible reference neighbour",
                })

    return match


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Multi-view fusion
# ─────────────────────────────────────────────────────────────────────────────

def fuse_observations(
    observed_graphs: list[ObservedGraph],
    reference_graph: ReferenceGraph,
    debug_log: list[dict] | None = None,
) -> FusionState:
    """
    Fuse evidence from multiple per-image observed graphs into a FusionState.

    For each image:
      1. Identify high-degree observed nodes as anchor candidates.
      2. For each anchor, attempt matching against reference anchors.
      3. Expand the first successful anchor match via BFS.
      4. If no anchor match succeeds, fall back to greedy flat matching.
      5. Accumulate node / edge support and record contradictions.
    """
    state = FusionState()
    ref_anchors = find_anchor_nodes(reference_graph)

    for obs_graph in observed_graphs:
        image_id = obs_graph.image_id
        logger.info(
            "[fusion] image: %s  (%d nodes)", image_id, len(obs_graph.nodes)
        )

        hypothesis: dict[str, str] | None = _build_hypothesis(
            obs_graph, reference_graph, ref_anchors, debug_log
        )

        if hypothesis:
            _accumulate_support(
                hypothesis, obs_graph, reference_graph, state, image_id, debug_log
            )
        else:
            logger.warning("[fusion] no hypothesis produced for image: %s", image_id)

    return state


def _build_hypothesis(
    obs_graph: ObservedGraph,
    reference_graph: ReferenceGraph,
    ref_anchors: list[ReferenceNode],
    debug_log: list[dict] | None,
) -> dict[str, str] | None:
    """
    Attempt to build a matching hypothesis for one observed graph.

    Tries anchor-based BFS expansion first; falls back to greedy flat matching.
    Returns the hypothesis dict (obs_id → ref_id) or None if nothing matched.
    """
    # Sort observed nodes by degree (high-degree first → better anchors)
    obs_by_degree = sorted(
        obs_graph.nodes.values(),
        key=lambda n: len(n.neighbors),
        reverse=True,
    )

    # Try at most 5 high-degree observed nodes as anchor candidates
    for obs_anchor in obs_by_degree[:5]:
        if len(obs_anchor.neighbors) < 2:
            break  # remaining nodes have too low degree to anchor reliably

        for ref_anchor in ref_anchors:
            if not is_type_compatible(obs_anchor, ref_anchor):
                continue

            hyp = expand_match(
                obs_anchor, ref_anchor, obs_graph, reference_graph, debug_log
            )

            if len(hyp) >= 2:
                return hyp  # success

    # Fallback: greedy flat matching (no structural expansion)
    flat = _greedy_flat_match(obs_graph, reference_graph)
    return flat if flat else None


def _greedy_flat_match(
    obs_graph: ObservedGraph,
    reference_graph: ReferenceGraph,
) -> dict[str, str]:
    """
    Fallback: greedily assign each observed node to the first compatible
    unassigned reference node, prioritising rarer component types.

    This produces a valid (if imperfect) hypothesis when no anchor BFS works.
    """
    _type_priority = {
        "cross": 0, "tee": 1, "reducer": 2, "valve": 3,
        "elbow": 4, "pipe": 5, "flange": 6, "unknown": 7,
    }
    sorted_obs = sorted(
        obs_graph.nodes.values(),
        key=lambda n: _type_priority.get(n.type, 7),
    )

    match: dict[str, str] = {}
    used: set[str] = set()

    for obs_node in sorted_obs:
        for ref_node in candidate_matches(obs_node, reference_graph):
            if ref_node.id not in used:
                match[obs_node.id] = ref_node.id
                used.add(ref_node.id)
                break

    return match


def _accumulate_support(
    hypothesis: dict[str, str],
    obs_graph: ObservedGraph,
    reference_graph: ReferenceGraph,
    state: FusionState,
    image_id: str,
    debug_log: list[dict] | None,
) -> None:
    """
    Update FusionState with evidence from one matched hypothesis.

    - Increments node_support for each matched reference node.
    - Records the image in supporting_images.
    - For each observed edge whose both endpoints are matched, increments
      edge_support for the corresponding reference edge pair.
    - Flags contradictions when a matched observed edge maps to a reference
      node pair that is NOT connected in the reference graph.
    """
    # Build fast lookup of reference edges
    ref_edge_set: set[tuple[str, str]] = {
        (min(e.node_a, e.node_b), max(e.node_a, e.node_b))
        for e in reference_graph.edges
    }

    # Node support
    for obs_id, ref_id in hypothesis.items():
        state.node_support[ref_id] = state.node_support.get(ref_id, 0) + 1
        state.supporting_images[ref_id].add(image_id)

    # Edge support + contradiction detection
    for obs_edge in obs_graph.edges:
        ref_a = hypothesis.get(obs_edge.node_a)
        ref_b = hypothesis.get(obs_edge.node_b)

        if ref_a is None or ref_b is None:
            continue  # one endpoint is unmatched — skip edge

        edge_key = (min(ref_a, ref_b), max(ref_a, ref_b))
        state.edge_support[edge_key] = state.edge_support.get(edge_key, 0) + 1

        if edge_key not in ref_edge_set:
            # Observed edge connects two reference nodes that are not linked
            # in the reference graph → potential prior error
            contradiction = {
                "type": "missing_reference_edge",
                "observed_edge": [obs_edge.node_a, obs_edge.node_b],
                "reference_edge": list(edge_key),
                "image_id": image_id,
            }
            state.contradictions.append(contradiction)
            _log(debug_log, {"step": "accumulate", "contradiction": contradiction})


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Reference graph correction
# ─────────────────────────────────────────────────────────────────────────────

def update_reference_graph_with_observations(
    reference_graph: ReferenceGraph,
    state: FusionState,
    total_images: int,
    min_support_fraction: float = 0.5,
    debug_log: list[dict] | None = None,
) -> ReferenceGraph:
    """
    Revise the reference graph based on accumulated multi-view observations.

    Operations:

    **A — Reduce confidence of contradicted prior edges**
        If a reference edge received zero support AND both its endpoints were
        observed in at least one image, confidence is halved.  These edges
        may be prior errors.

    **B — Log consistently unobserved nodes**
        Reference nodes never matched despite sufficient images are flagged
        in the debug log — they may be artefacts in the PDF extraction.

    **C — Insert strongly supported new edges**
        If an observed edge pair appears in ≥ ``min_support_fraction`` of
        images but is absent from the reference graph, it is inserted as a
        new edge with confidence proportional to its support.

    Returns a revised ReferenceGraph (the original is not mutated).
    """
    # Deep-copy nodes and edges
    new_nodes: dict[str, ReferenceNode] = {
        nid: ReferenceNode(
            id=n.id, type=n.type, dn_main=n.dn_main, dn_branch=n.dn_branch,
            neighbors=list(n.neighbors), degree=n.degree, shurikode=n.shurikode,
        )
        for nid, n in reference_graph.nodes.items()
    }
    new_edges: list[ReferenceEdge] = [
        ReferenceEdge(e.node_a, e.node_b, e.confidence)
        for e in reference_graph.edges
    ]

    observed_node_ids: set[str] = set(state.node_support.keys())

    # ── A: Reduce confidence of unsupported prior edges ──
    for edge in new_edges:
        edge_key = (min(edge.node_a, edge.node_b), max(edge.node_a, edge.node_b))
        both_observed = (edge.node_a in observed_node_ids
                         and edge.node_b in observed_node_ids)

        if both_observed and state.edge_support.get(edge_key, 0) == 0:
            old_conf = edge.confidence
            edge.confidence = round(edge.confidence * 0.5, 3)
            _log(debug_log, {
                "step": "update_reference",
                "action": "reduce_edge_confidence",
                "edge": list(edge_key),
                "old_confidence": old_conf,
                "new_confidence": edge.confidence,
            })

    # ── B: Log unobserved reference nodes ──
    min_images_to_penalise = max(2, total_images // 2)
    for nid, node in new_nodes.items():
        if state.node_support.get(nid, 0) == 0 and total_images >= min_images_to_penalise:
            _log(debug_log, {
                "step": "update_reference",
                "action": "unobserved_node",
                "node": nid,
                "type": node.type,
                "total_images": total_images,
            })

    # ── C: Insert consistently supported new edges ──
    existing_edge_keys: set[tuple[str, str]] = {
        (min(e.node_a, e.node_b), max(e.node_a, e.node_b))
        for e in new_edges
    }

    min_support = max(1, int(total_images * min_support_fraction))

    for edge_key, support in state.edge_support.items():
        if edge_key in existing_edge_keys:
            continue  # already in the reference graph

        if support < min_support:
            continue  # insufficient evidence

        a, b = edge_key
        if a not in new_nodes or b not in new_nodes:
            continue  # endpoints are novel — do not invent reference nodes

        new_conf = round(support / total_images, 3)
        new_edges.append(ReferenceEdge(node_a=a, node_b=b, confidence=new_conf))
        new_nodes[a].neighbors.append(b)
        new_nodes[b].neighbors.append(a)
        new_nodes[a].degree += 1
        new_nodes[b].degree += 1
        existing_edge_keys.add(edge_key)

        _log(debug_log, {
            "step": "update_reference",
            "action": "insert_new_edge",
            "edge": list(edge_key),
            "support": support,
            "confidence": new_conf,
        })

    return ReferenceGraph(nodes=new_nodes, edges=new_edges)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Output
# ─────────────────────────────────────────────────────────────────────────────

def save_fused_graph(
    reference_graph: ReferenceGraph,
    state: FusionState,
    total_images: int,
    output_path: str | Path,
) -> None:
    """
    Write ``fused_graph.json``.

    For every reference node the output includes:
      - ``support_count``: how many images produced a match for this node.
      - ``supporting_images``: sorted list of those image IDs.
      - ``observed_confidence``: support_count / total_images.

    For every reference edge:
      - ``prior_confidence``: confidence after Step 9 adjustments.
      - ``support_count``: how many images confirmed this edge.
      - ``observed_confidence``: support_count / total_images.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes_out = []
    for nid, node in reference_graph.nodes.items():
        support = state.node_support.get(nid, 0)
        imgs = sorted(state.supporting_images.get(nid, set()))
        obs_conf = round(support / total_images, 3) if total_images > 0 else 0.0

        nodes_out.append({
            "id": nid,
            "type": node.type,
            "dn_main": node.dn_main,
            "dn_branch": node.dn_branch,
            "degree": node.degree,
            "shurikode": node.shurikode,
            "support_count": support,
            "supporting_images": imgs,
            "observed_confidence": obs_conf,
        })

    edges_out = []
    for edge in reference_graph.edges:
        edge_key = (min(edge.node_a, edge.node_b), max(edge.node_a, edge.node_b))
        support = state.edge_support.get(edge_key, 0)
        obs_conf = round(support / total_images, 3) if total_images > 0 else 0.0

        edges_out.append({
            "node_a": edge.node_a,
            "node_b": edge.node_b,
            "prior_confidence": edge.confidence,
            "support_count": support,
            "observed_confidence": obs_conf,
        })

    output = {
        "total_images": total_images,
        "nodes": nodes_out,
        "edges": edges_out,
    }
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[output] fused graph → %s", output_path)


def save_fusion_debug(
    state: FusionState,
    debug_log: list[dict],
    output_path: str | Path,
) -> None:
    """
    Write ``fusion_debug.json`` with the full decision log.

    Contents:
      - ``node_support``: reference_node_id → support count
      - ``edge_support``: "a↔b" → support count
      - ``supporting_images``: reference_node_id → sorted image list
      - ``contradictions``: list of contradiction events
      - ``decision_log``: every matching decision logged during the run
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "node_support": dict(state.node_support),
        "edge_support": {
            f"{a}↔{b}": v for (a, b), v in state.edge_support.items()
        },
        "supporting_images": {
            k: sorted(v) for k, v in state.supporting_images.items()
        },
        "contradictions": state.contradictions,
        "decision_log": debug_log,
    }
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[output] fusion debug → %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_fusion(
    reference_graph_path: str | Path,
    observed_graphs_folder: str | Path,
    output_dir: str | Path = "./output/fusion",
) -> tuple[ReferenceGraph, FusionState]:
    """
    Execute the full graph fusion pipeline.

    Args:
        reference_graph_path:   Path to the PDF-derived ``reference_graph.json``.
        observed_graphs_folder: Folder containing per-image graph JSONs
                                (``graph_final.json`` style or raw detection lists).
        output_dir:             Destination for ``fused_graph.json`` and
                                ``fusion_debug.json``.

    Returns:
        A tuple of (updated_reference_graph, fusion_state).
    """
    output_dir = Path(output_dir)
    debug_log: list[dict] = []

    # Steps 1 & 2 — load
    ref_graph = load_reference_graph(reference_graph_path)
    obs_graphs = load_observed_graphs(observed_graphs_folder)
    total_images = len(obs_graphs)

    if total_images == 0:
        logger.warning("[fusion] no observed graphs found — writing empty output")
        state = FusionState()
        save_fused_graph(ref_graph, state, 0, output_dir / "fused_graph.json")
        save_fusion_debug(state, debug_log, output_dir / "fusion_debug.json")
        return ref_graph, state

    # Steps 3–8 — fuse
    state = fuse_observations(obs_graphs, ref_graph, debug_log)

    # Step 9 — correct reference graph
    updated_ref = update_reference_graph_with_observations(
        ref_graph, state, total_images, debug_log=debug_log
    )

    # Step 10 — output
    save_fused_graph(updated_ref, state, total_images, output_dir / "fused_graph.json")
    save_fusion_debug(state, debug_log, output_dir / "fusion_debug.json")

    logger.info(
        "[fusion] complete — nodes supported: %d  edges supported: %d  contradictions: %d",
        sum(1 for v in state.node_support.values() if v > 0),
        sum(1 for v in state.edge_support.values() if v > 0),
        len(state.contradictions),
    )
    return updated_ref, state


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _as_float(value: Any) -> float | None:
    """Convert *value* to float, returning None if not convertible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dn_within_one_step(dn_a: float, dn_b: float) -> bool:
    """
    Return True if *dn_a* and *dn_b* are within one step on the standard DN scale.

    Values that do not snap to a standard size (within 15% tolerance) are
    compared with a flat 20% relative tolerance.
    """
    def _snap(dn: float) -> int | None:
        idx = min(range(len(_DN_STANDARD)), key=lambda i: abs(_DN_STANDARD[i] - dn))
        if abs(_DN_STANDARD[idx] - dn) / max(dn, 1) < 0.15:
            return idx
        return None

    idx_a = _snap(dn_a)
    idx_b = _snap(dn_b)

    if idx_a is None or idx_b is None:
        # At least one non-standard size — use relative tolerance
        return abs(dn_a - dn_b) / max(dn_a, dn_b, 1) <= 0.20

    return abs(idx_a - idx_b) <= 1


def _any_triplet_supported(center_code: int, neighbor_codes: list[int]) -> bool:
    """
    Return True if any pair of *neighbor_codes* forms a triplet
    ``(min(a,c), center_code, max(a,c))`` present in TRIPLET_COUNTS.
    """
    n = len(neighbor_codes)
    for i in range(n):
        for j in range(i + 1, n):
            a, c = neighbor_codes[i], neighbor_codes[j]
            if (min(a, c), center_code, max(a, c)) in TRIPLET_COUNTS:
                return True
    return False


def _log(log: list[dict] | None, entry: dict) -> None:
    """Append *entry* to *log* if debug logging is active."""
    if log is not None:
        log.append(entry)
