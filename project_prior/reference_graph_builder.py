"""
Costruzione incrementale del reference graph a partire dal diagramma PDF isolato.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math
import re

from debug_utils import save_json
from .pdf_graph_extraction import (
    DiagramCandidate,
    DiagramCandidateExtractionResult,
    PdfCandidateComponent,
    PdfPageExtraction,
    PdfPriorExtractionResult,
    PdfPrimitive,
    PdfWord,
    _UnionFind,
    _bbox_area,
    _bbox_intersects,
    _cell_keys_for_bbox,
    _expand_bbox,
    _merge_bbox,
    _render_pdf_page,
)

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None
    np = None

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None


REFERENCE_DEBUG_DIRNAME = "reference_graph"
PRIMITIVE_MIN_REL_SPAN = 0.0022
COMPONENT_CLUSTER_MARGIN_FRAC = 0.010
TEXT_MATCH_MARGIN_FRAC = 0.018
TEXT_MATCH_MAX_DIST_FRAC = 0.12
EDGE_MAX_GAP_FRAC = 0.22
EDGE_NEIGHBORS = 3

TYPE_KEYWORDS = {
    "SARACINESCA": "Saracinesca",
    "SAR": "Saracinesca",
    "TEE": "Tee",
    "CROCE": "Croce",
    "RID": "Riduzione",
    "RIDUZ": "Riduzione",
    "CURVA": "Curva",
    "TOULIPE": "Toulipe",
    "BOUT": "Bout",
    "FLANG": "Flangia",
    "FL.": "Flangia",
}

MATERIAL_KEYWORDS = ("PVC", "PEAD", "PE", "GHISA", "ACCIAIO", "FERRO")


@dataclass
class PrimitiveCleanupStats:
    page_index: int
    diagram_candidate_id: str
    raw_count: int
    cleaned_count: int
    raw_by_kind: dict[str, int]
    cleaned_by_kind: dict[str, int]
    min_rel_span: float
    min_abs_span: float
    raw_bbox: list[float]
    cleaned_bbox: list[float]


@dataclass
class ReferenceComponentCandidate:
    component_id: str
    page_index: int
    bbox: list[float]
    anchor: list[float]
    primitive_ids: list[str]
    source_component_ids: list[str]
    text_block_ids: list[str]
    stats: dict[str, Any]
    score: float
    rank: int | None = None
    hypothesized_type: str | None = None
    text_labels: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceTextMatch:
    match_id: str
    text_block_id: str
    text: str
    bbox: list[float]
    component_id: str | None
    score: float
    parsed: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ReferenceGraphNode:
    node_id: str
    page_index: int
    bbox: list[float]
    anchor: list[float]
    hypothesized_type: str | None
    text_labels: list[str]
    attributes: dict[str, Any]
    primitive_ids: list[str]
    stats: dict[str, Any]


@dataclass
class ReferenceGraphEdge:
    edge_id: str
    source: str
    target: str
    score: float
    stats: dict[str, Any]


@dataclass
class ReferenceGraphResult:
    pdf_path: str
    status: str
    page_index: int | None
    diagram_candidate_id: str | None
    nodes: list[ReferenceGraphNode]
    edges: list[ReferenceGraphEdge]
    component_candidates: list[ReferenceComponentCandidate]
    text_matches: list[ReferenceTextMatch]
    debug_paths: dict[str, str] = field(default_factory=dict)
    summary_stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _init_reference_debug(parsed_pdf: PdfPriorExtractionResult) -> dict[str, Path]:
    root = Path(parsed_pdf.debug_paths.get("root", "."))
    ref_root = root / REFERENCE_DEBUG_DIRNAME
    dirs = {
        "root": ref_root,
        "json": ref_root / "json",
        "plots": ref_root / "plots",
        "logs": ref_root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _bbox_dims(bbox: list[float]) -> tuple[float, float]:
    return max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])


def _bbox_center(bbox: list[float]) -> list[float]:
    return [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]


def _bbox_diag(bbox: list[float]) -> float:
    w, h = _bbox_dims(bbox)
    return math.hypot(w, h)


def _bbox_gap(a: list[float], b: list[float]) -> float:
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).upper()


def _parse_text_label(text: str) -> dict[str, Any]:
    upper = _normalize_text(text)
    parsed: dict[str, Any] = {
        "raw": text,
        "normalized": upper,
        "type_hint": None,
        "dn_values": [],
        "size_ratios": [],
        "materials": [],
    }
    for token, type_name in TYPE_KEYWORDS.items():
        if token in upper:
            parsed["type_hint"] = type_name
            break
    parsed["dn_values"] = [int(val) for val in re.findall(r"DN\s*([0-9]+)", upper)]
    parsed["size_ratios"] = [
        {"a": int(a), "b": int(b)}
        for a, b in re.findall(r"([0-9]{2,4})\s*/\s*([0-9]{2,4})", upper)
    ]
    parsed["materials"] = [token for token in MATERIAL_KEYWORDS if token in upper]
    return parsed


def _group_words_into_blocks(words: list[PdfWord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[tuple[int, PdfWord]]] = {}
    fallback_rows: list[list[tuple[int, PdfWord]]] = []

    for idx, word in enumerate(words):
        if word.block_no is not None and word.line_no is not None:
            grouped.setdefault((word.block_no, word.line_no), []).append((idx, word))
            continue

        cy = _bbox_center(word.bbox)[1]
        placed = False
        for row in fallback_rows:
            ref_cy = _bbox_center(row[0][1].bbox)[1]
            if abs(cy - ref_cy) <= max(2.0, 0.35 * max(word.bbox[3] - word.bbox[1], 1.0)):
                row.append((idx, word))
                placed = True
                break
        if not placed:
            fallback_rows.append([(idx, word)])

    blocks: list[dict[str, Any]] = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda item: item[1].bbox[0])
        blocks.append({
            "text_block_id": f"blk_{key[0]}_{key[1]}",
            "word_indices": [idx for idx, _ in items],
            "text": " ".join(word.text for _, word in items).strip(),
            "bbox": _merge_bbox([word.bbox for _, word in items]),
        })

    for row_idx, row in enumerate(fallback_rows):
        row = sorted(row, key=lambda item: item[1].bbox[0])
        blocks.append({
            "text_block_id": f"row_{row_idx:04d}",
            "word_indices": [idx for idx, _ in row],
            "text": " ".join(word.text for _, word in row).strip(),
            "bbox": _merge_bbox([word.bbox for _, word in row]),
        })

    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return [block for block in blocks if block["text"]]


def _primitive_to_entry(kind: str, index: int, primitive: PdfPrimitive) -> dict[str, Any]:
    bbox = primitive.bbox
    width, height = _bbox_dims(bbox)
    if kind == "segment":
        x0 = float(primitive.data.get("x0", bbox[0]))
        y0 = float(primitive.data.get("y0", bbox[1]))
        x1 = float(primitive.data.get("x1", bbox[2]))
        y1 = float(primitive.data.get("y1", bbox[3]))
        span = math.hypot(x1 - x0, y1 - y0)
    else:
        span = _bbox_diag(bbox)
    aspect = max(width, height) / max(min(width, height), 1e-6)
    return {
        "primitive_id": f"{kind}_{index:06d}",
        "kind": kind,
        "bbox": bbox,
        "span": span,
        "width": width,
        "height": height,
        "aspect": aspect,
        "center": _bbox_center(bbox),
        "data": primitive.data,
    }


def _collect_primitives_for_candidate(page: PdfPageExtraction, candidate: DiagramCandidate) -> list[dict[str, Any]]:
    roi = _expand_bbox(candidate.bbox, max(page.width, page.height) * 0.006)
    entries: list[dict[str, Any]] = []
    for idx, primitive in enumerate(page.segments):
        if _bbox_intersects(roi, primitive.bbox):
            entries.append(_primitive_to_entry("segment", idx, primitive))
    for idx, primitive in enumerate(page.curves):
        if _bbox_intersects(roi, primitive.bbox):
            entries.append(_primitive_to_entry("curve", idx, primitive))
    for idx, primitive in enumerate(page.rects):
        if _bbox_intersects(roi, primitive.bbox):
            entries.append(_primitive_to_entry("rect", idx, primitive))
    return entries


def _count_by_kind(primitives: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in primitives:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    return counts


def _cleanup_primitives(page: PdfPageExtraction, candidate: DiagramCandidate, primitives: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], PrimitiveCleanupStats]:
    candidate_diag = max(_bbox_diag(candidate.bbox), 1.0)
    min_abs_span = candidate_diag * PRIMITIVE_MIN_REL_SPAN
    cleaned: list[dict[str, Any]] = []

    for item in primitives:
        span = float(item["span"])
        width = float(item["width"])
        height = float(item["height"])
        keep = False
        if item["kind"] == "segment":
            keep = span >= min_abs_span
        else:
            keep = span >= min_abs_span * 0.85 or max(width, height) >= min_abs_span * 0.7
        if keep:
            cleaned.append(item)

    stats = PrimitiveCleanupStats(
        page_index=page.page_index,
        diagram_candidate_id=candidate.candidate_id,
        raw_count=len(primitives),
        cleaned_count=len(cleaned),
        raw_by_kind=_count_by_kind(primitives),
        cleaned_by_kind=_count_by_kind(cleaned),
        min_rel_span=PRIMITIVE_MIN_REL_SPAN,
        min_abs_span=min_abs_span,
        raw_bbox=_merge_bbox([item["bbox"] for item in primitives]) if primitives else candidate.bbox,
        cleaned_bbox=_merge_bbox([item["bbox"] for item in cleaned]) if cleaned else candidate.bbox,
    )
    return cleaned, stats


def _classify_source_component(component: PdfCandidateComponent) -> dict[str, Any]:
    bbox = component.bbox
    width, height = _bbox_dims(bbox)
    area = _bbox_area(bbox)
    aspect = max(width, height) / max(min(width, height), 1e-6)
    segs = int(component.primitive_kinds.get("segment", 0))
    curves = int(component.primitive_kinds.get("curve", 0))
    rects = int(component.primitive_kinds.get("rect", 0))
    connector_like = (
        aspect > 10.0
        and segs >= max(curves + rects, 1)
        and area > 0.0
    )
    return {
        "component_id": component.component_id,
        "bbox": bbox,
        "center": _bbox_center(bbox),
        "primitive_count": component.primitive_count,
        "segment_count": segs,
        "curve_count": curves,
        "rect_count": rects,
        "width": width,
        "height": height,
        "area": area,
        "aspect": aspect,
        "connector_like": connector_like,
        "source": component.source,
    }


def _cluster_source_components(atoms: list[dict[str, Any]], candidate_diag: float) -> list[list[int]]:
    if not atoms:
        return []
    cell_size = max(candidate_diag * 0.22, 1.0)
    margin = max(candidate_diag * COMPONENT_CLUSTER_MARGIN_FRAC, 1.0)
    expanded = [_expand_bbox(atom["bbox"], margin) for atom in atoms]
    uf = _UnionFind(len(atoms))
    grid: dict[tuple[int, int], list[int]] = {}

    for idx, bbox in enumerate(expanded):
        seen: set[int] = set()
        for key in _cell_keys_for_bbox(bbox, cell_size):
            for other in grid.get(key, []):
                if other in seen:
                    continue
                seen.add(other)
                if _bbox_intersects(bbox, expanded[other]):
                    uf.union(idx, other)
            grid.setdefault(key, []).append(idx)

    groups: dict[int, list[int]] = {}
    for idx in range(len(atoms)):
        groups.setdefault(uf.find(idx), []).append(idx)
    return list(groups.values())


def _component_group_score(stats: dict[str, Any]) -> float:
    area_ratio = stats["area_ratio"]
    compact_bonus = 1.0 / max(stats["aspect"], 1.0)
    score = (
        1.25 * math.log1p(stats["primitive_count"])
        + 0.70 * math.log1p(stats["n_curves"] + 1)
        + 0.40 * math.log1p(stats["n_rects"] + 1)
        + 0.30 * math.log1p(stats["n_source_components"] + 1)
        + 0.20 * math.log1p(stats["n_near_words"] + 1)
        + 1.2 * compact_bonus
    )
    if area_ratio > 0.18:
        score -= 1.4
    if area_ratio < 0.0004:
        score -= 1.0
    if stats["connector_fraction"] > 0.7:
        score -= 2.0
    return score


def _build_component_candidates(
    page: PdfPageExtraction,
    diagram_candidate: DiagramCandidate,
    cleaned_primitives: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
) -> tuple[list[ReferenceComponentCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_ids = set(diagram_candidate.component_ids)
    atoms = [
        _classify_source_component(component)
        for component in page.candidate_components
        if component.component_id in selected_ids
    ]
    if not atoms:
        atoms = [
            _classify_source_component(component)
            for component in page.candidate_components
            if _bbox_intersects(_expand_bbox(diagram_candidate.bbox, 8.0), component.bbox)
        ]

    non_connector_indices = [idx for idx, atom in enumerate(atoms) if not atom["connector_like"]]
    cluster_atoms = [atoms[idx] for idx in non_connector_indices] if non_connector_indices else atoms
    candidate_diag = max(_bbox_diag(diagram_candidate.bbox), 1.0)
    grouped_indices = _cluster_source_components(cluster_atoms, candidate_diag)

    component_candidates: list[ReferenceComponentCandidate] = []
    group_debug: list[dict[str, Any]] = []
    connector_atoms = [atom for atom in atoms if atom["connector_like"]]

    for group_id, group in enumerate(grouped_indices):
        source_atoms = [cluster_atoms[idx] for idx in group]
        group_bbox = _merge_bbox([atom["bbox"] for atom in source_atoms])
        attachable_connectors = []
        expanded_group = _expand_bbox(group_bbox, candidate_diag * COMPONENT_CLUSTER_MARGIN_FRAC * 1.4)
        for atom in connector_atoms:
            if _bbox_intersects(expanded_group, atom["bbox"]):
                attachable_connectors.append(atom)

        merged_atoms = source_atoms + attachable_connectors
        bbox = _merge_bbox([atom["bbox"] for atom in merged_atoms])
        primitive_ids = [
            primitive["primitive_id"]
            for primitive in cleaned_primitives
            if _bbox_intersects(_expand_bbox(bbox, candidate_diag * 0.005), primitive["bbox"])
        ]
        near_blocks = [
            block for block in text_blocks
            if _bbox_intersects(_expand_bbox(bbox, candidate_diag * TEXT_MATCH_MARGIN_FRAC), block["bbox"])
        ]
        width, height = _bbox_dims(bbox)
        stats = {
            "bbox": bbox,
            "area_bbox": _bbox_area(bbox),
            "area_ratio": _bbox_area(bbox) / max(_bbox_area(diagram_candidate.bbox), 1.0),
            "width": width,
            "height": height,
            "aspect": max(width, height) / max(min(width, height), 1e-6),
            "primitive_count": len(primitive_ids),
            "n_source_components": len(merged_atoms),
            "n_segments": sum(atom["segment_count"] for atom in merged_atoms),
            "n_curves": sum(atom["curve_count"] for atom in merged_atoms),
            "n_rects": sum(atom["rect_count"] for atom in merged_atoms),
            "n_near_words": len(near_blocks),
            "connector_fraction": (
                sum(1 for atom in merged_atoms if atom["connector_like"]) / max(len(merged_atoms), 1)
            ),
        }
        score = _component_group_score(stats)
        candidate = ReferenceComponentCandidate(
            component_id=f"cmp_{group_id:03d}",
            page_index=page.page_index,
            bbox=bbox,
            anchor=_bbox_center(bbox),
            primitive_ids=primitive_ids,
            source_component_ids=[atom["component_id"] for atom in merged_atoms],
            text_block_ids=[],
            stats=stats,
            score=score,
        )
        component_candidates.append(candidate)
        group_debug.append({
            "component_id": candidate.component_id,
            "source_component_ids": candidate.source_component_ids,
            "source_atoms": merged_atoms,
            "stats": stats,
            "score": score,
        })

    component_candidates.sort(key=lambda item: (-item.score, -item.stats["primitive_count"], item.stats["area_bbox"]))
    for rank, item in enumerate(component_candidates, start=1):
        item.rank = rank

    source_component_debug = [
        {
            "component_id": atom["component_id"],
            "bbox": atom["bbox"],
            "primitive_count": atom["primitive_count"],
            "segment_count": atom["segment_count"],
            "curve_count": atom["curve_count"],
            "rect_count": atom["rect_count"],
            "aspect": atom["aspect"],
            "connector_like": atom["connector_like"],
            "source": atom["source"],
        }
        for atom in atoms
    ]
    return component_candidates, group_debug, source_component_debug


def _associate_text_blocks(
    component_candidates: list[ReferenceComponentCandidate],
    text_blocks: list[dict[str, Any]],
    diagram_bbox: list[float],
) -> list[ReferenceTextMatch]:
    if not component_candidates:
        return []

    diagram_diag = max(_bbox_diag(diagram_bbox), 1.0)
    matches: list[ReferenceTextMatch] = []
    for block in text_blocks:
        block_center = _bbox_center(block["bbox"])
        parsed = _parse_text_label(block["text"])
        scored: list[tuple[float, ReferenceComponentCandidate, str]] = []
        for candidate in component_candidates:
            gap = _bbox_gap(candidate.bbox, block["bbox"])
            dist = math.hypot(block_center[0] - candidate.anchor[0], block_center[1] - candidate.anchor[1])
            overlap_bonus = 1.0 if _bbox_intersects(_expand_bbox(candidate.bbox, diagram_diag * 0.004), block["bbox"]) else 0.0
            type_bonus = 0.5 if parsed.get("type_hint") else 0.0
            score = overlap_bonus + type_bonus + 1.0 / (0.15 + gap / max(diagram_diag, 1.0)) + 0.6 / (0.25 + dist / max(diagram_diag, 1.0))
            reason = f"gap={gap:.2f} dist={dist:.2f} overlap={overlap_bonus:.1f}"
            scored.append((score, candidate, reason))
        scored.sort(key=lambda item: -item[0])
        best_score, best_candidate, reason = scored[0]
        max_dist = diagram_diag * TEXT_MATCH_MAX_DIST_FRAC
        if _bbox_gap(best_candidate.bbox, block["bbox"]) <= max_dist or best_score >= 2.0:
            best_candidate.text_block_ids.append(block["text_block_id"])
            best_candidate.text_labels.append(block["text"])
            if parsed.get("type_hint") and not best_candidate.hypothesized_type:
                best_candidate.hypothesized_type = parsed["type_hint"]
            if parsed.get("dn_values"):
                best_candidate.attributes.setdefault("dn_values", [])
                best_candidate.attributes["dn_values"].extend(parsed["dn_values"])
            if parsed.get("size_ratios"):
                best_candidate.attributes.setdefault("size_ratios", [])
                best_candidate.attributes["size_ratios"].extend(parsed["size_ratios"])
            if parsed.get("materials"):
                best_candidate.attributes.setdefault("materials", [])
                best_candidate.attributes["materials"].extend(parsed["materials"])
            matches.append(
                ReferenceTextMatch(
                    match_id=f"match_{len(matches):03d}",
                    text_block_id=block["text_block_id"],
                    text=block["text"],
                    bbox=block["bbox"],
                    component_id=best_candidate.component_id,
                    score=best_score,
                    parsed=parsed,
                    reason=reason,
                )
            )
        else:
            matches.append(
                ReferenceTextMatch(
                    match_id=f"match_{len(matches):03d}",
                    text_block_id=block["text_block_id"],
                    text=block["text"],
                    bbox=block["bbox"],
                    component_id=None,
                    score=best_score,
                    parsed=parsed,
                    reason=f"unmatched {reason}",
                )
            )

    for candidate in component_candidates:
        if not candidate.hypothesized_type:
            if candidate.stats["n_curves"] >= 2:
                candidate.hypothesized_type = "Curved/Junction component"
            elif candidate.stats["n_rects"] >= 1:
                candidate.hypothesized_type = "Rect/Flanged component"
            else:
                candidate.hypothesized_type = "Unlabeled component"
        if "dn_values" in candidate.attributes:
            candidate.attributes["dn_values"] = sorted(set(candidate.attributes["dn_values"]))
        if "materials" in candidate.attributes:
            candidate.attributes["materials"] = sorted(set(candidate.attributes["materials"]))
    return matches


def _candidate_edge_rows(
    component_candidates: list[ReferenceComponentCandidate],
    diagram_bbox: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagram_diag = max(_bbox_diag(diagram_bbox), 1.0)
    for i, src in enumerate(component_candidates):
        distances = []
        for j, dst in enumerate(component_candidates):
            if i == j:
                continue
            gap = _bbox_gap(src.bbox, dst.bbox)
            dist = math.hypot(src.anchor[0] - dst.anchor[0], src.anchor[1] - dst.anchor[1])
            distances.append((j, gap, dist))
        distances.sort(key=lambda item: (item[1], item[2]))
        for j, gap, dist in distances[:EDGE_NEIGHBORS]:
            if i >= j:
                continue
            dst = component_candidates[j]
            if gap > diagram_diag * EDGE_MAX_GAP_FRAC:
                continue
            dx = abs(src.anchor[0] - dst.anchor[0])
            dy = abs(src.anchor[1] - dst.anchor[1])
            axis_align = max(
                1.0 - dx / max(diagram_diag * 0.18, 1.0),
                1.0 - dy / max(diagram_diag * 0.18, 1.0),
            )
            axis_align = max(0.0, axis_align)
            score = 1.8 / (0.20 + gap / diagram_diag) + 0.8 / (0.25 + dist / diagram_diag) + 0.9 * axis_align
            rows.append({
                "u": src.component_id,
                "v": dst.component_id,
                "gap": gap,
                "distance": dist,
                "axis_align": axis_align,
                "score": score,
            })
    rows.sort(key=lambda item: -item["score"])
    return rows


def _select_reference_edges(
    component_candidates: list[ReferenceComponentCandidate],
    diagram_bbox: list[float],
) -> list[ReferenceGraphEdge]:
    if len(component_candidates) <= 1:
        return []
    candidate_rows = _candidate_edge_rows(component_candidates, diagram_bbox)
    if not candidate_rows:
        ordered = sorted(component_candidates, key=lambda item: (item.anchor[0], item.anchor[1]))
        return [
            ReferenceGraphEdge(
                edge_id=f"edge_{idx:03d}",
                source=ordered[idx].component_id,
                target=ordered[idx + 1].component_id,
                score=0.0,
                stats={"source": "fallback_chain"},
            )
            for idx in range(len(ordered) - 1)
        ]

    if nx is not None:
        graph = nx.Graph()
        for component in component_candidates:
            graph.add_node(component.component_id)
        for row in candidate_rows:
            graph.add_edge(row["u"], row["v"], weight=row["score"], stats=row)
        tree = nx.maximum_spanning_tree(graph)
        edges = []
        for edge_idx, (u, v, attrs) in enumerate(tree.edges(data=True)):
            row = attrs.get("stats", {})
            edges.append(
                ReferenceGraphEdge(
                    edge_id=f"edge_{edge_idx:03d}",
                    source=u,
                    target=v,
                    score=float(attrs.get("weight", 0.0)),
                    stats={
                        "source": "mst_candidate_graph",
                        "gap": row.get("gap"),
                        "distance": row.get("distance"),
                        "axis_align": row.get("axis_align"),
                        "candidate_score": row.get("score"),
                    },
                )
            )
        return edges

    id_to_index = {component.component_id: idx for idx, component in enumerate(component_candidates)}
    uf = _UnionFind(len(component_candidates))
    edges: list[ReferenceGraphEdge] = []
    for row in candidate_rows:
        iu = id_to_index[row["u"]]
        iv = id_to_index[row["v"]]
        if uf.find(iu) == uf.find(iv):
            continue
        uf.union(iu, iv)
        edges.append(
            ReferenceGraphEdge(
                edge_id=f"edge_{len(edges):03d}",
                source=row["u"],
                target=row["v"],
                score=row["score"],
                stats={
                    "source": "kruskal_candidate_graph",
                    "gap": row["gap"],
                    "distance": row["distance"],
                    "axis_align": row["axis_align"],
                    "candidate_score": row["score"],
                },
            )
        )
        if len(edges) == len(component_candidates) - 1:
            break
    return edges


def _build_reference_nodes(component_candidates: list[ReferenceComponentCandidate]) -> list[ReferenceGraphNode]:
    return [
        ReferenceGraphNode(
            node_id=candidate.component_id,
            page_index=candidate.page_index,
            bbox=candidate.bbox,
            anchor=candidate.anchor,
            hypothesized_type=candidate.hypothesized_type,
            text_labels=candidate.text_labels,
            attributes=candidate.attributes,
            primitive_ids=candidate.primitive_ids,
            stats=candidate.stats,
        )
        for candidate in component_candidates
    ]


def _draw_crop_overlay(
    pdf_path: str,
    page: PdfPageExtraction,
    crop_bbox: list[float],
    out_path: Path,
    draw_fn,
) -> str | None:
    if cv2 is None:
        return None
    img = _render_pdf_page(Path(pdf_path), page.page_index)
    if img is None:
        return None
    scale_x = img.shape[1] / max(page.width, 1.0)
    scale_y = img.shape[0] / max(page.height, 1.0)
    x0, y0, x1, y1 = crop_bbox
    pad_x = max(20, int((x1 - x0) * scale_x * 0.05))
    pad_y = max(20, int((y1 - y0) * scale_y * 0.05))
    ix0 = max(0, int(x0 * scale_x) - pad_x)
    iy0 = max(0, int(y0 * scale_y) - pad_y)
    ix1 = min(img.shape[1], int(x1 * scale_x) + pad_x)
    iy1 = min(img.shape[0], int(y1 * scale_y) + pad_y)
    draw_fn(img, scale_x, scale_y)
    crop = img[iy0:iy1, ix0:ix1]
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def _save_primitive_overlay(
    pdf_path: str,
    page: PdfPageExtraction,
    crop_bbox: list[float],
    primitives: list[dict[str, Any]],
    out_path: Path,
) -> str | None:
    if cv2 is None:
        return None
    palette = {"segment": (0, 255, 0), "curve": (0, 165, 255), "rect": (255, 0, 0)}

    def draw(img, scale_x, scale_y):
        for item in primitives:
            x0, y0, x1, y1 = item["bbox"]
            color = palette.get(item["kind"], (255, 255, 255))
            cv2.rectangle(
                img,
                (int(x0 * scale_x), int(y0 * scale_y)),
                (int(x1 * scale_x), int(y1 * scale_y)),
                color,
                1,
            )

    return _draw_crop_overlay(pdf_path, page, crop_bbox, out_path, draw)


def _save_component_overlay(
    pdf_path: str,
    page: PdfPageExtraction,
    crop_bbox: list[float],
    components: list[ReferenceComponentCandidate],
    out_path: Path,
    show_rank: bool = True,
    show_labels: bool = False,
) -> str | None:
    if cv2 is None:
        return None

    def draw(img, scale_x, scale_y):
        for component in components:
            x0, y0, x1, y1 = component.bbox
            cv2.rectangle(
                img,
                (int(x0 * scale_x), int(y0 * scale_y)),
                (int(x1 * scale_x), int(y1 * scale_y)),
                (0, 255, 255),
                2,
            )
            text = component.component_id
            if show_rank and component.rank is not None:
                text = f"#{component.rank} {text}"
            if show_labels and component.text_labels:
                text = f"{text} | {component.text_labels[0]}"
            cv2.putText(
                img,
                text,
                (int(x0 * scale_x), max(24, int(y0 * scale_y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return _draw_crop_overlay(pdf_path, page, crop_bbox, out_path, draw)


def _save_text_match_overlay(
    pdf_path: str,
    page: PdfPageExtraction,
    crop_bbox: list[float],
    components: list[ReferenceComponentCandidate],
    matches: list[ReferenceTextMatch],
    out_path: Path,
) -> str | None:
    if cv2 is None:
        return None
    id_to_component = {component.component_id: component for component in components}

    def draw(img, scale_x, scale_y):
        for match in matches:
            x0, y0, x1, y1 = match.bbox
            color = (0, 255, 0) if match.component_id else (0, 0, 255)
            cv2.rectangle(
                img,
                (int(x0 * scale_x), int(y0 * scale_y)),
                (int(x1 * scale_x), int(y1 * scale_y)),
                color,
                2,
            )
            cv2.putText(
                img,
                match.text[:40],
                (int(x0 * scale_x), max(20, int(y0 * scale_y) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            if match.component_id and match.component_id in id_to_component:
                anchor = id_to_component[match.component_id].anchor
                word_center = _bbox_center(match.bbox)
                cv2.line(
                    img,
                    (int(word_center[0] * scale_x), int(word_center[1] * scale_y)),
                    (int(anchor[0] * scale_x), int(anchor[1] * scale_y)),
                    color,
                    1,
                )

    return _draw_crop_overlay(pdf_path, page, crop_bbox, out_path, draw)


def _save_reference_graph_overlay(
    pdf_path: str,
    page: PdfPageExtraction,
    crop_bbox: list[float],
    nodes: list[ReferenceGraphNode],
    edges: list[ReferenceGraphEdge],
    out_path: Path,
    mode: str,
) -> str | None:
    if cv2 is None:
        return None
    node_by_id = {node.node_id: node for node in nodes}

    def draw(img, scale_x, scale_y):
        for edge in edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                continue
            a = node_by_id[edge.source].anchor
            b = node_by_id[edge.target].anchor
            cv2.line(
                img,
                (int(a[0] * scale_x), int(a[1] * scale_y)),
                (int(b[0] * scale_x), int(b[1] * scale_y)),
                (0, 255, 0),
                3,
            )
        for node in nodes:
            cx, cy = node.anchor
            cv2.circle(img, (int(cx * scale_x), int(cy * scale_y)), 10, (0, 165, 255), -1)
            if mode == "numbered":
                label = node.node_id
            elif mode == "labels":
                label = node.text_labels[0] if node.text_labels else (node.hypothesized_type or node.node_id)
            else:
                label = node.hypothesized_type or node.node_id
            cv2.putText(
                img,
                label[:44],
                (int(cx * scale_x) + 6, int(cy * scale_y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return _draw_crop_overlay(pdf_path, page, crop_bbox, out_path, draw)


def build_reference_graph(
    diagram_candidate: DiagramCandidate,
    parsed_pdf: PdfPriorExtractionResult,
) -> ReferenceGraphResult:
    pdf_path = Path(parsed_pdf.pdf_path)
    debug_dirs = _init_reference_debug(parsed_pdf)
    warnings: list[str] = []

    try:
        page = parsed_pdf.pages[diagram_candidate.page_index]
    except IndexError:
        result = ReferenceGraphResult(
            pdf_path=str(pdf_path),
            status="invalid_page",
            page_index=None,
            diagram_candidate_id=diagram_candidate.candidate_id,
            nodes=[],
            edges=[],
            component_candidates=[],
            text_matches=[],
            debug_paths={name: str(path) for name, path in debug_dirs.items()},
            warnings=[f"Pagina {diagram_candidate.page_index} non presente nel parsing PDF."],
        )
        save_json(debug_dirs["json"] / "reference_graph.json", result.to_dict())
        return result

    raw_primitives = _collect_primitives_for_candidate(page, diagram_candidate)
    cleaned_primitives, cleanup_stats = _cleanup_primitives(page, diagram_candidate, raw_primitives)
    words_in_candidate = [
        word for word in page.words
        if _bbox_intersects(_expand_bbox(diagram_candidate.bbox, max(page.width, page.height) * 0.004), word.bbox)
    ]
    text_blocks = _group_words_into_blocks(words_in_candidate)
    component_candidates, component_group_debug, source_component_debug = _build_component_candidates(
        page,
        diagram_candidate,
        cleaned_primitives,
        text_blocks,
    )
    text_matches = _associate_text_blocks(component_candidates, text_blocks, diagram_candidate.bbox)
    nodes = _build_reference_nodes(component_candidates)
    edges = _select_reference_edges(component_candidates, diagram_candidate.bbox)

    if not component_candidates:
        warnings.append("Nessun component candidate costruito dal diagramma selezionato.")
    unlabeled = [component.component_id for component in component_candidates if not component.text_labels]
    if unlabeled:
        warnings.append(f"Componenti senza label associata: {', '.join(unlabeled[:10])}")

    crop_bbox = _expand_bbox(diagram_candidate.bbox, max(page.width, page.height) * 0.006)
    raw_img = _save_primitive_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        raw_primitives,
        debug_dirs["plots"] / f"page{page.page_index}_raw_diagram.png",
    )
    cleaned_img = _save_primitive_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        cleaned_primitives,
        debug_dirs["plots"] / f"page{page.page_index}_cleaned_primitives.png",
    )
    comp_overlay = _save_component_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        component_candidates,
        debug_dirs["plots"] / f"page{page.page_index}_component_candidates_overlay.png",
        show_rank=False,
        show_labels=False,
    )
    comp_ranked = _save_component_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        component_candidates,
        debug_dirs["plots"] / f"page{page.page_index}_component_candidates_ranked.png",
        show_rank=True,
        show_labels=False,
    )
    text_overlay = _save_text_match_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        component_candidates,
        text_matches,
        debug_dirs["plots"] / f"page{page.page_index}_text_matches_overlay.png",
    )
    graph_overlay = _save_reference_graph_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        nodes,
        edges,
        debug_dirs["plots"] / f"page{page.page_index}_reference_graph_overlay.png",
        mode="types",
    )
    graph_numbered = _save_reference_graph_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        nodes,
        edges,
        debug_dirs["plots"] / f"page{page.page_index}_reference_graph_numbered.png",
        mode="numbered",
    )
    graph_labels = _save_reference_graph_overlay(
        str(pdf_path),
        page,
        crop_bbox,
        nodes,
        edges,
        debug_dirs["plots"] / f"page{page.page_index}_reference_graph_labels.png",
        mode="labels",
    )

    cleanup_payload = asdict(cleanup_stats)
    cleanup_payload["diagram_bbox"] = diagram_candidate.bbox
    cleanup_payload["raw_preview_path"] = raw_img
    cleanup_payload["cleaned_preview_path"] = cleaned_img
    save_json(debug_dirs["json"] / "primitive_cleanup.json", cleanup_payload)
    save_json(
        debug_dirs["json"] / "component_candidates.json",
        {
            "diagram_candidate_id": diagram_candidate.candidate_id,
            "page_index": page.page_index,
            "source_component_debug": source_component_debug,
            "component_group_debug": component_group_debug,
            "component_candidates": [asdict(component) for component in component_candidates],
            "overlay_path": comp_overlay,
            "ranked_path": comp_ranked,
            "todo": [
                "axis extraction from pipe-like primitives",
                "symbol grouping refinement",
            ],
        },
    )
    save_json(
        debug_dirs["json"] / "text_matches.json",
        {
            "page_index": page.page_index,
            "text_blocks": text_blocks,
            "matches": [asdict(match) for match in text_matches],
            "overlay_path": text_overlay,
            "todo": [
                "text-to-component association refinement",
                "multi-line label merging",
            ],
        },
    )

    summary_stats = {
        "raw_primitive_count": len(raw_primitives),
        "cleaned_primitive_count": len(cleaned_primitives),
        "n_source_components": len(source_component_debug),
        "n_component_candidates": len(component_candidates),
        "n_text_blocks": len(text_blocks),
        "n_text_matches": sum(1 for match in text_matches if match.component_id is not None),
        "n_reference_nodes": len(nodes),
        "n_reference_edges": len(edges),
        "diagram_candidate_score": diagram_candidate.score,
        "diagram_bbox": diagram_candidate.bbox,
    }

    result = ReferenceGraphResult(
        pdf_path=str(pdf_path),
        status="ok" if nodes else "empty",
        page_index=page.page_index,
        diagram_candidate_id=diagram_candidate.candidate_id,
        nodes=nodes,
        edges=edges,
        component_candidates=component_candidates,
        text_matches=text_matches,
        debug_paths={
            "root": str(debug_dirs["root"]),
            "json": str(debug_dirs["json"]),
            "plots": str(debug_dirs["plots"]),
            "logs": str(debug_dirs["logs"]),
            "summary": str(debug_dirs["root"] / "summary.txt"),
            "raw_diagram": raw_img or "",
            "cleaned_primitives": cleaned_img or "",
            "component_candidates_overlay": comp_overlay or "",
            "component_candidates_ranked": comp_ranked or "",
            "text_matches_overlay": text_overlay or "",
            "reference_graph_overlay": graph_overlay or "",
            "reference_graph_numbered": graph_numbered or "",
            "reference_graph_labels": graph_labels or "",
        },
        summary_stats=summary_stats,
        warnings=warnings,
    )
    save_json(debug_dirs["json"] / "reference_graph.json", result.to_dict())

    summary_lines = [
        f"[ReferenceGraph] pdf={pdf_path}",
        f"[ReferenceGraph] page={page.page_index} diagram={diagram_candidate.candidate_id}",
        f"  raw_primitives={len(raw_primitives)}",
        f"  cleaned_primitives={len(cleaned_primitives)}",
        f"  source_components={len(source_component_debug)}",
        f"  component_candidates={len(component_candidates)}",
        f"  text_blocks={len(text_blocks)} matched={summary_stats['n_text_matches']}",
        f"  reference_nodes={len(nodes)} edges={len(edges)}",
    ]
    for warning in warnings:
        summary_lines.append(f"  [warn] {warning}")
    summary_lines.extend([
        "  TODO next steps:",
        "    - axis extraction from pipe-like primitives",
        "    - text-to-component association refinement",
        "    - prior graph construction refinement",
        "    - graph alignment with image observations",
    ])
    (debug_dirs["root"] / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (debug_dirs["logs"] / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (debug_dirs["logs"] / "reference_graph_log.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return result


def build_reference_graph_from_pdf_prior(
    extraction_result: PdfPriorExtractionResult,
    diagram_result: DiagramCandidateExtractionResult | None = None,
    candidate_rank: int = 1,
) -> ReferenceGraphResult:
    if diagram_result is None:
        from .pdf_graph_extraction import extract_diagram_candidates

        diagram_result = extract_diagram_candidates(extraction_result)

    ranked_candidates: list[DiagramCandidate] = []
    for _, items in sorted(diagram_result.page_candidates.items()):
        ranked_candidates.extend(items)
    ranked_candidates.sort(key=lambda item: (item.rank if item.rank is not None else 10_000, -item.score))
    if not ranked_candidates:
        return ReferenceGraphResult(
            pdf_path=extraction_result.pdf_path,
            status="missing_diagram_candidate",
            page_index=None,
            diagram_candidate_id=None,
            nodes=[],
            edges=[],
            component_candidates=[],
            text_matches=[],
            debug_paths={"root": str(_init_reference_debug(extraction_result)["root"])},
            warnings=["Nessun diagram candidate disponibile per costruire il reference graph."],
        )

    selected_index = max(0, candidate_rank - 1)
    selected = ranked_candidates[min(selected_index, len(ranked_candidates) - 1)]
    return build_reference_graph(selected, extraction_result)


def align_image_graph_to_pdf_prior(image_graph: Any, pdf_prior_graph: Any) -> dict[str, Any]:
    return {
        "status": "todo",
        "message": "Image-to-prior alignment non ancora implementato.",
        "todo_steps": [
            "graph alignment between image graph and reference graph",
            "node scoring observed-vs-prior",
            "support mismatch progetto/realtà",
            "prior-guided multi-view fusion",
        ],
    }
