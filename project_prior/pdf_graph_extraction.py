"""
Estrazione iniziale del prior dal PDF di progetto.

Obiettivo di questo modulo:
  - aprire il PDF del progetto
  - verificare la presenza di contenuto vettoriale estraibile
  - estrarre primitive geometriche e testuali
  - produrre una rappresentazione intermedia serializzabile
  - salvare artifact diagnostici ispezionabili

In questa fase NON costruisce ancora il grafo di prior finale.
Fornisce invece una base robusta e leggibile per i passi successivi:
  - reference graph construction
  - image-to-prior alignment
  - prior-guided graph refinement
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math

from config import (
    WORK_DIR,
    PDF_CLUSTER_TOP_K,
    PDF_CLUSTER_PREVIEW_TOP_K,
    PDF_COMPONENT_MIN_PRIMITIVES,
    PDF_CLUSTER_CELL_DIVISOR,
    PDF_CLUSTER_EXPAND_FRAC,
)
from core.debug_utils import save_json


try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - dipendenza opzionale
    fitz = None

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover - dipendenza opzionale
    pdfplumber = None

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - dipendenza opzionale
    cv2 = None
    np = None


PDF_DEBUG_ROOT = WORK_DIR / "pdf_prior"


@dataclass
class PdfPrimitive:
    kind: str
    bbox: list[float]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PdfWord:
    text: str
    bbox: list[float]
    block_no: int | None = None
    line_no: int | None = None
    word_no: int | None = None
    source: str = "unknown"


@dataclass
class PdfCandidateComponent:
    component_id: str
    page_index: int
    bbox: list[float]
    primitive_count: int
    primitive_kinds: dict[str, int]
    source: str


@dataclass
class PdfPageExtraction:
    page_index: int
    width: float
    height: float
    segments: list[PdfPrimitive] = field(default_factory=list)
    curves: list[PdfPrimitive] = field(default_factory=list)
    rects: list[PdfPrimitive] = field(default_factory=list)
    words: list[PdfWord] = field(default_factory=list)
    candidate_components: list[PdfCandidateComponent] = field(default_factory=list)
    has_vector_content: bool = False
    preview_path: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PdfPriorExtractionResult:
    pdf_path: str
    status: str
    backend_used: str
    pages: list[PdfPageExtraction]
    warnings: list[str] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagramCandidate:
    candidate_id: str
    page_index: int
    bbox: list[float]
    component_ids: list[str]
    word_indices: list[int]
    stats: dict[str, Any]
    score: float
    rank: int | None = None
    debug_image_path: str | None = None


@dataclass
class DiagramCandidateExtractionResult:
    pdf_path: str
    status: str
    page_candidates: dict[int, list[DiagramCandidate]]
    warnings: list[str] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def init_pdf_debug(pdf_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    root = output_dir or (PDF_DEBUG_ROOT / pdf_path.stem)
    dirs = {
        "root": root,
        "json": root / "json",
        "previews": root / "previews",
        "logs": root / "logs",
        "clusters": root / "clusters",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _bbox_from_points(points: list[tuple[float, float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _merge_bbox(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _rect_to_list(rect: Any) -> list[float]:
    return [
        _safe_float(rect.x0),
        _safe_float(rect.y0),
        _safe_float(rect.x1),
        _safe_float(rect.y1),
    ]


def _drawing_component_from_fitz(page_index: int, draw_index: int, primitives: list[PdfPrimitive]) -> PdfCandidateComponent:
    counts: dict[str, int] = {}
    for prim in primitives:
        counts[prim.kind] = counts.get(prim.kind, 0) + 1
    bbox = _merge_bbox([prim.bbox for prim in primitives])
    return PdfCandidateComponent(
        component_id=f"page{page_index}_draw{draw_index}",
        page_index=page_index,
        bbox=bbox,
        primitive_count=len(primitives),
        primitive_kinds=counts,
        source="fitz_drawing",
    )


def _extract_page_with_fitz(doc: Any, page_index: int, debug_dirs: dict[str, Path]) -> PdfPageExtraction:
    page = doc.load_page(page_index)
    page_result = PdfPageExtraction(
        page_index=page_index,
        width=_safe_float(page.rect.width),
        height=_safe_float(page.rect.height),
    )

    drawing_components: list[PdfCandidateComponent] = []
    drawings = page.get_drawings()
    for draw_index, drawing in enumerate(drawings):
        local_primitives: list[PdfPrimitive] = []
        for item in drawing.get("items", []):
            op = item[0]
            if op == "l":
                p1, p2 = item[1], item[2]
                bbox = _bbox_from_points([(p1.x, p1.y), (p2.x, p2.y)])
                primitive = PdfPrimitive(
                    kind="segment",
                    bbox=bbox,
                    data={
                        "x0": _safe_float(p1.x),
                        "y0": _safe_float(p1.y),
                        "x1": _safe_float(p2.x),
                        "y1": _safe_float(p2.y),
                        "width": _safe_float(drawing.get("width", 0.0)),
                        "stroke": drawing.get("color"),
                    },
                )
                page_result.segments.append(primitive)
                local_primitives.append(primitive)
            elif op == "re":
                rect = item[1]
                primitive = PdfPrimitive(
                    kind="rect",
                    bbox=_rect_to_list(rect),
                    data={
                        "width": _safe_float(drawing.get("width", 0.0)),
                        "stroke": drawing.get("color"),
                        "fill": drawing.get("fill"),
                    },
                )
                page_result.rects.append(primitive)
                local_primitives.append(primitive)
            elif op in {"c", "qu"}:
                points = []
                for point in item[1:]:
                    if hasattr(point, "x") and hasattr(point, "y"):
                        points.append((point.x, point.y))
                bbox = _bbox_from_points(points) if points else _rect_to_list(drawing.get("rect", page.rect))
                primitive = PdfPrimitive(
                    kind="curve",
                    bbox=bbox,
                    data={
                        "operator": op,
                        "width": _safe_float(drawing.get("width", 0.0)),
                        "stroke": drawing.get("color"),
                        "fill": drawing.get("fill"),
                    },
                )
                page_result.curves.append(primitive)
                local_primitives.append(primitive)

        if local_primitives:
            drawing_components.append(
                _drawing_component_from_fitz(page_index, draw_index, local_primitives)
            )

    word_entries = page.get_text("words")
    for word in word_entries:
        x0, y0, x1, y1, text, block_no, line_no, word_no = word
        page_result.words.append(
            PdfWord(
                text=str(text),
                bbox=[_safe_float(x0), _safe_float(y0), _safe_float(x1), _safe_float(y1)],
                block_no=int(block_no),
                line_no=int(line_no),
                word_no=int(word_no),
                source="fitz",
            )
        )

    page_result.candidate_components = drawing_components
    page_result.has_vector_content = bool(page_result.segments or page_result.curves or page_result.rects)
    page_result.summary = {
        "n_segments": len(page_result.segments),
        "n_curves": len(page_result.curves),
        "n_rects": len(page_result.rects),
        "n_words": len(page_result.words),
        "n_candidate_components": len(page_result.candidate_components),
    }

    preview_path = _save_page_preview(page, page_result, debug_dirs, backend="fitz")
    page_result.preview_path = str(preview_path) if preview_path is not None else None
    return page_result


def _extract_page_with_pdfplumber(pdf: Any, page_index: int) -> PdfPageExtraction:
    page = pdf.pages[page_index]
    page_result = PdfPageExtraction(
        page_index=page_index,
        width=_safe_float(page.width),
        height=_safe_float(page.height),
    )

    line_components: list[PdfCandidateComponent] = []
    for idx, line in enumerate(page.lines):
        primitive = PdfPrimitive(
            kind="segment",
            bbox=[
                _safe_float(min(line["x0"], line["x1"])),
                _safe_float(min(line["top"], line["bottom"])),
                _safe_float(max(line["x0"], line["x1"])),
                _safe_float(max(line["top"], line["bottom"])),
            ],
            data={
                "x0": _safe_float(line["x0"]),
                "y0": _safe_float(line["top"]),
                "x1": _safe_float(line["x1"]),
                "y1": _safe_float(line["bottom"]),
                "width": _safe_float(line.get("linewidth", 0.0)),
            },
        )
        page_result.segments.append(primitive)
        line_components.append(
            PdfCandidateComponent(
                component_id=f"page{page_index}_line{idx}",
                page_index=page_index,
                bbox=primitive.bbox,
                primitive_count=1,
                primitive_kinds={"segment": 1},
                source="pdfplumber_line",
            )
        )

    for idx, rect in enumerate(page.rects):
        primitive = PdfPrimitive(
            kind="rect",
            bbox=[
                _safe_float(rect["x0"]),
                _safe_float(rect["top"]),
                _safe_float(rect["x1"]),
                _safe_float(rect["bottom"]),
            ],
            data={"width": _safe_float(rect.get("linewidth", 0.0))},
        )
        page_result.rects.append(primitive)
        line_components.append(
            PdfCandidateComponent(
                component_id=f"page{page_index}_rect{idx}",
                page_index=page_index,
                bbox=primitive.bbox,
                primitive_count=1,
                primitive_kinds={"rect": 1},
                source="pdfplumber_rect",
            )
        )

    for word in page.extract_words():
        page_result.words.append(
            PdfWord(
                text=str(word.get("text", "")),
                bbox=[
                    _safe_float(word.get("x0", 0.0)),
                    _safe_float(word.get("top", 0.0)),
                    _safe_float(word.get("x1", 0.0)),
                    _safe_float(word.get("bottom", 0.0)),
                ],
                source="pdfplumber",
            )
        )

    page_result.candidate_components = line_components
    page_result.has_vector_content = bool(page_result.segments or page_result.curves or page_result.rects)
    page_result.summary = {
        "n_segments": len(page_result.segments),
        "n_curves": len(page_result.curves),
        "n_rects": len(page_result.rects),
        "n_words": len(page_result.words),
        "n_candidate_components": len(page_result.candidate_components),
    }
    return page_result


def _save_page_preview(page: Any, page_result: PdfPageExtraction, debug_dirs: dict[str, Path], backend: str) -> Path | None:
    if fitz is None or cv2 is None or np is None:
        return None

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

    scale_x = img.shape[1] / max(page_result.width, 1.0)
    scale_y = img.shape[0] / max(page_result.height, 1.0)

    def draw_bbox(box: list[float], color: tuple[int, int, int], thickness: int) -> None:
        x0, y0, x1, y1 = box
        p0 = (int(x0 * scale_x), int(y0 * scale_y))
        p1 = (int(x1 * scale_x), int(y1 * scale_y))
        cv2.rectangle(img, p0, p1, color, thickness)

    for segment in page_result.segments:
        draw_bbox(segment.bbox, (0, 255, 0), 1)
    for rect in page_result.rects:
        draw_bbox(rect.bbox, (255, 0, 0), 1)
    for curve in page_result.curves:
        draw_bbox(curve.bbox, (0, 165, 255), 1)
    for word in page_result.words[:200]:
        draw_bbox(word.bbox, (255, 255, 0), 1)

    out_path = debug_dirs["previews"] / f"page_{page_result.page_index:02d}_{backend}.png"
    cv2.imwrite(str(out_path), img)
    return out_path


def _write_log(debug_dirs: dict[str, Path], lines: list[str]) -> Path:
    log_path = debug_dirs["logs"] / "extraction_log.txt"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _bbox_intersects(a: list[float], b: list[float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: list[float], margin: float) -> list[float]:
    return [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin]


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _component_counts(component: PdfCandidateComponent) -> tuple[int, int, int]:
    kinds = component.primitive_kinds
    return (
        int(kinds.get("segment", 0)),
        int(kinds.get("curve", 0)),
        int(kinds.get("rect", 0)),
    )


def _cell_keys_for_bbox(bbox: list[float], cell_size: float) -> list[tuple[int, int]]:
    x0 = int(bbox[0] // cell_size)
    y0 = int(bbox[1] // cell_size)
    x1 = int(bbox[2] // cell_size)
    y1 = int(bbox[3] // cell_size)
    keys = []
    for gx in range(x0, x1 + 1):
        for gy in range(y0, y1 + 1):
            keys.append((gx, gy))
    return keys


def _render_pdf_page(pdf_path: Path, page_index: int, scale: float = 2.0):
    if fitz is None or cv2 is None or np is None:
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return None


def _save_cluster_overlay(
    pdf_path: Path,
    page: PdfPageExtraction,
    candidates: list[DiagramCandidate],
    debug_dirs: dict[str, Path],
    top_k: int,
) -> str | None:
    if cv2 is None:
        return None
    img = _render_pdf_page(pdf_path, page.page_index)
    if img is None:
        return None

    scale_x = img.shape[1] / max(page.width, 1.0)
    scale_y = img.shape[0] / max(page.height, 1.0)
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 100, 0),
        (255, 0, 200),
        (80, 255, 255),
        (255, 255, 0),
    ]

    for idx, candidate in enumerate(candidates[:top_k], start=1):
        color = palette[(idx - 1) % len(palette)]
        x0, y0, x1, y1 = candidate.bbox
        p0 = (int(x0 * scale_x), int(y0 * scale_y))
        p1 = (int(x1 * scale_x), int(y1 * scale_y))
        cv2.rectangle(img, p0, p1, color, 3)
        cv2.putText(
            img,
            f"#{idx} sc={candidate.score:.2f}",
            (p0[0], max(24, p0[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    out_path = debug_dirs["clusters"] / f"page{page.page_index}_clusters_overlay.png"
    cv2.imwrite(str(out_path), img)
    return str(out_path)


def _save_cluster_preview(
    pdf_path: Path,
    page: PdfPageExtraction,
    candidate: DiagramCandidate,
    debug_dirs: dict[str, Path],
) -> str | None:
    if cv2 is None:
        return None
    img = _render_pdf_page(pdf_path, page.page_index)
    if img is None:
        return None

    scale_x = img.shape[1] / max(page.width, 1.0)
    scale_y = img.shape[0] / max(page.height, 1.0)
    x0, y0, x1, y1 = candidate.bbox
    pad_x = max(10, int((x1 - x0) * scale_x * 0.05))
    pad_y = max(10, int((y1 - y0) * scale_y * 0.05))
    ix0 = max(0, int(x0 * scale_x) - pad_x)
    iy0 = max(0, int(y0 * scale_y) - pad_y)
    ix1 = min(img.shape[1], int(x1 * scale_x) + pad_x)
    iy1 = min(img.shape[0], int(y1 * scale_y) + pad_y)
    crop = img[iy0:iy1, ix0:ix1].copy()
    cv2.rectangle(crop, (pad_x, pad_y), (crop.shape[1] - pad_x, crop.shape[0] - pad_y), (0, 255, 0), 2)
    out_path = debug_dirs["clusters"] / f"page{page.page_index}_cluster_rank_{candidate.rank:02d}.png"
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def _cluster_page_components(page: PdfPageExtraction) -> list[list[int]]:
    components = [
        comp for comp in page.candidate_components
        if comp.primitive_count >= PDF_COMPONENT_MIN_PRIMITIVES and _bbox_area(comp.bbox) > 0.0
    ]
    if not components:
        return []

    cell_size = max(page.width, page.height) / max(PDF_CLUSTER_CELL_DIVISOR, 1.0)
    margin = max(page.width, page.height) * PDF_CLUSTER_EXPAND_FRAC
    expanded_boxes = [_expand_bbox(comp.bbox, margin) for comp in components]

    grid: dict[tuple[int, int], list[int]] = {}
    uf = _UnionFind(len(components))

    for idx, bbox in enumerate(expanded_boxes):
        seen_neighbors: set[int] = set()
        for key in _cell_keys_for_bbox(bbox, cell_size):
            for other in grid.get(key, []):
                if other in seen_neighbors:
                    continue
                seen_neighbors.add(other)
                if _bbox_intersects(bbox, expanded_boxes[other]):
                    uf.union(idx, other)
            grid.setdefault(key, []).append(idx)

    groups: dict[int, list[int]] = {}
    for idx in range(len(components)):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    return list(groups.values())


def _score_cluster(
    page: PdfPageExtraction,
    cluster_indices: list[int],
    filtered_components: list[PdfCandidateComponent],
) -> tuple[dict[str, Any], float]:
    cluster_components = [filtered_components[i] for i in cluster_indices]
    bbox = _merge_bbox([comp.bbox for comp in cluster_components])
    bbox_area = _bbox_area(bbox)
    page_area = max(page.width * page.height, 1.0)
    area_ratio = bbox_area / page_area

    n_segments = 0
    n_curves = 0
    n_rects = 0
    primitive_count = 0
    for comp in cluster_components:
        segs, curves, rects = _component_counts(comp)
        n_segments += segs
        n_curves += curves
        n_rects += rects
        primitive_count += comp.primitive_count

    word_indices: list[int] = []
    expanded_bbox = _expand_bbox(bbox, max(page.width, page.height) * 0.003)
    for idx, word in enumerate(page.words):
        if _bbox_intersects(expanded_bbox, word.bbox):
            word_indices.append(idx)

    density_segments = n_segments / max(bbox_area / 1_000_000.0, 1e-6)
    density_curves = n_curves / max(bbox_area / 1_000_000.0, 1e-6)
    word_density = len(word_indices) / max(bbox_area / 1_000_000.0, 1e-6)

    compactness_bonus = 1.0 / max(area_ratio, 1e-6)
    moderate_area_bonus = 1.0 - min(abs(area_ratio - 0.12) / 0.12, 1.0)

    score = (
        1.4 * math.log1p(len(cluster_components))
        + 0.9 * math.log1p(primitive_count)
        + 0.5 * math.log1p(n_segments)
        + 0.7 * math.log1p(n_curves)
        + 0.6 * math.log1p(len(word_indices))
        + 0.18 * math.log1p(density_segments)
        + 0.22 * math.log1p(density_curves + 1.0)
        + 0.10 * math.log1p(word_density + 1.0)
        + 0.35 * moderate_area_bonus
    )

    if area_ratio > 0.65:
        score -= 3.0
    elif area_ratio > 0.35:
        score -= 1.2
    elif area_ratio < 0.0005:
        score -= 2.5
    elif area_ratio < 0.002:
        score -= 1.0

    stats = {
        "bbox": bbox,
        "area_bbox": bbox_area,
        "area_ratio": area_ratio,
        "component_count": len(cluster_components),
        "primitive_count": primitive_count,
        "n_segments": n_segments,
        "n_curves": n_curves,
        "n_rects": n_rects,
        "n_words": len(word_indices),
        "density_segments": density_segments,
        "density_curves": density_curves,
        "density_words": word_density,
        "compactness_bonus": compactness_bonus,
        "moderate_area_bonus": moderate_area_bonus,
    }
    return stats, score


def extract_diagram_candidates(
    parsed_pdf: PdfPriorExtractionResult,
    top_k: int = PDF_CLUSTER_TOP_K,
) -> DiagramCandidateExtractionResult:
    """
    Raggruppa le primitive estratte in cluster diagrammatici plausibili.

    Strategia:
      - usa i candidate_components geometrici del parsing PDF come atomi
      - costruisce connected components spaziali con tolleranza locale
      - calcola feature geometriche/testuali per ogni cluster
      - ordina i cluster per score
      - salva overlay e crop dei top candidati
    """
    pdf_path = Path(parsed_pdf.pdf_path)
    debug_root = Path(parsed_pdf.debug_paths.get("root", PDF_DEBUG_ROOT / pdf_path.stem))
    debug_dirs = init_pdf_debug(pdf_path, debug_root)
    warnings: list[str] = []
    page_candidates: dict[int, list[DiagramCandidate]] = {}
    log_lines = [f"[DiagramCandidates] pdf={pdf_path}"]

    for page in parsed_pdf.pages:
        filtered_components = [
            comp for comp in page.candidate_components
            if comp.primitive_count >= PDF_COMPONENT_MIN_PRIMITIVES and _bbox_area(comp.bbox) > 0.0
        ]
        log_lines.append(
            f"[Page {page.page_index}] raw_components={len(page.candidate_components)} "
            f"filtered_components={len(filtered_components)} words={len(page.words)}"
        )
        if not filtered_components:
            page_candidates[page.page_index] = []
            continue

        cluster_groups = _cluster_page_components(page)
        candidates: list[DiagramCandidate] = []
        for cluster_id, cluster_indices in enumerate(cluster_groups):
            stats, score = _score_cluster(page, cluster_indices, filtered_components)
            bbox = stats["bbox"]
            component_ids = [filtered_components[i].component_id for i in cluster_indices]
            expanded_bbox = _expand_bbox(bbox, max(page.width, page.height) * 0.003)
            word_indices = [idx for idx, word in enumerate(page.words) if _bbox_intersects(expanded_bbox, word.bbox)]
            candidates.append(
                DiagramCandidate(
                    candidate_id=f"page{page.page_index}_cluster{cluster_id:04d}",
                    page_index=page.page_index,
                    bbox=bbox,
                    component_ids=component_ids,
                    word_indices=word_indices,
                    stats=stats,
                    score=score,
                )
            )

        candidates.sort(key=lambda cand: (-cand.score, -cand.stats["primitive_count"], cand.stats["area_bbox"]))
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            candidate.rank = rank
            if rank <= PDF_CLUSTER_PREVIEW_TOP_K:
                candidate.debug_image_path = _save_cluster_preview(pdf_path, page, candidate, debug_dirs)
            log_lines.append(
                f"  [Rank {rank}] {candidate.candidate_id} "
                f"score={candidate.score:.3f} comps={candidate.stats['component_count']} "
                f"prims={candidate.stats['primitive_count']} words={candidate.stats['n_words']} "
                f"area_ratio={candidate.stats['area_ratio']:.4f} bbox={candidate.bbox}"
            )

        overlay_path = _save_cluster_overlay(pdf_path, page, candidates, debug_dirs, top_k=top_k)
        page_candidates[page.page_index] = candidates[:top_k]
        save_json(
            debug_dirs["clusters"] / f"page{page.page_index}_clusters.json",
            {
                "page_index": page.page_index,
                "overlay_path": overlay_path,
                "total_candidates": len(candidates),
                "top_k": top_k,
                "top_candidates": [asdict(candidate) for candidate in candidates[:top_k]],
                "ranked_candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "rank": rank,
                        "score": candidate.score,
                        "bbox": candidate.bbox,
                        "stats": candidate.stats,
                        "component_ids": candidate.component_ids,
                        "word_indices": candidate.word_indices,
                    }
                    for rank, candidate in enumerate(candidates, start=1)
                ],
            },
        )

    result = DiagramCandidateExtractionResult(
        pdf_path=str(pdf_path),
        status="ok" if page_candidates else "empty",
        page_candidates=page_candidates,
        warnings=warnings,
        debug_paths={name: str(path) for name, path in debug_dirs.items()},
        parameters={
            "top_k": top_k,
            "component_min_primitives": PDF_COMPONENT_MIN_PRIMITIVES,
            "cluster_cell_divisor": PDF_CLUSTER_CELL_DIVISOR,
            "cluster_expand_frac": PDF_CLUSTER_EXPAND_FRAC,
        },
    )
    save_json(debug_dirs["json"] / "diagram_candidates.json", result.to_dict())
    cluster_log = debug_dirs["logs"] / "diagram_candidates_log.txt"
    cluster_log.write_text("\n".join(log_lines), encoding="utf-8")
    return result


def extract_pdf_prior(pdf_path: str | Path, output_dir: str | Path | None = None) -> PdfPriorExtractionResult:
    """
    Estrae una rappresentazione intermedia del progetto PDF.

    Strategia:
      1. prova il parsing vettoriale con PyMuPDF
      2. in fallback usa pdfplumber
      3. salva JSON + preview + log di debug

    Ritorna un oggetto serializzabile ispezionabile, NON ancora il prior graph.
    """
    pdf_path = Path(pdf_path)
    debug_dirs = init_pdf_debug(pdf_path, Path(output_dir) if output_dir is not None else None)
    warnings: list[str] = []
    log_lines = [f"[PDF] path={pdf_path}"]

    if not pdf_path.exists():
        result = PdfPriorExtractionResult(
            pdf_path=str(pdf_path),
            status="missing_pdf",
            backend_used="none",
            pages=[],
            warnings=[f"PDF non trovato: {pdf_path}"],
            debug_paths={name: str(path) for name, path in debug_dirs.items()},
        )
        save_json(debug_dirs["json"] / "pdf_prior_extraction.json", result.to_dict())
        _write_log(debug_dirs, log_lines + [f"[PDF] missing: {pdf_path}"])
        return result

    pages: list[PdfPageExtraction] = []
    backend_used = "none"

    if fitz is not None:
        log_lines.append("[PDF] backend=fitz")
        try:
            doc = fitz.open(pdf_path)
            for page_index in range(doc.page_count):
                page_result = _extract_page_with_fitz(doc, page_index, debug_dirs)
                pages.append(page_result)
                log_lines.append(
                    f"[Page {page_index}] fitz "
                    f"segments={page_result.summary['n_segments']} "
                    f"curves={page_result.summary['n_curves']} "
                    f"rects={page_result.summary['n_rects']} "
                    f"words={page_result.summary['n_words']} "
                    f"components={page_result.summary['n_candidate_components']}"
                )
            backend_used = "fitz"
        except Exception as exc:  # pragma: no cover - robust fallback
            warnings.append(f"PyMuPDF fallito: {exc}")
            log_lines.append(f"[PDF] fitz failed: {exc}")
            pages = []

    if not pages and pdfplumber is not None:
        log_lines.append("[PDF] backend=pdfplumber")
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_index in range(len(pdf.pages)):
                    page_result = _extract_page_with_pdfplumber(pdf, page_index)
                    pages.append(page_result)
                    log_lines.append(
                        f"[Page {page_index}] pdfplumber "
                        f"segments={page_result.summary['n_segments']} "
                        f"rects={page_result.summary['n_rects']} "
                        f"words={page_result.summary['n_words']} "
                        f"components={page_result.summary['n_candidate_components']}"
                    )
            backend_used = "pdfplumber"
        except Exception as exc:  # pragma: no cover - robust fallback
            warnings.append(f"pdfplumber fallito: {exc}")
            log_lines.append(f"[PDF] pdfplumber failed: {exc}")
            pages = []

    if not pages:
        if fitz is None and pdfplumber is None:
            warnings.append("Nessun backend PDF disponibile. Installa PyMuPDF o pdfplumber.")
        status = "failed"
    else:
        status = "ok"

    result = PdfPriorExtractionResult(
        pdf_path=str(pdf_path),
        status=status,
        backend_used=backend_used,
        pages=pages,
        warnings=warnings,
        debug_paths={name: str(path) for name, path in debug_dirs.items()},
    )

    save_json(debug_dirs["json"] / "pdf_prior_extraction.json", result.to_dict())
    summary_payload = {
        "pdf_path": str(pdf_path),
        "status": status,
        "backend_used": backend_used,
        "n_pages": len(pages),
        "warnings": warnings,
        "page_summaries": [page.summary for page in pages],
    }
    save_json(debug_dirs["json"] / "pdf_prior_summary.json", summary_payload)
    log_lines.append(f"[PDF] status={status}")
    log_lines.append(f"[PDF] warnings={len(warnings)}")
    _write_log(debug_dirs, log_lines)
    return result


def build_reference_graph_from_pdf_prior(
    extraction_result: PdfPriorExtractionResult,
    diagram_result=None,
    candidate_rank: int = 1,
):
    """
    Wrapper locale per mantenere stabile l'API del modulo PDF.
    L'implementazione effettiva vive in `project_prior.llm_reference_graph`.
    """
    from .llm_reference_graph import build_reference_graph_from_pdf_prior as _impl

    return _impl(
        extraction_result,
        diagram_result=diagram_result,
        candidate_rank=candidate_rank,
    )


def align_image_graph_to_pdf_prior(image_graph: Any, pdf_prior_graph: Any) -> dict[str, Any]:
    from .llm_reference_graph import align_image_graph_to_pdf_prior as _impl

    return _impl(image_graph, pdf_prior_graph)


def run_pdf_prior_extraction(pdf_path: str | Path, output_dir: str | Path | None = None) -> PdfPriorExtractionResult:
    """
    Entry point leggibile per lanciare il flusso PDF:
      parsing -> diagram candidates -> reference graph.
    """
    result = extract_pdf_prior(pdf_path, output_dir=output_dir)
    diagram_result = extract_diagram_candidates(result)
    print(f"[PDF] file={result.pdf_path}")
    print(f"[PDF] status={result.status} backend={result.backend_used} pages={len(result.pages)}")
    if result.warnings:
        for warning in result.warnings:
            print(f"[PDF][warn] {warning}")
    for page in result.pages:
        print(
            f"  [Page {page.page_index}] size=({page.width:.1f}, {page.height:.1f}) "
            f"segments={len(page.segments)} curves={len(page.curves)} rects={len(page.rects)} "
            f"words={len(page.words)} components={len(page.candidate_components)}"
        )
    print(f"[PDF] debug_dir={result.debug_paths.get('root')}")
    for page_index, candidates in sorted(diagram_result.page_candidates.items()):
        print(f"[PDF][Candidates] page={page_index} top={len(candidates)}")
        for candidate in candidates[: min(5, len(candidates))]:
            print(
                f"  [Rank {candidate.rank}] {candidate.candidate_id} "
                f"score={candidate.score:.3f} bbox={candidate.bbox} "
                f"comps={candidate.stats['component_count']} words={candidate.stats['n_words']}"
            )
    reference_graph = build_reference_graph_from_pdf_prior(result, diagram_result=diagram_result)
    print(
        f"[PDF][ReferenceGraph] status={reference_graph.status} "
        f"nodes={len(reference_graph.nodes)} edges={len(reference_graph.edges)} "
        f"debug={reference_graph.debug_paths.get('root')}"
    )
    for warning in reference_graph.warnings:
        print(f"[PDF][ReferenceGraph][warn] {warning}")
    return result
