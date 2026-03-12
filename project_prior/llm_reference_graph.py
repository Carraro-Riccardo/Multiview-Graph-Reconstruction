"""
Flusso PDF -> best cluster -> LLM multimodale -> reference graph JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

from config import (
    COMPONENTS_PDF_PATH,
    PDF_REFERENCE_GRAPH_PROVIDER,
    PDF_REFERENCE_GRAPH_OPENAI_MODEL,
    PDF_REFERENCE_GRAPH_GEMINI_MODEL,
    PDF_REFERENCE_GRAPH_OPENAI_API_KEY_ENV,
    PDF_REFERENCE_GRAPH_GEMINI_API_KEY_ENV,
)
from core.debug_utils import save_json
from .pdf_graph_extraction import (
    DiagramCandidate,
    DiagramCandidateExtractionResult,
    PdfPriorExtractionResult,
    _bbox_intersects,
    _expand_bbox,
    _render_pdf_page,
)
from .reference_graph_visualization import (
    save_reference_graph_overlay,
    save_reference_graph_schema_view,
)

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


LLM_DEBUG_DIRNAME = "llm_reference_graph"
ALLOWED_COMPONENT_TYPES = {
    "valve",
    "tee",
    "cross",
    "reducer",
    "pipe",
    "flange",
    "elbow",
    "unknown",
}

BASE_PROMPT = """You are a system specialized in understanding engineering diagrams, specifically hydraulic pipeline nodes.

Your task is to convert a diagram into a structured graph representation.

The diagram represents a hydraulic node composed of pipes, valves, tees, crosses, reducers and other components.

The diagram may contain text labels such as:
- SAR DN100
- SAR DN200
- TEE 200/100
- CROCE DN200
- RID FL 350/200
- CURVA PVC DE110
- TOULIPE DN100

These labels usually describe the component they are closest to.

Your job is to:

1. Identify the hydraulic components in the diagram
2. Associate nearby text labels to components
3. Infer the connections between components
4. Output a structured JSON graph

The output must strictly follow the schema described below.

Do NOT invent components that are not visible.
If uncertain, mark fields as null.

Prefer geometric proximity between components when inferring connections.
Text labels usually describe the nearest component.

Important:
Multiple components may have similar labels. Treat them as separate nodes.
The graph should represent physical connections between components.

Return ONLY valid JSON in the following format:

{
  "components": [
    {
      "id": "component_id",
      "type": "valve | tee | cross | reducer | pipe | flange | elbow | unknown",
      "label_text": "string or null",
      "dn_main": "number or null",
      "dn_branch": "number or null",
      "bbox": [x_min, y_min, x_max, y_max],
      "center": [x, y],
      "confidence": 0.0
    }
  ],
  "connections": [
    {
      "from": "component_id",
      "to": "component_id",
      "connection_type": "pipe_connection",
      "confidence": 0.0
    }
  ],
  "notes": [
    "optional reasoning notes"
  ]
}

YOU ARE ONLY INTERESTED IN THE COMPONENTS PRESENT IN THE ATTACHED REFERENCE PDF"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReferenceGraphComponent:
    id: str
    type: str
    label_text: str | None
    dn_main: float | None
    dn_branch: float | None
    bbox: list[float] | None
    center: list[float] | None
    confidence: float


@dataclass
class ReferenceGraphConnection:
    from_id: str
    to_id: str
    connection_type: str
    confidence: float


@dataclass
class ProviderResponse:
    provider: str
    model: str
    request_payload: dict[str, Any]
    raw_response: Any
    extracted_text: str | None
    error: str | None = None


@dataclass
class ReferenceGraphResult:
    pdf_path: str
    status: str
    provider: str
    model: str
    page_index: int | None
    diagram_candidate_id: str | None
    components: list[ReferenceGraphComponent]
    connections: list[ReferenceGraphConnection]
    notes: list[str] = field(default_factory=list)
    debug_paths: dict[str, str] = field(default_factory=dict)
    summary_stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw_response_path: str | None = None

    @property
    def nodes(self) -> list[ReferenceGraphComponent]:
        return self.components

    @property
    def edges(self) -> list[ReferenceGraphConnection]:
        return self.connections

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "page_index": self.page_index,
            "diagram_candidate_id": self.diagram_candidate_id,
            "components": [asdict(item) for item in self.components],
            "connections": [asdict(item) for item in self.connections],
            "notes": self.notes,
            "debug_paths": self.debug_paths,
            "summary_stats": self.summary_stats,
            "warnings": self.warnings,
            "raw_response_path": self.raw_response_path,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_llm_debug(parsed_pdf: PdfPriorExtractionResult) -> dict[str, Path]:
    root = Path(parsed_pdf.debug_paths.get("root", "."))
    llm_root = root / LLM_DEBUG_DIRNAME
    dirs = {
        "root": llm_root,
        "inputs": llm_root / "inputs",
        "prompts": llm_root / "prompts",
        "responses": llm_root / "responses",
        "graph": llm_root / "graph",
        "visuals": llm_root / "visuals",
        "summaries": llm_root / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _select_best_candidate(
    diagram_result: DiagramCandidateExtractionResult,
    candidate_rank: int = 1,
) -> DiagramCandidate | None:
    ranked: list[DiagramCandidate] = []
    for _, items in sorted(diagram_result.page_candidates.items()):
        ranked.extend(items)
    ranked.sort(key=lambda item: (item.rank if item.rank is not None else 10_000, -item.score))
    if not ranked:
        return None
    return ranked[min(max(0, candidate_rank - 1), len(ranked) - 1)]


def _render_best_candidate_images(
    parsed_pdf: PdfPriorExtractionResult,
    best_candidate: DiagramCandidate,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page = parsed_pdf.pages[best_candidate.page_index]
    image = _render_pdf_page(Path(parsed_pdf.pdf_path), page.page_index)
    if image is None or cv2 is None:
        return {}, []

    scale_x = image.shape[1] / max(page.width, 1.0)
    scale_y = image.shape[0] / max(page.height, 1.0)
    x0, y0, x1, y1 = best_candidate.bbox
    pad_x = max(24, int((x1 - x0) * scale_x * 0.06))
    pad_y = max(24, int((y1 - y0) * scale_y * 0.06))
    ix0 = max(0, int(x0 * scale_x) - pad_x)
    iy0 = max(0, int(y0 * scale_y) - pad_y)
    ix1 = min(image.shape[1], int(x1 * scale_x) + pad_x)
    iy1 = min(image.shape[0], int(y1 * scale_y) + pad_y)

    plain_crop = image[iy0:iy1, ix0:ix1].copy()
    bbox_crop = plain_crop.copy()

    selected_regions = []
    component_ids = set(best_candidate.component_ids)
    for component in parsed_pdf.pages[best_candidate.page_index].candidate_components:
        if component.component_id not in component_ids:
            continue
        bx0, by0, bx1, by1 = component.bbox
        rx0 = int(bx0 * scale_x) - ix0
        ry0 = int(by0 * scale_y) - iy0
        rx1 = int(bx1 * scale_x) - ix0
        ry1 = int(by1 * scale_y) - iy0
        cv2.rectangle(bbox_crop, (rx0, ry0), (rx1, ry1), (0, 255, 255), 2)
        selected_regions.append({
            "component_id": component.component_id,
            "bbox_pdf": component.bbox,
            "bbox_image": [rx0, ry0, rx1, ry1],
            "primitive_count": component.primitive_count,
            "primitive_kinds": component.primitive_kinds,
        })

    plain_path = out_dir / f"page{page.page_index}_best_cluster.png"
    bbox_path = out_dir / f"page{page.page_index}_best_cluster_with_bbox.png"
    cv2.imwrite(str(plain_path), plain_crop)
    cv2.imwrite(str(bbox_path), bbox_crop)
    return {
        "plain": plain_path,
        "with_bbox": bbox_path,
        "image_size": [int(plain_crop.shape[1]), int(plain_crop.shape[0])],
    }, selected_regions


def _candidate_words(
    parsed_pdf: PdfPriorExtractionResult,
    best_candidate: DiagramCandidate,
) -> list[dict[str, Any]]:
    page = parsed_pdf.pages[best_candidate.page_index]
    expanded = _expand_bbox(best_candidate.bbox, max(page.width, page.height) * 0.004)
    words = []
    for idx, word in enumerate(page.words):
        if _bbox_intersects(expanded, word.bbox):
            words.append({"word_index": idx, "text": word.text, "bbox": word.bbox})
    return words


def _reference_graph_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string", "enum": sorted(ALLOWED_COMPONENT_TYPES)},
                        "label_text": {"type": ["string", "null"]},
                        "dn_main": {"type": ["number", "null"]},
                        "dn_branch": {"type": ["number", "null"]},
                        "bbox": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "center": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "type", "label_text", "dn_main", "dn_branch", "bbox", "center", "confidence"],
                },
            },
            "connections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "connection_type": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["from", "to", "connection_type", "confidence"],
                },
            },
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["components", "connections", "notes"],
    }


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, data


def _encode_pdf(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class BaseProvider:
    provider_name = "base"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        cluster_image_path: Path,
        catalog_pdf_path: Path | None,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def generate(
        self,
        prompt: str,
        cluster_image_path: Path,
        catalog_pdf_path: Path | None,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        mime, b64_image = _encode_image(cluster_image_path)
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:{mime};base64,{b64_image}"},
        ]
        if catalog_pdf_path is not None and catalog_pdf_path.exists():
            b64_pdf = _encode_pdf(catalog_pdf_path)
            content.append({
                "type": "input_file",
                "filename": catalog_pdf_path.name,
                "file_data": f"data:application/pdf;base64,{b64_pdf}",
            })
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "reference_graph",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            raw = _http_post_json("https://api.openai.com/v1/responses", payload, headers)
            return ProviderResponse(
                provider=self.provider_name,
                model=self.model,
                request_payload=payload,
                raw_response=raw,
                extracted_text=_extract_openai_text(raw),
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return ProviderResponse(self.provider_name, self.model, payload, {"http_error": body, "status": exc.code}, None, f"HTTP {exc.code}")
        except Exception as exc:
            return ProviderResponse(self.provider_name, self.model, payload, {"exception": str(exc)}, None, str(exc))


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def generate(
        self,
        prompt: str,
        cluster_image_path: Path,
        catalog_pdf_path: Path | None,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        mime, b64_image = _encode_image(cluster_image_path)
        parts: list[dict[str, Any]] = [
            {"text": prompt},
            {"inline_data": {"mime_type": mime, "data": b64_image}},
        ]
        if catalog_pdf_path is not None and catalog_pdf_path.exists():
            b64_pdf = _encode_pdf(catalog_pdf_path)
            parts.append({"inline_data": {"mime_type": "application/pdf", "data": b64_pdf}})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        try:
            raw = _http_post_json(url, payload, headers)
            return ProviderResponse(
                provider=self.provider_name,
                model=self.model,
                request_payload=payload,
                raw_response=raw,
                extracted_text=_extract_gemini_text(raw),
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return ProviderResponse(self.provider_name, self.model, payload, {"http_error": body, "status": exc.code}, None, f"HTTP {exc.code}")
        except Exception as exc:
            return ProviderResponse(self.provider_name, self.model, payload, {"exception": str(exc)}, None, str(exc))


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def _extract_openai_text(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("output_text"), str) and raw["output_text"]:
        return raw["output_text"]
    output = raw.get("output")
    if isinstance(output, list):
        texts = []
        for item in output:
            for content in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    texts.append(content["text"])
        if texts:
            return "\n".join(texts)
    return None


def _extract_gemini_text(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return "\n".join(texts) if texts else None


# ---------------------------------------------------------------------------
# JSON validation / normalization
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(text), None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1]), None
        except Exception as exc:
            return None, str(exc)
    return None, "Nessun oggetto JSON trovato nella risposta."


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_bbox(bbox: Any, image_size: list[int]) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    vals = [_to_float_or_none(v) for v in bbox]
    if any(v is None for v in vals):
        return None
    x0, y0, x1, y1 = [float(v) for v in vals]
    w, h = image_size
    x0 = min(max(x0, 0.0), float(w))
    x1 = min(max(x1, 0.0), float(w))
    y0 = min(max(y0, 0.0), float(h))
    y1 = min(max(y1, 0.0), float(h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _normalize_center(center: Any, bbox: list[float] | None) -> list[float] | None:
    if isinstance(center, list) and len(center) == 2:
        vals = [_to_float_or_none(v) for v in center]
        if all(v is not None for v in vals):
            return [float(vals[0]), float(vals[1])]
    if bbox is not None:
        return [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]
    return None


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _validate_reference_graph_json(
    payload: dict[str, Any],
    image_size: list[int],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    components_in = payload.get("components", [])
    connections_in = payload.get("connections", [])
    notes_in = payload.get("notes", [])
    components: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(components_in if isinstance(components_in, list) else []):
        if not isinstance(item, dict):
            warnings.append(f"Componente {idx} scartato: non è un oggetto.")
            continue
        comp_id = str(item.get("id") or f"component_{idx:03d}")
        if comp_id in seen_ids:
            comp_id = f"{comp_id}_{idx:02d}"
            warnings.append(f"ID duplicato corretto: {comp_id}")
        seen_ids.add(comp_id)
        comp_type = str(item.get("type") or "unknown").strip().lower()
        if comp_type not in ALLOWED_COMPONENT_TYPES:
            warnings.append(f"Tipo non riconosciuto '{comp_type}' -> unknown")
            comp_type = "unknown"
        bbox = _normalize_bbox(item.get("bbox"), image_size)
        center = _normalize_center(item.get("center"), bbox)
        components.append({
            "id": comp_id,
            "type": comp_type,
            "label_text": (str(item["label_text"]).strip() if item.get("label_text") is not None else None),
            "dn_main": _to_float_or_none(item.get("dn_main")),
            "dn_branch": _to_float_or_none(item.get("dn_branch")),
            "bbox": bbox,
            "center": center,
            "confidence": _clamp_confidence(item.get("confidence")),
        })

    valid_ids = {item["id"] for item in components}
    connections: list[dict[str, Any]] = []
    for idx, item in enumerate(connections_in if isinstance(connections_in, list) else []):
        if not isinstance(item, dict):
            warnings.append(f"Connessione {idx} scartata: non è un oggetto.")
            continue
        src = str(item.get("from") or "").strip()
        dst = str(item.get("to") or "").strip()
        if src not in valid_ids or dst not in valid_ids:
            warnings.append(f"Connessione {idx} scartata: endpoint inesistenti ({src}, {dst})")
            continue
        if src == dst:
            warnings.append(f"Connessione {idx} scartata: self-loop su {src}")
            continue
        connections.append({
            "from": src,
            "to": dst,
            "connection_type": str(item.get("connection_type") or "pipe_connection"),
            "confidence": _clamp_confidence(item.get("confidence")),
        })

    notes = [str(item) for item in notes_in] if isinstance(notes_in, list) else []
    return {"components": components, "connections": connections, "notes": notes}, warnings


def _request_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))
    for item in clone.get("input", []):
        for content in item.get("content", []):
            if "image_url" in content and isinstance(content["image_url"], str):
                content["image_url"] = f"<base64_image len={len(content['image_url'])}>"
            if "file_data" in content and isinstance(content["file_data"], str):
                content["file_data"] = f"<base64_pdf len={len(content['file_data'])}>"
    for item in clone.get("contents", []):
        for part in item.get("parts", []):
            inline = part.get("inline_data")
            if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                inline["data"] = f"<base64_data len={len(inline['data'])}>"
    return clone


def _provider_from_config() -> tuple[BaseProvider | None, list[str]]:
    provider_name = (PDF_REFERENCE_GRAPH_PROVIDER or "gemini").strip().lower()
    if provider_name == "gemini":
        key = os.getenv(PDF_REFERENCE_GRAPH_GEMINI_API_KEY_ENV, "").strip()
        if not key:
            return None, [f"Chiave API mancante: {PDF_REFERENCE_GRAPH_GEMINI_API_KEY_ENV}"]
        return GeminiProvider(PDF_REFERENCE_GRAPH_GEMINI_MODEL, key), []
    if provider_name == "openai":
        key = os.getenv(PDF_REFERENCE_GRAPH_OPENAI_API_KEY_ENV, "").strip()
        if not key:
            return None, [f"Chiave API mancante: {PDF_REFERENCE_GRAPH_OPENAI_API_KEY_ENV}"]
        return OpenAIProvider(PDF_REFERENCE_GRAPH_OPENAI_MODEL, key), []
    return None, [f"Provider non supportato: {provider_name}"]


def _write_run_summary(out_dir: Path, lines: list[str]) -> Path:
    summary_path = out_dir / "run_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reference_graph_from_diagram(
    parsed_pdf: PdfPriorExtractionResult,
    diagram_result: DiagramCandidateExtractionResult,
    candidate_rank: int = 1,
) -> ReferenceGraphResult:
    debug_dirs = _init_llm_debug(parsed_pdf)
    out_dir = debug_dirs["root"]
    warnings: list[str] = []

    # --- select best candidate ---
    best_candidate = _select_best_candidate(diagram_result, candidate_rank=candidate_rank)
    if best_candidate is None:
        result = ReferenceGraphResult(
            pdf_path=parsed_pdf.pdf_path,
            status="no_diagram_candidate",
            provider=(PDF_REFERENCE_GRAPH_PROVIDER or "gemini"),
            model="",
            page_index=None,
            diagram_candidate_id=None,
            components=[],
            connections=[],
            warnings=["Nessun diagram candidate disponibile."],
            debug_paths={"root": str(out_dir)},
            summary_stats={"candidate_rank": candidate_rank},
        )
        save_json(debug_dirs["graph"] / "reference_graph_result.json", result.to_dict())
        save_json(debug_dirs["graph"] / "reference_graph.json", {"components": [], "connections": [], "notes": []})
        _write_run_summary(debug_dirs["summaries"], [f"pdf={parsed_pdf.pdf_path}", "status=no_diagram_candidate"])
        return result

    # --- render cluster image ---
    render_info, candidate_regions = _render_best_candidate_images(parsed_pdf, best_candidate, debug_dirs["inputs"])
    cluster_image_path: Path | None = render_info.get("plain")
    image_size: list[int] = render_info.get("image_size", [0, 0])
    nearby_words = _candidate_words(parsed_pdf, best_candidate)
    save_json(debug_dirs["inputs"] / "best_candidate_words.json", nearby_words)
    save_json(debug_dirs["inputs"] / "best_candidate_regions.json", candidate_regions)

    # --- catalog PDF (optional) ---
    catalog_pdf_path: Path | None = COMPONENTS_PDF_PATH if COMPONENTS_PDF_PATH.exists() else None
    if catalog_pdf_path is None:
        warnings.append(f"Catalog PDF non trovato: {COMPONENTS_PDF_PATH}")

    # --- prompt ---
    prompt_path = debug_dirs["prompts"] / "llm_prompt.txt"
    prompt_path.write_text(BASE_PROMPT, encoding="utf-8")

    summary_lines = [
        f"pdf={parsed_pdf.pdf_path}",
        f"page_index={best_candidate.page_index}",
        f"diagram_candidate_id={best_candidate.candidate_id}",
        f"candidate_rank={candidate_rank}",
        f"candidate_score={best_candidate.score:.4f}",
        f"cluster_image={cluster_image_path}",
        f"catalog_pdf={catalog_pdf_path}",
        f"nearby_words={len(nearby_words)}",
        f"candidate_regions={len(candidate_regions)}",
    ]
    request_summary_path = debug_dirs["summaries"] / "llm_request_summary.txt"
    request_summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if cluster_image_path is None:
        result = ReferenceGraphResult(
            pdf_path=parsed_pdf.pdf_path,
            status="missing_cluster_image",
            provider=(PDF_REFERENCE_GRAPH_PROVIDER or "gemini"),
            model="",
            page_index=best_candidate.page_index,
            diagram_candidate_id=best_candidate.candidate_id,
            components=[],
            connections=[],
            warnings=warnings + ["Impossibile renderizzare l'immagine del best cluster."],
            debug_paths={"root": str(out_dir), "llm_prompt": str(prompt_path)},
            summary_stats={"nearby_words": len(nearby_words), "candidate_regions": len(candidate_regions)},
        )
        save_json(debug_dirs["graph"] / "reference_graph_result.json", result.to_dict())
        save_json(debug_dirs["graph"] / "reference_graph.json", {"components": [], "connections": [], "notes": []})
        _write_run_summary(debug_dirs["summaries"], summary_lines + ["status=missing_cluster_image"])
        return result

    # --- provider ---
    provider, provider_warnings = _provider_from_config()
    warnings.extend(provider_warnings)
    if provider is None:
        result = ReferenceGraphResult(
            pdf_path=parsed_pdf.pdf_path,
            status="missing_provider_configuration",
            provider=(PDF_REFERENCE_GRAPH_PROVIDER or "gemini"),
            model="",
            page_index=best_candidate.page_index,
            diagram_candidate_id=best_candidate.candidate_id,
            components=[],
            connections=[],
            warnings=warnings,
            debug_paths={
                "root": str(out_dir),
                "best_cluster_image": str(cluster_image_path),
                "llm_prompt": str(prompt_path),
            },
            summary_stats={"nearby_words": len(nearby_words), "candidate_regions": len(candidate_regions)},
        )
        save_json(debug_dirs["graph"] / "reference_graph_result.json", result.to_dict())
        save_json(debug_dirs["graph"] / "reference_graph.json", {"components": [], "connections": [], "notes": []})
        _write_run_summary(debug_dirs["summaries"], summary_lines + ["status=missing_provider_configuration"] + [f"warning={w}" for w in warnings])
        return result

    # --- call LLM: 1 cluster image + 1 catalog PDF ---
    provider_response = provider.generate(
        BASE_PROMPT,
        cluster_image_path,
        catalog_pdf_path,
        _reference_graph_schema(),
    )
    save_json(debug_dirs["responses"] / "llm_request_payload.json", _request_payload_summary(provider_response.request_payload))
    raw_response_path = debug_dirs["responses"] / "llm_raw_response.txt"
    raw_response_path.write_text(
        json.dumps(provider_response.raw_response, ensure_ascii=False, indent=2)
        if isinstance(provider_response.raw_response, (dict, list))
        else str(provider_response.raw_response),
        encoding="utf-8",
    )
    if provider_response.extracted_text:
        (debug_dirs["responses"] / "llm_extracted_text.txt").write_text(provider_response.extracted_text, encoding="utf-8")

    if provider_response.error:
        warnings.append(f"{provider.provider_name} error: {provider_response.error}")
        result = ReferenceGraphResult(
            pdf_path=parsed_pdf.pdf_path,
            status="provider_error",
            provider=provider.provider_name,
            model=provider.model,
            page_index=best_candidate.page_index,
            diagram_candidate_id=best_candidate.candidate_id,
            components=[],
            connections=[],
            warnings=warnings,
            raw_response_path=str(raw_response_path),
            debug_paths={
                "root": str(out_dir),
                "best_cluster_image": str(cluster_image_path),
                "best_cluster_with_bbox": str(render_info.get("with_bbox", "")),
                "llm_prompt": str(prompt_path),
                "llm_request_summary": str(request_summary_path),
                "llm_request_payload": str(debug_dirs["responses"] / "llm_request_payload.json"),
                "llm_raw_response": str(raw_response_path),
            },
            summary_stats={"nearby_words": len(nearby_words), "candidate_regions": len(candidate_regions)},
        )
        save_json(debug_dirs["responses"] / "llm_parsed_response.json", {"status": "provider_error", "error": provider_response.error})
        save_json(debug_dirs["graph"] / "reference_graph_result.json", result.to_dict())
        save_json(debug_dirs["graph"] / "reference_graph.json", {"components": [], "connections": [], "notes": []})
        _write_run_summary(debug_dirs["summaries"], summary_lines + ["status=provider_error"] + [f"warning={w}" for w in warnings])
        return result

    # --- parse JSON ---
    extracted_text = provider_response.extracted_text or ""
    parsed_payload, parse_error = _extract_json_object(extracted_text)
    if parsed_payload is None:
        warnings.append(f"JSON parse error: {parse_error}")
        save_json(debug_dirs["responses"] / "llm_parsed_response.json", {"status": "parse_error", "error": parse_error, "extracted_text_preview": extracted_text[:4000]})
        result = ReferenceGraphResult(
            pdf_path=parsed_pdf.pdf_path,
            status="json_parse_error",
            provider=provider.provider_name,
            model=provider.model,
            page_index=best_candidate.page_index,
            diagram_candidate_id=best_candidate.candidate_id,
            components=[],
            connections=[],
            warnings=warnings,
            raw_response_path=str(raw_response_path),
            debug_paths={
                "root": str(out_dir),
                "best_cluster_image": str(cluster_image_path),
                "best_cluster_with_bbox": str(render_info.get("with_bbox", "")),
                "llm_prompt": str(prompt_path),
                "llm_request_summary": str(request_summary_path),
                "llm_request_payload": str(debug_dirs["responses"] / "llm_request_payload.json"),
                "llm_raw_response": str(raw_response_path),
                "llm_parsed_response": str(debug_dirs["responses"] / "llm_parsed_response.json"),
            },
            summary_stats={"nearby_words": len(nearby_words), "candidate_regions": len(candidate_regions)},
        )
        save_json(debug_dirs["graph"] / "reference_graph_result.json", result.to_dict())
        save_json(debug_dirs["graph"] / "reference_graph.json", {"components": [], "connections": [], "notes": []})
        _write_run_summary(debug_dirs["summaries"], summary_lines + ["status=json_parse_error"] + [f"warning={w}" for w in warnings])
        return result

    # --- validate & normalize ---
    normalized_payload, validation_warnings = _validate_reference_graph_json(parsed_payload, image_size=image_size)
    warnings.extend(validation_warnings)
    save_json(debug_dirs["responses"] / "llm_parsed_response.json", normalized_payload)
    save_json(debug_dirs["graph"] / "reference_graph.json", normalized_payload)
    overlay_paths = save_reference_graph_overlay(cluster_image_path, normalized_payload, debug_dirs["visuals"])
    schema_paths = save_reference_graph_schema_view(normalized_payload, debug_dirs["visuals"])

    components = [
        ReferenceGraphComponent(
            id=item["id"],
            type=item["type"],
            label_text=item.get("label_text"),
            dn_main=item.get("dn_main"),
            dn_branch=item.get("dn_branch"),
            bbox=item.get("bbox"),
            center=item.get("center"),
            confidence=item.get("confidence", 0.0),
        )
        for item in normalized_payload["components"]
    ]
    connections = [
        ReferenceGraphConnection(
            from_id=item["from"],
            to_id=item["to"],
            connection_type=item.get("connection_type", "pipe_connection"),
            confidence=item.get("confidence", 0.0),
        )
        for item in normalized_payload["connections"]
    ]

    result = ReferenceGraphResult(
        pdf_path=parsed_pdf.pdf_path,
        status="ok",
        provider=provider.provider_name,
        model=provider.model,
        page_index=best_candidate.page_index,
        diagram_candidate_id=best_candidate.candidate_id,
        components=components,
        connections=connections,
        notes=normalized_payload.get("notes", []),
        warnings=warnings,
        raw_response_path=str(raw_response_path),
        debug_paths={
            "root": str(out_dir),
            "best_cluster_image": str(cluster_image_path),
            "best_cluster_with_bbox": str(render_info.get("with_bbox", "")),
            "best_candidate_words": str(debug_dirs["inputs"] / "best_candidate_words.json"),
            "best_candidate_regions": str(debug_dirs["inputs"] / "best_candidate_regions.json"),
            "llm_prompt": str(prompt_path),
            "llm_request_summary": str(request_summary_path),
            "llm_request_payload": str(debug_dirs["responses"] / "llm_request_payload.json"),
            "llm_raw_response": str(raw_response_path),
            "llm_parsed_response": str(debug_dirs["responses"] / "llm_parsed_response.json"),
            "reference_graph_json": str(debug_dirs["graph"] / "reference_graph.json"),
            **overlay_paths,
            **schema_paths,
        },
        summary_stats={
            "nearby_words": len(nearby_words),
            "candidate_regions": len(candidate_regions),
            "catalog_pdf": str(catalog_pdf_path) if catalog_pdf_path else None,
            "component_count": len(components),
            "connection_count": len(connections),
        },
    )
    save_json(debug_dirs["graph"] / "reference_graph_result.json", result.to_dict())
    _write_run_summary(
        debug_dirs["summaries"],
        summary_lines + ["status=ok", f"components={len(components)}", f"connections={len(connections)}"]
        + [f"warning={w}" for w in warnings],
    )
    return result


def build_reference_graph_from_pdf_prior(
    extraction_result: PdfPriorExtractionResult,
    diagram_result: DiagramCandidateExtractionResult | None = None,
    candidate_rank: int = 1,
) -> ReferenceGraphResult:
    if diagram_result is None:
        from .pdf_graph_extraction import extract_diagram_candidates
        diagram_result = extract_diagram_candidates(extraction_result)
    return generate_reference_graph_from_diagram(
        extraction_result,
        diagram_result,
        candidate_rank=candidate_rank,
    )


def align_image_graph_to_pdf_prior(image_graph: Any, pdf_prior_graph: Any) -> dict[str, Any]:
    return {
        "status": "todo",
        "image_graph_nodes": len(getattr(image_graph, "nodes", [])) if image_graph is not None else 0,
        "pdf_prior_nodes": len(getattr(pdf_prior_graph, "nodes", [])) if pdf_prior_graph is not None else 0,
        "todo": [
            "graph alignment between observed image graph and reference graph",
            "scoring observed node vs prior component",
            "support mismatch progetto/realta'",
            "prior-guided multi-view fusion",
        ],
    }
