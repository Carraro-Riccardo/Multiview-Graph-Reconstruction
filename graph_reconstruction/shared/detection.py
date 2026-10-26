"""
Funzioni di discovery, pre-processing, detection YOLO e decodifica Shurikode.
"""

import re
import json
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

from graph_reconstruction.config import CONF_THRESHOLD, YOLO_CONF, YOLO_IOU, YOLO_IMGSZ, COMPONENT_INFO, DIAMETER_CODES, WORK_DIR, YOLO_MODEL_PATH

_MODEL_BUNDLE = None
__all__ = ["discover_nodes", "detect_and_decode_cached"]


def _resource_metrics_enabled() -> bool:
    return os.getenv("PIPELINE_RESOURCE_METRICS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _reset_cuda_peak_metrics() -> None:
    if not _resource_metrics_enabled():
        return
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _save_cuda_metrics(images_completed: int) -> None:
    """Salva i contatori dell'allocatore PyTorch usati dai benchmark CUDA/MPS."""
    if not _resource_metrics_enabled():
        return
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        payload = {
            "source": "torch_cuda_allocator",
            "component": "yolo_parent_process",
            "device_index": int(device),
            "device_name": torch.cuda.get_device_name(device),
            "images_completed": int(images_completed),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        metrics_path = WORK_DIR / "benchmark_yolo_cuda.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _get_model_bundle():
    """Inizializza modelli e pre-processing una sola volta per processo.

    Gli import pesanti restano locali: comandi che leggono soltanto cache o
    configurazione non devono caricare preventivamente PyTorch e Ultralytics.
    """
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _reset_cuda_peak_metrics()
        from shurikodecontext import Dec  # noqa: PLC0415
        from ultralytics import YOLO  # noqa: PLC0415

        class _DataAugmentor:
            def __init__(self, clahe_clip_limit: float = 1.0, clahe_tile_grid: tuple = (16, 16)):
                self.clahe_clip_limit = clahe_clip_limit
                self.clahe_tile_grid = clahe_tile_grid

            def augment(self, image_input) -> Image.Image:
                if isinstance(image_input, (str, Path)):
                    image = Image.open(image_input).convert("RGB")
                elif isinstance(image_input, np.ndarray):
                    image = Image.fromarray(
                        cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                        if image_input.ndim == 3 else image_input
                    )
                elif isinstance(image_input, Image.Image):
                    image = image_input
                else:
                    raise ValueError(f"Input type {type(image_input)} not supported")

                gray = np.array(image.convert("L"), dtype=np.uint8)
                enhanced = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=self.clahe_tile_grid,
                ).apply(gray)
                return Image.fromarray(enhanced, mode="L").convert("RGB")

        augmentor = _DataAugmentor()
        decoder_engine = Dec("L")
        yolo_model = YOLO(str(YOLO_MODEL_PATH))

        _MODEL_BUNDLE = {
            "augmentor": augmentor,
            "decoder_engine": decoder_engine,
            "yolo_model": yolo_model,
        }
    return _MODEL_BUNDLE

def discover_nodes(images_dir: Path) -> dict:
    """Raccoglie le immagini per nodo dal pattern node<id>_<frame>.jpg.

    L'ID del nodo è tutto ciò che segue "node" fino all'ultimo segmento
    puramente numerico (il frame index). Esempi:
      node17_0042.jpg    -> id "17"
      node17_2_0042.jpg  -> id "17_2"
      node15_1_0042.jpg  -> id "15_1"
    """
    node_map = defaultdict(list)
    for img in sorted(images_dir.glob("node*_*.jpg")):
        m = re.match(r"^node(.+)_(\d+)\.jpg$", img.name)
        if m:
            node_map[m.group(1)].append(img)

    def _sort_key(node_id: str):
        # Gli ID composti, ad esempio 15_2, devono mantenere un ordine numerico.
        return [int(p) if p.isdigit() else p for p in node_id.split("_")]

    return dict(sorted(node_map.items(), key=lambda kv: _sort_key(kv[0])))



def _cache_path(image_path: Path) -> Path:
    return CACHE_DIR / f"{image_path.stem}.json"


CACHE_VERSION = 2


def _cache_signature() -> dict:
    """Parametri che invalidano una detection salvata quando cambiano."""
    return {
        "version": CACHE_VERSION,
        "yolo_conf": float(YOLO_CONF),
        "yolo_iou": float(YOLO_IOU),
        "yolo_imgsz": int(YOLO_IMGSZ),
        "conf_threshold": float(CONF_THRESHOLD),
    }



def _load_cached_detections(image_path: Path) -> list | None:
    """Restituisce ``None`` per cache assente, obsoleta o non leggibile."""
    cache_path = _cache_path(image_path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("signature") != _cache_signature():
            return None
        return payload.get("detections", [])
    except Exception:
        return None



def _save_cached_detections(image_path: Path, dets: list) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(image_path)
    payload = {
        "image": image_path.name,
        "signature": _cache_signature(),
        "detections": dets,
    }
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)



def detect_and_decode_cached(image_path: Path, force: bool = False) -> tuple:
    """Restituisce detection e immagine, riutilizzando la cache quando valida.

    ``force=True`` ricalcola il risultato e sovrascrive la cache della sola
    immagine richiesta.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return [], None

    if not force:
        cached = _load_cached_detections(image_path)
        if cached is not None:
            print(f"    [DetectCache] {image_path.name} cache hit ({len(cached)} det)")
            return cached, img_bgr
    print(f"    [DetectCache] {image_path.name} cache miss -> running detection")

    dets, _ = _detect_and_decode(image_path)
    _save_cached_detections(image_path, dets)
    return dets, img_bgr

CACHE_DIR = WORK_DIR / "detection_cache"

def _bbox_center(bbox_norm: list) -> tuple:
    x1, y1, x2, y2 = bbox_norm
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _decode_patch(img_bgr: np.ndarray, bbox_norm: list) -> tuple:
    """
    Estrae la patch dall'immagine BGR e la decodifica.
    Ritorna ``(code, conf)`` oppure ``(None, 0.0)`` se la patch è troppo piccola
    o il codice non è valido.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    if (px2 - px1) < 8 or (py2 - py1) < 8:
        return None, 0.0
    patch   = img_bgr[py1:py2, px1:px2]
    pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    try:
        val, conf = _get_model_bundle()["decoder_engine"](pil_img)
        code = int(val)
        return (None, 0.0) if code == 256 else (code, conf)
    except Exception:
        return None, 0.0


def _detect_and_decode(image_path: Path) -> tuple:
    """
    Esegue YOLO sull'immagine pre-processata e decodifica ogni bbox
    sull'immagine originale.
    Ritorna (lista di detection dict completi, img_bgr originale).
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return [], None

    models = _get_model_bundle()
    img_pre = cv2.cvtColor(np.array(models["augmentor"].augment(img_bgr)), cv2.COLOR_RGB2BGR)
    try:
        results = models["yolo_model"](
            img_pre,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )
    except Exception:
        # Prima di propagare un possibile OOM si conserva il picco raggiunto,
        # altrimenti il benchmark perderebbe proprio il dato più significativo.
        _save_cuda_metrics(_load_json_metric_count())
        raise
    dets = []
    yolo_boxes = 0
    for result in results:
        for box in result.boxes:
            yolo_boxes += 1
            bbox = box.xyxyn[0].tolist()
            code, conf = _decode_patch(img_bgr, bbox)
            if code is None:
                continue
            cx, cy = _bbox_center(bbox)
            info = COMPONENT_INFO.get(code, (2, f"code_{code}"))
            dets.append({
                "bbox_norm": bbox,
                "code": code,
                "confidence": conf,
                "cx": cx,
                "cy": cy,
                "name": info[1],
                "max_degree": info[0],
                "is_diameter": code in DIAMETER_CODES,
                "accepted": conf >= CONF_THRESHOLD,
            })
    print(
        f"    [DetectRun] {image_path.name} "
        f"iou={YOLO_IOU:.2f} yolo_boxes={yolo_boxes} "
        f"decoded={len(dets)} accepted={sum(1 for d in dets if d['accepted'])}"
    )
    previous = _load_json_metric_count()
    _save_cuda_metrics(previous + 1)
    return dets, img_bgr


def _load_json_metric_count() -> int:
    if not _resource_metrics_enabled():
        return 0
    try:
        payload = json.loads((WORK_DIR / "benchmark_yolo_cuda.json").read_text(encoding="utf-8"))
        return int(payload.get("images_completed", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0

