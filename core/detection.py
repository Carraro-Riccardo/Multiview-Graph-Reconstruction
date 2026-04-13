"""
Funzioni di discovery, pre-processing, detection YOLO e decodifica Shurikode.
"""

import re
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

from config import CONF_THRESHOLD, YOLO_CONF, YOLO_IOU, YOLO_IMGSZ, COMPONENT_INFO, DIAMETER_CODES, WORK_DIR

_MODEL_BUNDLE = None


def _get_model_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        from .models import augmentor, decoder_engine, yolo_model  # noqa: PLC0415

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
        return [int(p) for p in node_id.split("_")]

    return dict(sorted(node_map.items(), key=lambda kv: _sort_key(kv[0])))



def _cache_path(image_path: Path) -> Path:
    return CACHE_DIR / f"{image_path.stem}.json"


CACHE_VERSION = 2


def _cache_signature() -> dict:
    return {
        "version": CACHE_VERSION,
        "yolo_conf": float(YOLO_CONF),
        "yolo_iou": float(YOLO_IOU),
        "yolo_imgsz": int(YOLO_IMGSZ),
        "conf_threshold": float(CONF_THRESHOLD),
    }



def _load_cached_detections(image_path: Path) -> list | None:
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
    """
    Ritorna detection+decoding da cache locale se disponibile.
    Se manca la cache, esegue il passo completo una sola volta e la salva.
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

    dets, _ = detect_and_decode(image_path)
    _save_cached_detections(image_path, dets)
    return dets, img_bgr

CACHE_DIR = WORK_DIR / "detection_cache"

def bbox_center(bbox_norm: list) -> tuple:
    x1, y1, x2, y2 = bbox_norm
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def decode_patch(img_bgr: np.ndarray, bbox_norm: list) -> tuple:
    """
    Estrae la patch dall'immagine BGR e la decodifica.
    Ritorna (code, conf) oppure (None, 0.0) se la patch Ã¨ troppo piccola
    o il codice non Ã¨ valido.
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


def detect_and_decode(image_path: Path) -> tuple:
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
    results = models["yolo_model"](
        img_pre,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )
    dets = []
    yolo_boxes = 0
    for result in results:
        for box in result.boxes:
            yolo_boxes += 1
            bbox = box.xyxyn[0].tolist()
            code, conf = decode_patch(img_bgr, bbox)
            if code is None:
                continue
            cx, cy = bbox_center(bbox)
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
    return dets, img_bgr


def detect_and_decode_with_conf(image_path: Path) -> tuple:
    """
    Wrapper che restituisce solo i campi necessari al motore del grafo.
    Legge dal cache locale quando disponibile.
    """
    dets_raw, img_bgr = detect_and_decode_cached(image_path)
    dets = [{
        "bbox_norm": d["bbox_norm"],
        "code":      d["code"],
        "conf":      d["confidence"],
        "cx":        d["cx"],
        "cy":        d["cy"],
    } for d in dets_raw]
    return dets, img_bgr

def detect_and_decode_simple(image_path: Path) -> tuple:
    """
    Versione semplificata per Step 1: filtra subito per CONF_THRESHOLD
    e ritorna solo i campi bbox_norm, cx, cy, code.
    Usa lo stesso cache del passo completo.
    """
    dets_raw, img_bgr = detect_and_decode_cached(image_path)
    dets = [{
        "bbox_norm": d["bbox_norm"],
        "cx":        d["cx"],
        "cy":        d["cy"],
        "code":      d["code"],
    } for d in dets_raw if d["confidence"] >= CONF_THRESHOLD]
    return dets, img_bgr

def score_view(dets: list) -> float:
    """Punteggio di qualitÃ  di una vista basato su nodi topologici."""
    topo = [d for d in dets if d["code"] not in DIAMETER_CODES]
    if not topo:
        return 0.0
    return len(topo) + np.mean([d["conf"] for d in topo])


