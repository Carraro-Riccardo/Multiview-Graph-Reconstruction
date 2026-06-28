"""Visualizza, per ogni immagine di un nodo, dove si proiettano i cluster VGGT.

Lo script non modifica la pipeline: legge solo gli artefatti gia' prodotti
(`output/multiview/nodeX/scene_nodes.json`, `output/detection_cache/*.json`,
immagini originali e, se disponibili, le annotazioni GT).

Output per nodo:
  - DEBUG_OUTPUT_DIR/nodeX/cluster_projections/<image>_clusters.png
  - DEBUG_OUTPUT_DIR/nodeX/cluster_projections/cluster_projection_report.json

Uso:
    python visualize_vggt_cluster_projections.py 7
    python visualize_vggt_cluster_projections.py 7 9
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_reconstruction.config import COMPONENT_INFO, CONF_THRESHOLD, DEBUG_OUTPUT_DIR, DIAMETER_CODES, IMAGES_DIR, LABELS_EXT_DIR, WORK_DIR

MULTIVIEW_DIR = WORK_DIR / "multiview"
OUT_DIR = DEBUG_OUTPUT_DIR
ANNOTATIONS_DIR = Path("C:\\Users\\ricca\\Desktop\\annotations")

PALETTE = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#5dade2",
    "#af7ac5",
    "#58d68d",
    "#f5b041",
    "#ec7063",
    "#48c9b0",
    "#85c1e9",
    "#f7dc6f",
    "#bb8fce",
    "#52be80",
    "#eb984e",
    "#f1948a",
    "#7fb3d5",
    "#73c6b6",
    "#f8c471",
    "#82e0aa",
    "#a569bd",
]

STATUS_STYLE = {
    "ok": {"edge": "#2ecc71", "badge": "#145a32", "label": "OK"},
    "duplicate_gt": {"edge": "#e67e22", "badge": "#7e5109", "label": "DUP_GT"},
    "no_gt_match": {"edge": "#e74c3c", "badge": "#78281f", "label": "NO_GT"},
    "low_support": {"edge": "#f1c40f", "badge": "#7d6608", "label": "LOW_SUP"},
    "unmatched": {"edge": "#95a5a6", "badge": "#424949", "label": "UNMATCH"},
}


def _find_image(image_stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = Path(IMAGES_DIR) / f"{image_stem}{ext}"
        if path.exists():
            return path
    return None


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def load_scene_nodes(node_id: int) -> list[dict]:
    path = MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_detections_for_image(image_stem: str) -> list[dict]:
    path = WORK_DIR / "detection_cache" / f"{image_stem}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8")).get("detections", [])
    return [
        det
        for det in raw
        if det.get("confidence", 0.0) >= CONF_THRESHOLD
        and not det.get("is_diameter", det.get("code", -1) in DIAMETER_CODES)
    ]


def load_gt_for_node(node_id: int) -> dict:
    image_stems = sorted(path.stem for path in IMAGES_DIR.glob(f"node{node_id}_*.jpg"))
    instances_by_image: dict[str, list[dict]] = {}

    for image_stem in image_stems:
        label_path = LABELS_EXT_DIR / f"{image_stem}_extended.txt"
        items: list[dict] = []
        if label_path.exists():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().split()
                if len(parts) < 7:
                    continue
                code = int(parts[6])
                if code in DIAMETER_CODES:
                    continue
                items.append(
                    {
                        "instance_id": int(parts[5]),
                        "code": code,
                        "bbox_norm": _xywh_to_xyxy(
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        ),
                    }
                )
        instances_by_image[image_stem] = items

    return {"instances_by_image": instances_by_image}


def match_detections_to_gt(det_bbox: list[float], det_code: int, gt_instances: list[dict]) -> tuple[int | None, float]:
    best_id = None
    best_iou = 0.0
    for gt in gt_instances:
        if gt["code"] != det_code:
            continue
        score = _iou(det_bbox, gt["bbox_norm"])
        if score > best_iou:
            best_iou = score
            best_id = gt["instance_id"]
    if best_iou < 0.30:
        return None, 0.0
    return best_id, best_iou


def _format_gt_label(cluster_info: dict) -> str:
    gt = cluster_info.get("majority_gt_id")
    purity = cluster_info.get("purity")
    if gt is None:
        return "GT: none"
    if purity is None:
        return f"GT:{gt}"
    return f"GT:{gt} ({purity:.2f})"


def _build_cluster_analysis(node_id: int) -> tuple[list[dict], dict[str, dict], dict[str, list[dict]]]:
    scene_nodes = load_scene_nodes(node_id)
    if not scene_nodes:
        raise FileNotFoundError(f"scene_nodes.json non trovato per node {node_id}")

    gt_data = load_gt_for_node(node_id)
    gt_by_image = gt_data["instances_by_image"]

    observations_by_image: dict[str, list[dict]] = defaultdict(list)
    cluster_infos: dict[str, dict] = {}

    for scene_node in scene_nodes:
        cluster_id = scene_node["scene_node_id"]
        votes: Counter = Counter()
        details: list[dict] = []

        for obs_id in scene_node.get("supporting_observed_ids", []):
            parts = obs_id.rsplit("_", 1)
            if len(parts) != 2:
                continue
            image_stem, det_idx_txt = parts
            det_idx = int(det_idx_txt)
            dets = load_detections_for_image(image_stem)
            if det_idx >= len(dets):
                details.append(
                    {
                        "obs_id": obs_id,
                        "image_stem": image_stem,
                        "det_idx": det_idx,
                        "missing_detection": True,
                    }
                )
                continue

            det = dets[det_idx]
            gt_id, iou = match_detections_to_gt(
                det["bbox_norm"],
                det["code"],
                gt_by_image.get(image_stem, []),
            )
            if gt_id is not None:
                votes[gt_id] += 1

            obs = {
                "obs_id": obs_id,
                "image_stem": image_stem,
                "det_idx": det_idx,
                "bbox_norm": det["bbox_norm"],
                "code": det["code"],
                "confidence": float(det.get("confidence", det.get("conf", 0.0))),
                "name": det.get("name") or COMPONENT_INFO.get(det["code"], (0, f"code_{det['code']}"))[1],
                "gt_id": gt_id,
                "iou_to_gt": float(iou),
            }
            details.append(obs)
            observations_by_image[image_stem].append(
                {
                    **obs,
                    "cluster_id": cluster_id,
                }
            )

        majority_gt_id = votes.most_common(1)[0][0] if votes else None
        majority_count = votes[majority_gt_id] if majority_gt_id is not None else 0
        matched_obs = sum(votes.values())
        purity = (majority_count / matched_obs) if matched_obs else None

        cluster_infos[cluster_id] = {
            "cluster_id": cluster_id,
            "code": scene_node["fused_code"],
            "name": scene_node.get("fused_name") or COMPONENT_INFO.get(scene_node["fused_code"], (0, f"code_{scene_node['fused_code']}"))[1],
            "fused_confidence": float(scene_node.get("fused_confidence", 0.0)),
            "support_count": int(scene_node.get("support_count", 0)),
            "support_images": list(scene_node.get("support_images", [])),
            "votes": dict(votes),
            "majority_gt_id": majority_gt_id,
            "majority_count": int(majority_count),
            "matched_obs": int(matched_obs),
            "purity": purity,
            "details": details,
            "status": "ok",
            "status_reason": "",
        }

    by_code_gt: dict[tuple[int, int], list[str]] = defaultdict(list)
    for cluster_id, info in cluster_infos.items():
        gt_id = info["majority_gt_id"]
        if gt_id is not None:
            by_code_gt[(info["code"], gt_id)].append(cluster_id)

    for (_code, _gt_id), cluster_ids in by_code_gt.items():
        if len(cluster_ids) <= 1:
            continue
        ranked = sorted(
            cluster_ids,
            key=lambda cid: (
                cluster_infos[cid]["majority_count"],
                cluster_infos[cid]["support_count"],
                cluster_infos[cid]["fused_confidence"],
            ),
            reverse=True,
        )
        keeper = ranked[0]
        for cid in ranked[1:]:
            cluster_infos[cid]["status"] = "duplicate_gt"
            cluster_infos[cid]["status_reason"] = (
                f"duplica GT {cluster_infos[cid]['majority_gt_id']} "
                f"gia' coperto meglio da {keeper}"
            )

    for cluster_id, info in cluster_infos.items():
        if info["status"] != "ok":
            continue
        if info["majority_gt_id"] is None:
            info["status"] = "no_gt_match"
            info["status_reason"] = "nessuna observation matcha il GT"
        elif info["support_count"] <= 2:
            info["status"] = "low_support"
            info["status_reason"] = "cluster con supporto molto basso"

    ordered_scene_nodes = sorted(
        scene_nodes,
        key=lambda sn: (sn["fused_code"], sn["scene_node_id"]),
    )
    return ordered_scene_nodes, cluster_infos, observations_by_image


def _draw_badge(ax, x: float, y: float, text: str, edge: str, face: str, fontsize: int = 8) -> None:
    raise NotImplementedError


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _blend(base_rgb: tuple[int, int, int], overlay_rgb: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return tuple(
        int(round((1.0 - alpha) * b + alpha * o))
        for b, o in zip(base_rgb, overlay_rgb)
    )


def _text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, outline: str) -> None:
    left, top = xy
    bbox = draw.multiline_textbbox((left, top), text, font=font, spacing=2)
    pad_x = 4
    pad_y = 3
    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rounded_rectangle(rect, radius=5, fill=fill, outline=outline, width=2)
    draw.multiline_text((left, top), text, font=font, fill="white", spacing=2)


def _render_image(
    node_id: int,
    image_stem: str,
    cluster_infos: dict[str, dict],
    observations: list[dict],
    out_path: Path,
) -> dict:
    image_path = _find_image(image_stem)
    if image_path is None:
        raise FileNotFoundError(f"Immagine non trovata per {image_stem}")

    base = Image.open(image_path).convert("RGB")
    width, height = base.size
    filtered_dets = load_detections_for_image(image_stem)

    cluster_ids = sorted(cluster_infos.keys())
    cluster_colors = {
        cluster_id: PALETTE[i % len(PALETTE)]
        for i, cluster_id in enumerate(cluster_ids)
    }

    canvas = Image.new("RGB", (width + 760, max(height, 900)), "#101418")
    canvas.paste(base, (0, 0))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(canvas)

    font_small = _load_font(14)
    font_main = _load_font(16)
    font_title = _load_font(22)

    obs_by_det = {obs["det_idx"]: obs for obs in observations}

    for det_idx, det in enumerate(filtered_dets):
        x1, y1, x2, y2 = det["bbox_norm"]
        x1 = int(round(x1 * width))
        x2 = int(round(x2 * width))
        y1 = int(round(y1 * height))
        y2 = int(round(y2 * height))

        draw.rectangle((x1, y1, x2, y2), outline="#7f8c8d", width=2)
        _text_box(
            draw,
            (x1 + 2, max(2, y1 - 20)),
            f"d{det_idx}:{det['code']}",
            font=font_small,
            fill="#17202a",
            outline="#566573",
        )

    for det_idx, obs in sorted(obs_by_det.items()):
        cluster_id = obs["cluster_id"]
        info = cluster_infos[cluster_id]
        style = STATUS_STYLE[info["status"]]
        color = cluster_colors[cluster_id]
        color_rgb = _hex_to_rgb(color)

        x1, y1, x2, y2 = obs["bbox_norm"]
        x1 = int(round(x1 * width))
        x2 = int(round(x2 * width))
        y1 = int(round(y1 * height))
        y2 = int(round(y2 * height))

        draw_overlay.rectangle((x1, y1, x2, y2), fill=color_rgb + (55,))
        draw.rectangle((x1, y1, x2, y2), outline=style["edge"], width=5)
        draw.rectangle((x1 + 2, y1 + 2, x2 - 2, y2 - 2), outline=color, width=3)

        label = (
            f"{cluster_id} | d{det_idx} | {info['name']} | "
            f"{STATUS_STYLE[info['status']]['label']}"
        )
        _text_box(
            draw,
            (x1 + 3, max(2, y1 - 42)),
            label,
            font=font_main,
            fill=style["badge"],
            outline=style["edge"],
        )

        if obs.get("gt_id") is not None:
            gt_text = f"GT:{obs['gt_id']} IoU:{obs['iou_to_gt']:.2f}"
            _text_box(
                draw,
                (x1 + 3, min(height - 24, y2 + 3)),
                gt_text,
                font=font_small,
                fill="#0e6251",
                outline="#1abc9c",
            )

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"node {node_id} | {image_stem} | det filtered: {len(filtered_dets)} | "
        f"cluster proiettati: {len(observations)}"
    )
    draw.rectangle((0, 0, width + 760, 38), fill="#0b0f12")
    draw.text((12, 8), title, font=font_title, fill="white")

    panel_left = width + 18
    panel_top = 58
    draw.rounded_rectangle(
        (width + 10, 48, width + 740, canvas.size[1] - 12),
        radius=10,
        fill="#161c22",
        outline="#273746",
        width=2,
    )

    lines: list[tuple[str, str]] = []
    lines.append((f"NODE {node_id}  |  {image_stem}", "white"))
    lines.append(("", "white"))

    if not observations:
        lines.append(("Nessun cluster presente in questa immagine.", "#d5dbdb"))
    else:
        for obs in sorted(observations, key=lambda item: (item["code"], item["cluster_id"])):
            info = cluster_infos[obs["cluster_id"]]
            style = STATUS_STYLE[info["status"]]
            reason = info["status_reason"] or "cluster coerente"
            line = (
                f"{obs['cluster_id']}  d{obs['det_idx']}  {info['name']}\n"
                f"status={style['label']}  supp={info['support_count']}  "
                f"conf={info['fused_confidence']:.2f}  {_format_gt_label(info)}\n"
                f"causa: {reason}"
            )
            lines.append((line, cluster_colors[obs["cluster_id"]]))
            lines.append(("", "white"))

    y = panel_top
    for text, color in lines:
        if text:
            draw.multiline_text(
                (panel_left, y),
                text,
                font=font_main,
                fill=color,
                spacing=4,
            )
            line_count = text.count("\n") + 1
        else:
            line_count = 1
        y += 30 * line_count
        if y > canvas.size[1] - 40:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

    return {
        "image_stem": image_stem,
        "image_path": str(image_path),
        "output_path": str(out_path),
        "n_filtered_detections": len(filtered_dets),
        "projected_clusters": [
            {
                "cluster_id": obs["cluster_id"],
                "det_idx": obs["det_idx"],
                "code": obs["code"],
                "gt_id": obs["gt_id"],
                "iou_to_gt": obs["iou_to_gt"],
                "status": cluster_infos[obs["cluster_id"]]["status"],
                "status_reason": cluster_infos[obs["cluster_id"]]["status_reason"],
            }
            for obs in sorted(observations, key=lambda item: item["det_idx"])
        ],
    }


def _render_node(node_id: int) -> Path:
    scene_nodes, cluster_infos, observations_by_image = _build_cluster_analysis(node_id)
    node_out_dir = OUT_DIR / f"node{node_id}" / "cluster_projections"
    node_out_dir.mkdir(parents=True, exist_ok=True)

    image_stems = sorted(
        {stem for info in cluster_infos.values() for stem in info["support_images"]}
    )

    image_reports = []
    for image_stem in image_stems:
        out_path = node_out_dir / f"{image_stem}_clusters.png"
        report = _render_image(
            node_id=node_id,
            image_stem=image_stem,
            cluster_infos=cluster_infos,
            observations=observations_by_image.get(image_stem, []),
            out_path=out_path,
        )
        image_reports.append(report)

    summary = {
        "node_id": node_id,
        "scene_node_file": str(MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes.json"),
        "images": image_reports,
        "clusters": [
            {
                key: value
                for key, value in info.items()
                if key != "details"
            }
            for info in sorted(
                cluster_infos.values(),
                key=lambda item: (item["code"], item["cluster_id"]),
            )
        ],
        "counts": {
            "n_scene_nodes": len(scene_nodes),
            "n_duplicate_gt": sum(1 for info in cluster_infos.values() if info["status"] == "duplicate_gt"),
            "n_no_gt_match": sum(1 for info in cluster_infos.values() if info["status"] == "no_gt_match"),
            "n_low_support": sum(1 for info in cluster_infos.values() if info["status"] == "low_support"),
        },
    }

    report_path = node_out_dir / "cluster_projection_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    node_ids = [int(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else [7]

    for node_id in node_ids:
        print(f"[cluster-proj] node {node_id}")
        report_path = _render_node(node_id)
        print(f"  report: {report_path}")
        print(f"  images: {(report_path.parent).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
