"""Generate the real-data figures used in Chapter 6 of the technical manual."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


MANUAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MANUAL_DIR.parents[1]
FIGURES_DIR = MANUAL_DIR / "figures"


def _read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def _resolved_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def generate_detection_overlay(images_dir: Path, work_dir: Path) -> None:
    cache_path = work_dir / "detection_cache" / "node11_0006.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    detections = [det for det in payload["detections"] if det.get("accepted", True)]

    image = Image.open(images_dir / payload["image"]).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label_font = _font(20, bold=True)

    pixel_boxes: list[tuple[int, int, int, int]] = []
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox_norm"]
        box = (
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        )
        pixel_boxes.append(box)
        color = (224, 119, 33, 255) if det["is_diameter"] else (0, 137, 168, 255)
        draw.rectangle(box, outline=color, width=6)

        label = f"{index:02d} | {det['code']} | {det['confidence']:.2f}"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        label_w = text_box[2] - text_box[0]
        label_h = text_box[3] - text_box[1]
        label_x = max(0, min(box[0], width - label_w - 12))
        label_y = box[1] - label_h - 12
        if label_y < 0:
            label_y = box[3] + 6
        draw.rounded_rectangle(
            (label_x, label_y, label_x + label_w + 12, label_y + label_h + 8),
            radius=4,
            fill=(15, 23, 28, 220),
        )
        draw.text((label_x + 6, label_y + 2), label, fill=color, font=label_font)

    annotated = Image.alpha_composite(image, overlay).convert("RGB")
    crop_margin_left = 160
    crop_margin_right = 260
    crop_margin_y = 150
    crop_box = (
        max(0, min(box[0] for box in pixel_boxes) - crop_margin_left),
        max(0, min(box[1] for box in pixel_boxes) - crop_margin_y),
        min(width, max(box[2] for box in pixel_boxes) + crop_margin_right),
        min(height, max(box[3] for box in pixel_boxes) + crop_margin_y),
    )
    cropped = annotated.crop(crop_box)

    header_height = 62
    canvas = Image.new("RGB", (cropped.width, cropped.height + header_height), "#0f171c")
    canvas.paste(cropped, (0, header_height))
    header = ImageDraw.Draw(canvas)
    title_font = _font(28, bold=True)
    legend_font = _font(22)
    header.text((18, 14), "Detection filtrate - node11_0006", fill="white", font=title_font)
    header.rectangle((cropped.width - 405, 17, cropped.width - 383, 39), fill="#0089a8")
    header.text((cropped.width - 374, 14), "componenti", fill="white", font=legend_font)
    header.rectangle((cropped.width - 215, 17, cropped.width - 193, 39), fill="#e07721")
    header.text((cropped.width - 184, 14), "diametri", fill="white", font=legend_font)

    canvas.save(FIGURES_DIR / "detection_overlay.png", optimize=True)


def _observed_detection(work_dir: Path, observed_id: str) -> tuple[str, dict]:
    image_id, index_text = observed_id.rsplit("_", 1)
    payload = json.loads(
        (work_dir / "detection_cache" / f"{image_id}.json").read_text(encoding="utf-8")
    )
    filtered = [
        det
        for det in payload["detections"]
        if det.get("confidence", det.get("conf", 0.0)) >= 0.5
        and not det.get("is_diameter", False)
    ]
    return image_id, filtered[int(index_text)]


def _detection_crop(image: Image.Image, bbox_norm: list[float]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox_norm
    left, top = x1 * width, y1 * height
    right, bottom = x2 * width, y2 * height
    padding = 0.22 * max(right - left, bottom - top)
    crop_box = (
        max(0, round(left - padding)),
        max(0, round(top - padding)),
        min(width, round(right + padding)),
        min(height, round(bottom + padding)),
    )
    return ImageOps.fit(
        image.crop(crop_box),
        (184, 136),
        method=Image.Resampling.LANCZOS,
    )


def generate_reid_clusters(images_dir: Path, work_dir: Path) -> None:
    scene_nodes = json.loads(
        (work_dir / "multiview" / "node7" / "scene_nodes.json").read_text(encoding="utf-8")
    )
    selected = [node for node in scene_nodes if node["scene_node_id"] in {"15#1", "15#2", "15#3"}]
    selected.sort(key=lambda node: node["scene_node_id"])

    canvas = Image.new("RGB", (1320, 640), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    label_font = _font(25, bold=True)
    detail_font = _font(20)
    view_font = _font(17)
    border_color = "#007f7b"
    text_color = "#17324d"

    draw.text((28, 18), "Raggruppamenti ReID - nodo 7", fill=text_color, font=title_font)
    draw.text(
        (28, 55),
        "Tre istanze dello stesso codice riconosciute in viste differenti",
        fill="#4c5d6b",
        font=detail_font,
    )

    row_top = 92
    row_height = 176
    label_width = 230
    thumb_gap = 16
    for row_index, node in enumerate(selected):
        top = row_top + row_index * row_height
        if row_index % 2 == 0:
            draw.rounded_rectangle((18, top, 1302, top + 160), radius=8, fill="#f4f7f8")

        draw.text((34, top + 34), node["scene_node_id"], fill=border_color, font=label_font)
        draw.text(
            (34, top + 70),
            f"{node['fused_name']} [{node['fused_code']}]\n{node['support_count']} viste",
            fill=text_color,
            font=detail_font,
            spacing=4,
        )

        observed_ids = sorted(node["supporting_observed_ids"])
        for column, observed_id in enumerate(observed_ids):
            image_id, detection = _observed_detection(work_dir, observed_id)
            image = Image.open(images_dir / f"{image_id}.jpg").convert("RGB")
            thumbnail = _detection_crop(image, detection["bbox_norm"])
            left = label_width + column * (thumbnail.width + thumb_gap)
            canvas.paste(thumbnail, (left, top + 10))
            draw.rectangle(
                (left, top + 10, left + thumbnail.width, top + 10 + thumbnail.height),
                outline=border_color,
                width=3,
            )
            draw.text(
                (left + 55, top + 147),
                image_id.rsplit("_", 1)[-1],
                fill="#4c5d6b",
                font=view_font,
            )

    canvas.save(FIGURES_DIR / "reid_clusters.png", optimize=True)


def generate_vggt_geometry(work_dir: Path) -> None:
    source = Image.open(
        work_dir / "debug" / "node7" / "pointcloud_bboxes" / "node7_fused_centroids.png"
    ).convert("RGB")
    source.thumbnail((1800, 1100), Image.Resampling.LANCZOS)
    source.save(FIGURES_DIR / "vggt_geometry.png", optimize=True)


def main() -> None:
    env = _read_env()
    images_dir = _resolved_path(env["IMAGES_DIR"])
    work_dir = _resolved_path(env.get("WORK_DIR", "./output"))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_detection_overlay(images_dir, work_dir)
    generate_reid_clusters(images_dir, work_dir)
    generate_vggt_geometry(work_dir)


if __name__ == "__main__":
    main()
