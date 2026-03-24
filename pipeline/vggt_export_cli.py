"""CLI run by the embedded Python runtime to export VGGT artefacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _save_predictions(output_dir: Path, image_paths: list[Path], predictions: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    world_points = predictions["world_points"].detach().cpu().numpy()[0]
    world_points_conf = predictions["world_points_conf"].detach().cpu().numpy()[0]
    depth = predictions.get("depth")
    depth_conf = predictions.get("depth_conf")
    depth_np = depth.detach().cpu().numpy()[0] if depth is not None else None
    depth_conf_np = depth_conf.detach().cpu().numpy()[0] if depth_conf is not None else None
    pose_enc = predictions.get("pose_enc")
    pose_np = pose_enc.detach().cpu().numpy()[0] if pose_enc is not None else None

    saved = []
    for idx, image_path in enumerate(image_paths):
        image_id = image_path.stem
        out_path = output_dir / f"{image_id}.npz"
        payload = {
            "world_points": world_points[idx].astype(np.float32),
            "world_points_conf": world_points_conf[idx].astype(np.float32),
        }
        if depth_np is not None:
            payload["depth"] = depth_np[idx].astype(np.float32)
        if depth_conf_np is not None:
            payload["depth_conf"] = depth_conf_np[idx].astype(np.float32)
        if pose_np is not None:
            payload["pose_enc"] = pose_np[idx].astype(np.float32)
        np.savez_compressed(out_path, **payload)
        saved.append(str(out_path))

    return {"saved_files": saved}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_dir = Path(request["output_dir"])
    image_paths = [Path(p) for p in request["image_paths"]]
    model_id = request.get("model_id", "facebook/VGGT-1B")
    preprocess_mode = request.get("preprocess_mode", "crop")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()

    images = load_and_preprocess_images([str(p) for p in image_paths], mode=preprocess_mode).to(device)
    model = VGGT.from_pretrained(model_id).to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(images)

    save_info = _save_predictions(output_dir, image_paths, predictions)
    summary = {
        "status": "ok",
        "device": device,
        "model_id": model_id,
        "preprocess_mode": preprocess_mode,
        "n_images": len(image_paths),
        "elapsed_sec": round(time.time() - start, 3),
        **save_info,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
