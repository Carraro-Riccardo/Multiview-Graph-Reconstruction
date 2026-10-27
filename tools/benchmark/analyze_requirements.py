"""Estimate the resources required by the graph-reconstruction pipeline.

The analysis is intentionally dependency-free: it can be run before installing
the packages in ``requirements.txt``.  Values derived from files and hardware
are reported separately from conservative heuristic estimates.

Examples:
    python tools/benchmark/analyze_requirements.py
    python tools/benchmark/analyze_requirements.py 2 9 15_2
    python tools/benchmark/analyze_requirements.py --json
    python tools/benchmark/analyze_requirements.py --output output/requirements_report.json
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "graph_reconstruction" / "config.py"
DOTENV_PATH = PROJECT_ROOT / ".env"


def _natural_key(value: str) -> list[tuple[int, Any]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    ]


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def _read_literal_constants(path: Path) -> dict[str, Any]:
    """Read simple assignments without importing the application config."""
    try:
        # ``utf-8-sig`` also accepts ordinary UTF-8 and strips a possible BOM.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError):
        return {}
    constants: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def _setting(env: dict[str, str], name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, env.get(name, default))


def _bool_setting(env: dict[str, str], name: str, default: bool) -> bool:
    raw = _setting(env, name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _path_setting(env: dict[str, str], name: str, default: str | None = None) -> Path | None:
    raw = _setting(env, name, default)
    if raw is None:
        return None
    path = Path(os.path.expandvars(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _jpeg_size(path: Path) -> tuple[int, int] | None:
    """Read JPEG dimensions without Pillow or OpenCV."""
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    try:
        with path.open("rb") as stream:
            if stream.read(2) != b"\xff\xd8":
                return None
            while True:
                byte = stream.read(1)
                while byte and byte != b"\xff":
                    byte = stream.read(1)
                if not byte:
                    return None
                while byte == b"\xff":
                    byte = stream.read(1)
                if not byte:
                    return None
                marker = byte[0]
                if marker in {0xD8, 0xD9}:
                    continue
                length_raw = stream.read(2)
                if len(length_raw) != 2:
                    return None
                length = int.from_bytes(length_raw, "big")
                if length < 2:
                    return None
                if marker in sof_markers:
                    payload = stream.read(5)
                    if len(payload) != 5:
                        return None
                    height = int.from_bytes(payload[1:3], "big")
                    width = int.from_bytes(payload[3:5], "big")
                    return width, height
                stream.seek(length - 2, 1)
    except OSError:
        return None


def _discover_nodes(images_dir: Path | None) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    nodes: dict[str, list[Path]] = {}
    image_info: list[dict[str, Any]] = []
    if images_dir is None or not images_dir.is_dir():
        return nodes, image_info
    pattern = re.compile(r"^node(.+)_(\d+)\.jpg$", re.IGNORECASE)
    for path in sorted(images_dir.glob("node*_*.jpg")):
        match = pattern.match(path.name)
        if not match:
            continue
        node_id = match.group(1)
        nodes.setdefault(node_id, []).append(path)
        dimensions = _jpeg_size(path)
        image_info.append(
            {
                "path": str(path),
                "node_id": node_id,
                "bytes": path.stat().st_size,
                "width": dimensions[0] if dimensions else None,
                "height": dimensions[1] if dimensions else None,
            }
        )
    return dict(sorted(nodes.items(), key=lambda item: _natural_key(item[0]))), image_info


def _directory_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    total = 0
    try:
        for root, _, files in os.walk(path):
            for filename in files:
                try:
                    total += (Path(root) / filename).stat().st_size
                except OSError:
                    pass
    except OSError:
        return None
    return total


def _windows_memory() -> tuple[int | None, int | None]:
    if os.name != "nt":
        return None, None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None, None
    if not ok:
        return None, None
    return int(status.total_physical), int(status.available_physical)


def _system_memory() -> tuple[int | None, int | None]:
    total, available = _windows_memory()
    if total is not None:
        return total, available
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES")
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
        return int(total), int(available)
    except (AttributeError, OSError, ValueError):
        return None, None


def _torch_probe_current() -> dict[str, Any]:
    result: dict[str, Any] = {"installed": False, "cuda_available": False}
    try:
        import torch  # type: ignore

        result.update(
            {
                "installed": True,
                "version": str(torch.__version__),
                "cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
    except Exception as exc:  # A broken CUDA install is useful diagnostic information.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _python_runtime_probe(executable: Path | None) -> dict[str, Any]:
    if executable is None or not executable.is_file():
        return {"available": False, "path": str(executable) if executable else None}
    code = (
        "import json,platform; out={'available':True,'python':platform.python_version()}; "
        "\ntry:\n import torch; out.update({'torch':str(torch.__version__),"
        "'cuda_build':torch.version.cuda,'cuda_available':bool(torch.cuda.is_available()),"
        "'device_count':int(torch.cuda.device_count()) if torch.cuda.is_available() else 0})"
        "\nexcept Exception as e: out['torch_error']=type(e).__name__+': '+str(e)"
        "\nprint(json.dumps(out))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {"available": True}
        payload["path"] = str(executable)
        payload["returncode"] = completed.returncode
        if completed.returncode and completed.stderr:
            payload["error"] = completed.stderr.strip()[-1000:]
        return payload
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"available": True, "path": str(executable), "error": f"{type(exc).__name__}: {exc}"}


def _gpu_probe() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                    "driver_version": fields[4],
                }
            )
        except ValueError:
            continue
    return gpus


def _package_probe() -> dict[str, bool]:
    modules = {
        "numpy": "numpy",
        "scipy": "scipy",
        "opencv-python": "cv2",
        "Pillow": "PIL",
        "networkx": "networkx",
        "matplotlib": "matplotlib",
        "ultralytics": "ultralytics",
        "shurikodecontext (private)": "shurikodecontext",
    }
    return {name: importlib.util.find_spec(module) is not None for name, module in modules.items()}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _cache_analysis(
    nodes: dict[str, list[Path]],
    work_dir: Path,
    yolo_imgsz: int,
    yolo_conf: float,
    yolo_iou: float,
    conf_threshold: float,
    vggt_enabled: bool,
    vggt_skip_nodes: set[str],
) -> dict[str, Any]:
    detection_total = 0
    detection_valid = 0
    vggt_expected = 0
    vggt_present = 0
    node_rows = []
    elapsed_samples = []
    expected_signature = {
        "version": 2,
        "yolo_conf": float(yolo_conf),
        "yolo_iou": float(yolo_iou),
        "yolo_imgsz": int(yolo_imgsz),
        "conf_threshold": float(conf_threshold),
    }
    for node_id, images in nodes.items():
        valid_detection = 0
        for image in images:
            detection_total += 1
            payload = _load_json(work_dir / "detection_cache" / f"{image.stem}.json")
            if payload and payload.get("signature") == expected_signature:
                detection_valid += 1
                valid_detection += 1

        active = vggt_enabled and node_id not in vggt_skip_nodes
        present = 0
        if active:
            vggt_expected += len(images)
            vggt_dir = work_dir / "multiview" / f"node{node_id}" / "vggt"
            present = sum((vggt_dir / f"{image.stem}.npz").is_file() for image in images)
            vggt_present += present
            summary = _load_json(vggt_dir / "summary.json")
            if summary and isinstance(summary.get("elapsed_sec"), (int, float)):
                elapsed_samples.append(
                    {
                        "node_id": node_id,
                        "views": int(summary.get("n_images", len(images))),
                        "seconds": float(summary["elapsed_sec"]),
                        "device": summary.get("device"),
                    }
                )
        node_rows.append(
            {
                "node_id": node_id,
                "views": len(images),
                "detection_cache_valid": valid_detection,
                "vggt_active": active,
                "vggt_cache_present": present,
            }
        )
    return {
        "detection": {
            "images": detection_total,
            "valid": detection_valid,
            "missing_or_stale": detection_total - detection_valid,
        },
        "vggt": {
            "expected": vggt_expected,
            "present": vggt_present,
            "missing": vggt_expected - vggt_present,
        },
        "nodes": node_rows,
        "historical_vggt_runs": elapsed_samples,
    }


def _ceil_standard_capacity(value: float, capacities: tuple[int, ...]) -> int:
    for capacity in capacities:
        if capacity >= value:
            return capacity
    return int(math.ceil(value / 8.0) * 8)


def _estimates(
    *,
    max_pixels: int,
    max_views: int,
    max_active_views: int,
    next_vggt_views: int,
    yolo_model_bytes: int,
    yolo_imgsz: int,
    vggt_model_id: str,
    vggt_is_cuda: bool,
    vggt_will_generate: bool,
    input_bytes: int,
    existing_output_bytes: int,
    embedded_runtime_bytes: int,
) -> dict[str, Any]:
    # YOLO is batch=1.  Model file size is only a proxy because activations are
    # architecture-dependent; the range includes CUDA context and 1280px input.
    scale_sq = (max(yolo_imgsz, 32) / 640.0) ** 2
    checkpoint_gib = yolo_model_bytes / GIB
    yolo_low = 0.55 + 0.18 * scale_sq + 4.0 * checkpoint_gib
    yolo_high = 0.90 + 0.36 * scale_sq + 10.0 * checkpoint_gib
    yolo_low = round(max(1.0, yolo_low), 1)
    yolo_high = round(max(yolo_low + 0.5, yolo_high), 1)
    yolo_target = _ceil_standard_capacity(yolo_high * 1.25, (2, 4, 6, 8, 12, 16, 24, 32, 48))

    image_working_gib = max_pixels * 3 * 7 / GIB if max_pixels else 0.25
    cached_ram_low = round(max(3.0, 2.5 + image_working_gib), 1)
    cached_ram_high = round(max(6.0, 4.5 + image_working_gib * 1.8), 1)

    model_is_1b = "1b" in vggt_model_id.lower()
    model_factor = 1.0 if model_is_1b else 0.75
    # Current exporter runs the model in float32 and retains 24 intermediate
    # token tensors.  The upper bound grows with the number of views processed
    # together and deliberately leaves room for the dense prediction heads.
    cpu_vggt_low = round((8.0 + 1.25 * max_active_views) * model_factor, 1)
    cpu_vggt_high = round((13.0 + 2.1 * max_active_views) * model_factor, 1)
    cpu_ram_target = _ceil_standard_capacity(cpu_vggt_high * 1.10, (8, 16, 24, 32, 48, 64, 96, 128))

    vggt_cuda_low = round((7.0 + 0.9 * max_active_views) * model_factor, 1)
    vggt_cuda_high = round((10.0 + 1.65 * max_active_views) * model_factor, 1)
    combined_cuda_high = round(vggt_cuda_high + yolo_high, 1)
    vggt_cuda_target = _ceil_standard_capacity(combined_cuda_high * 1.15, (8, 12, 16, 24, 32, 48, 64, 80))

    # A fresh install includes the public PyTorch/OpenCV stack, the embedded
    # VGGT runtime, the 1B checkpoint, inputs and output/cache growth.
    runtime_gib = embedded_runtime_bytes / GIB if embedded_runtime_bytes else 5.0
    data_gib = (input_bytes + existing_output_bytes) / GIB
    fresh_disk = round(max(12.0, 4.0 + runtime_gib + 4.5 * model_factor + data_gib * 1.5), 1)
    free_disk_target = _ceil_standard_capacity(fresh_disk * 1.20, (16, 24, 32, 48, 64, 96, 128))

    next_cpu_peak = cached_ram_high
    if next_vggt_views and vggt_will_generate and not vggt_is_cuda:
        next_cpu_peak = round(max(next_cpu_peak, (13.0 + 2.1 * next_vggt_views) * model_factor), 1)

    if not vggt_will_generate:
        current_ram_range = [cached_ram_low, cached_ram_high]
        current_ram_target = _ceil_standard_capacity(cached_ram_high * 1.2, (8, 16, 24, 32))
        current_vggt_execution = "disabled/cache-only"
        current_gpu_target = yolo_target
    elif vggt_is_cuda:
        current_ram_range = [cached_ram_low, cached_ram_high]
        current_ram_target = 16
        current_vggt_execution = "CUDA"
        current_gpu_target = vggt_cuda_target
    else:
        current_ram_range = [cpu_vggt_low, cpu_vggt_high]
        current_ram_target = cpu_ram_target
        current_vggt_execution = "CPU"
        current_gpu_target = yolo_target

    return {
        "method": "conservative_heuristic_not_a_benchmark",
        "pipeline_concurrency": "nodes sequential; YOLO batch=1; VGGT batches all views of one node",
        "cached_pipeline": {
            "ram_range_gib": [cached_ram_low, cached_ram_high],
            "recommended_ram_gib": _ceil_standard_capacity(cached_ram_high * 1.2, (8, 16, 24, 32)),
            "vram_gib": 0.0,
            "notes": "With valid detection and VGGT caches, neither neural model is loaded.",
        },
        "cold_pipeline_current_vggt_runtime": {
            "ram_range_gib": current_ram_range,
            "recommended_ram_gib": current_ram_target,
            "yolo_vram_range_gib": [yolo_low, yolo_high],
            "recommended_gpu_vram_gib": current_gpu_target,
            "vggt_execution": current_vggt_execution,
        },
        "cold_pipeline_if_vggt_cuda_float32": {
            "vggt_vram_range_gib": [vggt_cuda_low, vggt_cuda_high],
            "combined_yolo_plus_vggt_vram_upper_gib": combined_cuda_high,
            "recommended_gpu_vram_gib": vggt_cuda_target,
            "notes": "YOLO stays resident in the parent process while the VGGT subprocess runs.",
        },
        "next_run_estimated_peak": {
            "ram_upper_gib": next_cpu_peak,
            "vggt_views_requiring_generation_in_largest_node": next_vggt_views,
        },
        "storage": {
            "fresh_install_and_working_space_gib": fresh_disk,
            "recommended_free_space_gib": free_disk_target,
        },
        "assumptions": {
            "max_views_any_selected_node": max_views,
            "max_views_vggt_active_node": max_active_views,
            "max_original_image_pixels": max_pixels,
            "yolo_imgsz": yolo_imgsz,
            "yolo_batch": 1,
            "vggt_input_long_side_px": 518,
            "vggt_precision_in_current_exporter": "float32",
            "vggt_model_id": vggt_model_id,
        },
    }


def analyze(selected_node_ids: list[str]) -> dict[str, Any]:
    env = _read_dotenv(DOTENV_PATH)
    constants = _read_literal_constants(CONFIG_PATH)
    images_dir = _path_setting(env, "IMAGES_DIR")
    work_dir = _path_setting(env, "WORK_DIR", "output") or PROJECT_ROOT / "output"
    yolo_model = _path_setting(env, "YOLO_MODEL_PATH")
    embedded_python = _path_setting(env, "VGGT_EMBEDDED_PYTHON", sys.executable)
    vggt_enabled = _bool_setting(env, "VGGT_EDGE_ENABLED", True)
    vggt_generate = _bool_setting(env, "VGGT_GENERATE_IF_MISSING", True)
    vggt_model_id = _setting(env, "VGGT_MODEL_ID", "facebook/VGGT-1B") or "facebook/VGGT-1B"

    yolo_imgsz = int(constants.get("YOLO_IMGSZ", 1280))
    yolo_conf = float(constants.get("YOLO_CONF", 0.05))
    yolo_iou = float(constants.get("YOLO_IOU", 0.3))
    conf_threshold = float(constants.get("CONF_THRESHOLD", 0.5))
    raw_skip_nodes = constants.get("VGGT_SKIP_NODES", set())
    vggt_skip_nodes = {str(value) for value in raw_skip_nodes}

    all_nodes, all_images = _discover_nodes(images_dir)
    unknown_nodes = [node_id for node_id in selected_node_ids if node_id not in all_nodes]
    nodes = (
        {node_id: all_nodes[node_id] for node_id in selected_node_ids if node_id in all_nodes}
        if selected_node_ids
        else all_nodes
    )
    selected_paths = {str(path) for images in nodes.values() for path in images}
    image_info = [item for item in all_images if item["path"] in selected_paths]

    cache = _cache_analysis(
        nodes,
        work_dir,
        yolo_imgsz,
        yolo_conf,
        yolo_iou,
        conf_threshold,
        vggt_enabled,
        vggt_skip_nodes,
    )
    max_views = max((len(images) for images in nodes.values()), default=0)
    active_rows = [row for row in cache["nodes"] if row["vggt_active"]]
    max_active_views = max((row["views"] for row in active_rows), default=0)
    missing_rows = [
        row for row in active_rows if row["vggt_cache_present"] < row["views"]
    ]
    next_vggt_views = max((row["views"] for row in missing_rows), default=0)
    max_pixels = max(
        (item["width"] * item["height"] for item in image_info if item["width"] and item["height"]),
        default=0,
    )

    total_ram, available_ram = _system_memory()
    try:
        disk = shutil.disk_usage(PROJECT_ROOT)
        disk_info = {"total_bytes": disk.total, "free_bytes": disk.free}
    except OSError:
        disk_info = {"total_bytes": None, "free_bytes": None}

    current_torch = _torch_probe_current()
    vggt_runtime = _python_runtime_probe(embedded_python)
    vggt_cuda = bool(vggt_runtime.get("cuda_available"))
    yolo_model_bytes = yolo_model.stat().st_size if yolo_model and yolo_model.is_file() else 0
    output_bytes = _directory_size(work_dir) or 0
    embedded_runtime_bytes = _directory_size(embedded_python.parent if embedded_python else None) or 0
    input_bytes = sum(item["bytes"] for item in image_info)

    estimates = _estimates(
        max_pixels=max_pixels,
        max_views=max_views,
        max_active_views=max_active_views,
        next_vggt_views=next_vggt_views if vggt_generate else 0,
        yolo_model_bytes=yolo_model_bytes,
        yolo_imgsz=yolo_imgsz,
        vggt_model_id=vggt_model_id,
        vggt_is_cuda=vggt_cuda,
        vggt_will_generate=vggt_enabled and vggt_generate,
        input_bytes=input_bytes,
        existing_output_bytes=output_bytes,
        embedded_runtime_bytes=embedded_runtime_bytes,
    )

    historical = cache["historical_vggt_runs"]
    cpu_samples = [sample for sample in historical if sample.get("device") == "cpu"]
    historical_summary: dict[str, Any] = {"samples": len(historical)}
    if cpu_samples:
        historical_summary.update(
            {
                "cpu_samples": len(cpu_samples),
                "cpu_total_seconds": round(sum(sample["seconds"] for sample in cpu_samples), 1),
                "cpu_seconds_range": [
                    round(min(sample["seconds"] for sample in cpu_samples), 1),
                    round(max(sample["seconds"] for sample in cpu_samples), 1),
                ],
            }
        )

    return {
        "report_version": 1,
        "project_root": str(PROJECT_ROOT),
        "configuration": {
            "dotenv_found": DOTENV_PATH.is_file(),
            "images_dir": str(images_dir) if images_dir else None,
            "images_dir_found": bool(images_dir and images_dir.is_dir()),
            "work_dir": str(work_dir),
            "yolo_model_path": str(yolo_model) if yolo_model else None,
            "yolo_model_found": bool(yolo_model and yolo_model.is_file()),
            "yolo_model_bytes": yolo_model_bytes,
            "yolo_imgsz": yolo_imgsz,
            "vggt_enabled": vggt_enabled,
            "vggt_generate_if_missing": vggt_generate,
            "vggt_model_id": vggt_model_id,
            "vggt_embedded_python": str(embedded_python) if embedded_python else None,
            "vggt_skip_nodes": sorted(vggt_skip_nodes, key=_natural_key),
        },
        "dataset": {
            "selected_nodes": list(nodes),
            "unknown_requested_nodes": unknown_nodes,
            "node_count": len(nodes),
            "image_count": len(image_info),
            "views_per_node": {node_id: len(images) for node_id, images in nodes.items()},
            "input_bytes": input_bytes,
            "max_image_dimensions": (
                max(
                    ([item["width"], item["height"]] for item in image_info if item["width"] and item["height"]),
                    key=lambda size: size[0] * size[1],
                    default=None,
                )
            ),
        },
        "cache": cache,
        "hardware": {
            "os": platform.platform(),
            "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "ram_total_bytes": total_ram,
            "ram_available_bytes": available_ram,
            "disk": disk_info,
            "nvidia_gpus": _gpu_probe(),
        },
        "runtimes": {
            "analysis_python": {
                "path": sys.executable,
                "version": platform.python_version(),
                "torch": current_torch,
                "packages": _package_probe(),
            },
            "vggt_embedded_python": vggt_runtime,
        },
        "footprint": {
            "current_output_bytes": output_bytes,
            "embedded_runtime_bytes": embedded_runtime_bytes,
        },
        "historical_timing": historical_summary,
        "estimates": estimates,
    }


def _gib(value: int | float | None) -> str:
    return "n/d" if value is None else f"{value / GIB:.1f} GiB"


def _bytes(value: int | float | None) -> str:
    if value is None:
        return "n/d"
    if value < GIB:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value / GIB:.1f} GiB"


def _range(values: list[float]) -> str:
    return f"{values[0]:.1f}-{values[1]:.1f} GiB"


def _render_text(report: dict[str, Any]) -> str:
    config = report["configuration"]
    dataset = report["dataset"]
    cache = report["cache"]
    hardware = report["hardware"]
    runtimes = report["runtimes"]
    estimates = report["estimates"]
    current = estimates["cold_pipeline_current_vggt_runtime"]
    future_cuda = estimates["cold_pipeline_if_vggt_cuda_float32"]
    cached = estimates["cached_pipeline"]
    storage = estimates["storage"]
    dimensions = dataset["max_image_dimensions"]
    dimensions_text = f"{dimensions[0]}x{dimensions[1]} px" if dimensions else "n/d"

    lines = [
        "ANALISI REQUISITI PIPELINE",
        "=" * 72,
        f"Dataset: {dataset['node_count']} nodi, {dataset['image_count']} immagini, "
        f"max {max(dataset['views_per_node'].values(), default=0)} viste/nodo",
        f"Immagine massima: {dimensions_text}; input selezionato: {_bytes(dataset['input_bytes'])}",
        f"YOLO: imgsz={config['yolo_imgsz']}, batch=1, pesi={_bytes(config['yolo_model_bytes'])}",
        f"VGGT: {config['vggt_model_id']}, 518 px, tutte le viste del nodo insieme",
        "",
        "CACHE DEL PROSSIMO RUN",
        f"  Detection valide: {cache['detection']['valid']}/{cache['detection']['images']} "
        f"(mancanti/obsolete: {cache['detection']['missing_or_stale']})",
        f"  Artefatti VGGT:  {cache['vggt']['present']}/{cache['vggt']['expected']} "
        f"(mancanti: {cache['vggt']['missing']})",
        "",
        "HARDWARE RILEVATO",
        f"  CPU: {hardware['cpu']} ({hardware['logical_cpu_count']} thread logici)",
        f"  RAM: {_gib(hardware['ram_total_bytes'])} totali, {_gib(hardware['ram_available_bytes'])} disponibili",
        f"  Disco libero: {_gib(hardware['disk']['free_bytes'])}",
    ]
    if hardware["nvidia_gpus"]:
        for gpu in hardware["nvidia_gpus"]:
            lines.append(
                f"  GPU {gpu['index']}: {gpu['name']} - {gpu['memory_total_mib'] / 1024:.1f} GiB VRAM "
                f"({gpu['memory_free_mib'] / 1024:.1f} GiB liberi)"
            )
    else:
        lines.append("  GPU NVIDIA: non rilevata tramite nvidia-smi")

    main_torch = runtimes["analysis_python"]["torch"]
    vggt_runtime = runtimes["vggt_embedded_python"]
    lines.extend(
        [
            "",
            "RUNTIME",
            f"  Python usato per l'analisi: {runtimes['analysis_python']['path']}",
            f"  Torch principale: {main_torch.get('version', 'non installato')}; "
            f"CUDA={'sì' if main_torch.get('cuda_available') else 'no'}",
            f"  Torch VGGT embedded: {vggt_runtime.get('torch', 'non disponibile')}; "
            f"CUDA={'sì' if vggt_runtime.get('cuda_available') else 'no'}",
        ]
    )
    missing_packages = [
        name for name, installed in runtimes["analysis_python"]["packages"].items() if not installed
    ]
    if missing_packages:
        lines.append(f"  Dipendenze mancanti nel Python corrente: {', '.join(missing_packages)}")

    lines.extend(
        [
            "",
            "STIME DI PICCO (CONSERVATIVE, DA VALIDARE CON UN BENCHMARK)",
            "  1) Tutte le cache valide:",
            f"     RAM {_range(cached['ram_range_gib'])}; consigliati {cached['recommended_ram_gib']} GiB; VRAM 0 GiB",
            "  2) Run completo da zero con il runtime VGGT attuale:",
            f"     VGGT su {current['vggt_execution']}; RAM {_range(current['ram_range_gib'])}; "
            f"consigliati {current['recommended_ram_gib']} GiB",
            f"     YOLO VRAM {_range(current['yolo_vram_range_gib'])}; "
            f"GPU consigliata >= {current['recommended_gpu_vram_gib']} GiB",
            "  3) Se in futuro VGGT viene eseguito su CUDA in float32:",
            f"     VGGT VRAM {_range(future_cuda['vggt_vram_range_gib'])}; "
            f"picco combinato superiore ~{future_cuda['combined_yolo_plus_vggt_vram_upper_gib']:.1f} GiB",
            f"     GPU consigliata >= {future_cuda['recommended_gpu_vram_gib']} GiB",
            f"  Spazio libero consigliato per installazione + dati di lavoro: {storage['recommended_free_space_gib']} GiB",
            "  CPU: minimo pratico 4 core / 8 thread; consigliati 8-12 core e SSD NVMe",
            "",
            "LETTURA DEL RISULTATO",
        ]
    )
    if vggt_runtime.get("available") and not vggt_runtime.get("cuda_available"):
        lines.append(
            "  Il runtime VGGT installato e' CPU-only: oggi la VRAM serve solo a YOLO; "
            "VGGT pesa invece soprattutto su RAM e tempo CPU."
        )
    elif vggt_runtime.get("cuda_available"):
        lines.append(
            "  VGGT usa CUDA: il suo picco domina la VRAM e si somma alla memoria YOLO "
            "rimasta allocata nel processo principale."
        )
    else:
        lines.append("  Il runtime VGGT non e' sondabile; verificare il relativo Python embedded.")

    next_peak = estimates["next_run_estimated_peak"]
    if next_peak["vggt_views_requiring_generation_in_largest_node"]:
        lines.append(
            "  Nel prossimo run risultano artefatti VGGT mancanti; il nodo piu' grande da "
            f"generare ha {next_peak['vggt_views_requiring_generation_in_largest_node']} viste."
        )
    else:
        lines.append("  Per la selezione corrente gli artefatti VGGT risultano gia' in cache.")

    history = report["historical_timing"]
    if history.get("cpu_samples"):
        low, high = history["cpu_seconds_range"]
        lines.append(
            f"  Dati storici locali VGGT CPU: {history['cpu_samples']} run, {low:.1f}-{high:.1f} s/nodo, "
            f"totale {history['cpu_total_seconds'] / 60:.1f} min."
        )
    if dataset["unknown_requested_nodes"]:
        lines.append(f"  Nodi richiesti ma non trovati: {', '.join(dataset['unknown_requested_nodes'])}")
    lines.append(
        "  Le stime VRAM sono intervalli prudenziali basati su modello, risoluzione e viste; "
        "non sostituiscono la misura di torch.cuda.max_memory_allocated()."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analizza hardware, dataset, cache e requisiti della pipeline.",
    )
    parser.add_argument("nodes", nargs="*", help="ID nodo opzionali (es. 2 9 15_2).")
    parser.add_argument("--json", action="store_true", help="Stampa il report JSON invece del testo.")
    parser.add_argument("--output", type=Path, help="Salva sempre il report strutturato in un file JSON.")
    args = parser.parse_args()

    report = analyze(args.nodes)
    if args.output:
        output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(_render_text(report))
        if args.output:
            print(f"\nReport JSON: {output_path.resolve()}")
    return 2 if report["dataset"]["unknown_requested_nodes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
