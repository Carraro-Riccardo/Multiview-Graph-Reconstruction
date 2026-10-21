"""Run a cache-free pipeline inference and measure its hardware peaks.

This tool is designed for a Slurm GPU node.  It launches the real ``run.py``
entry point with a new WORK_DIR, samples the complete process tree, and writes:

* ``benchmark_report.json``: environment, timings, RAM/VRAM peaks and a VRAM
  recommendation with 20% headroom;
* ``benchmark_samples.csv``: the raw time series used for the report;
* the normal pipeline artefacts below the selected benchmark directory.

GPU memory is read per process with ``nvidia-smi``.  Memory owned by the parent
pipeline process is labelled ``main`` (normally YOLO); memory owned by child
processes is labelled ``children`` (normally the VGGT exporter).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIB = 1024**3
MIB = 1024**2


def _natural_key(value: str) -> list[tuple[int, Any]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _module_status() -> dict[str, bool]:
    modules = [
        "numpy",
        "scipy",
        "cv2",
        "PIL",
        "networkx",
        "matplotlib",
        "ultralytics",
        "shurikodecontext",
        "torch",
        "vggt",
    ]
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    payload: dict[str, Any] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        payload["device_count"] = int(torch.cuda.device_count())
        # Avoid creating a CUDA context in the monitoring parent. Device model
        # and capacity are collected out-of-process through nvidia-smi instead.
    return payload


def _nvidia_devices() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    command = [
        executable,
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            devices.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "driver_version": fields[4],
                }
            )
        except ValueError:
            continue
    return devices


def _gpu_process_memory() -> dict[int, dict[str, Any]]:
    """Return CUDA memory attributed to each compute-process PID."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {}
    completed = None
    # Driver releases have exposed the memory property under both names.
    for memory_field in ("used_gpu_memory", "used_memory"):
        command = [
            executable,
            f"--query-compute-apps=pid,{memory_field},gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
        try:
            candidate = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return {}
        if candidate.returncode == 0:
            completed = candidate
            break
    if completed is None:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            used_mib = float(fields[1])
        except ValueError:
            continue
        result[pid] = {"used_mib": used_mib, "gpu_uuid": fields[2]}
    return result


def _linux_process_tree(root_pid: int) -> set[int]:
    ppid_by_pid: dict[int, int] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return {root_pid}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # Field 2 can contain spaces and parentheses. Everything following
            # the final ')' starts with state and PPID.
            stat = (entry / "stat").read_text(encoding="utf-8")
            tail = stat.rsplit(")", 1)[1].strip().split()
            ppid_by_pid[int(entry.name)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue

    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_by_pid.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def _process_tree(root_pid: int) -> set[int]:
    try:
        import psutil  # type: ignore

        process = psutil.Process(root_pid)
        return {root_pid, *(child.pid for child in process.children(recursive=True))}
    except Exception:
        return _linux_process_tree(root_pid)


def _linux_memory(pids: set[int]) -> tuple[float, float]:
    rss_kib = 0
    swap_kib = 0
    for pid in pids:
        try:
            lines = (Path("/proc") / str(pid) / "status").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                rss_kib += int(line.split()[1])
            elif line.startswith("VmSwap:"):
                swap_kib += int(line.split()[1])
    return rss_kib / 1024.0, swap_kib / 1024.0


def _process_memory(pids: set[int]) -> tuple[float, float]:
    try:
        import psutil  # type: ignore

        rss = 0
        swap = 0
        for pid in pids:
            try:
                info = psutil.Process(pid).memory_full_info()
                rss += int(info.rss)
                swap += int(getattr(info, "swap", 0))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss / MIB, swap / MIB
    except Exception:
        return _linux_memory(pids)


def _sample(root_pid: int, started: float) -> dict[str, Any]:
    pids = _process_tree(root_pid)
    rss_mib, swap_mib = _process_memory(pids)
    gpu_processes = _gpu_process_memory()
    main_gpu_mib = gpu_processes.get(root_pid, {}).get("used_mib", 0.0)
    child_gpu_mib = sum(
        payload["used_mib"]
        for pid, payload in gpu_processes.items()
        if pid in pids and pid != root_pid
    )
    uuids = sorted(
        {
            payload["gpu_uuid"]
            for pid, payload in gpu_processes.items()
            if pid in pids
        }
    )
    return {
        "elapsed_sec": round(time.monotonic() - started, 3),
        "process_count": len(pids),
        "rss_mib": round(rss_mib, 1),
        "swap_mib": round(swap_mib, 1),
        "gpu_main_mib": round(main_gpu_mib, 1),
        "gpu_children_mib": round(child_gpu_mib, 1),
        "gpu_total_mib": round(main_gpu_mib + child_gpu_mib, 1),
        "gpu_uuids": ";".join(uuids),
    }


def _discover_nodes(images_dir: Path) -> dict[str, list[Path]]:
    pattern = re.compile(r"^node(.+)_(\d+)\.jpg$", re.IGNORECASE)
    nodes: dict[str, list[Path]] = {}
    for path in sorted(images_dir.glob("node*_*.jpg")):
        match = pattern.match(path.name)
        if match:
            nodes.setdefault(match.group(1), []).append(path)
    return dict(sorted(nodes.items(), key=lambda item: _natural_key(item[0])))


def _load_pipeline_settings() -> tuple[Path, set[str]]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from graph_reconstruction.config import IMAGES_DIR, VGGT_SKIP_NODES

    return Path(IMAGES_DIR), {str(node_id) for node_id in VGGT_SKIP_NODES}


def _select_nodes(requested: list[str], run_all: bool) -> tuple[list[str], dict[str, int]]:
    images_dir, vggt_skip_nodes = _load_pipeline_settings()
    nodes = _discover_nodes(images_dir)
    if not nodes:
        raise RuntimeError(f"Nessuna immagine node<id>_<frame>.jpg trovata in {images_dir}")
    if requested:
        unknown = [node_id for node_id in requested if node_id not in nodes]
        if unknown:
            raise RuntimeError(f"Nodi non trovati: {', '.join(unknown)}")
        selected = requested
    elif run_all:
        selected = list(nodes)
    else:
        candidates = [item for item in nodes.items() if item[0] not in vggt_skip_nodes]
        if not candidates:
            candidates = list(nodes.items())
        selected = [max(candidates, key=lambda item: len(item[1]))[0]]
    return selected, {node_id: len(nodes[node_id]) for node_id in selected}


def _standard_vram_capacity(required_gib: float) -> int:
    for capacity in (4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80):
        if capacity >= required_gib:
            return capacity
    return int((required_gib + 7) // 8 * 8)


def _write_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    fieldnames = [
        "elapsed_sec",
        "process_count",
        "rss_mib",
        "swap_mib",
        "gpu_main_mib",
        "gpu_children_mib",
        "gpu_total_mib",
        "gpu_uuids",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _internal_cuda_metrics(output_dir: Path) -> dict[str, Any]:
    """Collect allocator peaks written inside YOLO and VGGT processes.

    These values remain available when CUDA MPS hides client allocations from
    nvidia-smi's process table.
    """
    yolo = _load_json(output_dir / "benchmark_yolo_cuda.json") or {}
    vggt_runs = []
    for summary_path in sorted((output_dir / "multiview").glob("node*/vggt/summary.json")):
        summary = _load_json(summary_path)
        cuda_memory = summary.get("cuda_memory") if summary else None
        if isinstance(cuda_memory, dict):
            vggt_runs.append(
                {
                    "summary_path": str(summary_path),
                    "node": summary_path.parents[1].name,
                    **cuda_memory,
                }
            )

    yolo_persistent = int(yolo.get("current_reserved_bytes", 0))
    yolo_peak = int(yolo.get("peak_reserved_bytes", 0))
    vggt_peak = max((int(run.get("peak_reserved_bytes", 0)) for run in vggt_runs), default=0)
    return {
        "yolo": yolo or None,
        "vggt_runs": vggt_runs,
        "yolo_peak_reserved_bytes": yolo_peak,
        "yolo_persistent_reserved_bytes": yolo_persistent,
        "vggt_peak_reserved_bytes": vggt_peak,
        "combined_reserved_bytes": yolo_persistent + vggt_peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Esegue run.py senza cache e misura picchi RAM/VRAM dell'intero process tree.",
    )
    parser.add_argument("nodes", nargs="*", help="Nodi da provare; default: nodo VGGT con piu' viste.")
    parser.add_argument("--all", action="store_true", help="Esegue tutti i nodi sequenzialmente.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="WORK_DIR nuovo per benchmark; default output/benchmarks/<timestamp>.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.25, help="Intervallo campioni in secondi.")
    parser.add_argument("--require-cuda", action="store_true", help="Interrompe se CUDA non e' disponibile.")
    args = parser.parse_args()

    if args.all and args.nodes:
        parser.error("Usare --all oppure gli ID nodo, non entrambi.")
    if args.sample_interval < 0.1:
        parser.error("--sample-interval deve essere >= 0.1 secondi.")

    packages = _module_status()
    missing = [name for name, present in packages.items() if not present]
    if missing:
        print(f"ERRORE: dipendenze mancanti: {', '.join(missing)}", file=sys.stderr)
        return 2

    torch_info = _torch_info()
    if args.require_cuda and not torch_info.get("cuda_available"):
        print("ERRORE: CUDA non disponibile nel Python del benchmark.", file=sys.stderr)
        print(json.dumps(torch_info, indent=2), file=sys.stderr)
        return 2

    try:
        selected_nodes, views_per_node = _select_nodes(args.nodes, args.all)
    except Exception as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = args.output_dir or (PROJECT_ROOT / "output" / "benchmarks" / timestamp)
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (PROJECT_ROOT / output_dir).resolve()
    cache_markers = [output_dir / "detection_cache", output_dir / "multiview"]
    if any(marker.exists() for marker in cache_markers):
        print(
            f"ERRORE: {output_dir} contiene gia' cache/artefatti. Scegliere una directory nuova.",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env.update(
        {
            "WORK_DIR": str(output_dir),
            "DEBUG_OUTPUT_ENABLED": "false",
            "VGGT_EDGE_ENABLED": "true",
            "VGGT_GENERATE_IF_MISSING": "true",
            # The same CUDA-enabled environment is used for both pipeline and VGGT.
            "VGGT_EMBEDDED_PYTHON": sys.executable,
            "PIPELINE_RESOURCE_METRICS": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [sys.executable, str(PROJECT_ROOT / "run.py"), *selected_nodes]

    print("=" * 72, flush=True)
    print("CACHE-FREE INFERENCE BENCHMARK", flush=True)
    print(f"Started UTC:  {_utc_now()}", flush=True)
    print(f"Command:      {' '.join(command)}", flush=True)
    print(f"Nodes/views:  {views_per_node}", flush=True)
    print(f"Cold WORK_DIR:{output_dir}", flush=True)
    print(f"CUDA:         {torch_info.get('cuda_available')}", flush=True)
    print("=" * 72, flush=True)

    samples_path = output_dir / "benchmark_samples.csv"
    _write_samples(samples_path, [])
    started_wall = _utc_now()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_env)
    samples: list[dict[str, Any]] = []
    try:
        while process.poll() is None:
            samples.append(_sample(process.pid, started))
            # Preserve recent data even if Slurm stops the job at its time limit.
            if len(samples) % 10 == 0:
                _write_samples(samples_path, samples)
            time.sleep(args.sample_interval)
        # One final sample is useful when the child exits between polling ticks.
        samples.append(_sample(process.pid, started))
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=30)
        raise

    elapsed = time.monotonic() - started
    peak = lambda key: max((float(sample[key]) for sample in samples), default=0.0)
    peak_gpu_mib = peak("gpu_total_mib")
    internal_cuda = _internal_cuda_metrics(output_dir)
    internal_combined_mib = internal_cuda["combined_reserved_bytes"] / MIB
    sizing_peak_mib = max(peak_gpu_mib, internal_combined_mib)
    measured_gib = sizing_peak_mib / 1024.0
    gpu_measurement_valid = sizing_peak_mib > 0
    recommended_gib = _standard_vram_capacity(measured_gib * 1.20) if gpu_measurement_valid else None
    report = {
        "report_version": 1,
        "status": "ok" if process.returncode == 0 else "error",
        "returncode": process.returncode,
        "started_utc": started_wall,
        "finished_utc": _utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "project_root": str(PROJECT_ROOT),
        "work_dir": str(output_dir),
        "command": command,
        "nodes": selected_nodes,
        "views_per_node": views_per_node,
        "cold_run": True,
        "sample_interval_seconds": args.sample_interval,
        "sample_count": len(samples),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": packages,
            "torch": torch_info,
            "nvidia_devices": _nvidia_devices(),
        },
        "peaks": {
            "process_tree_rss_mib": peak("rss_mib"),
            "process_tree_swap_mib": peak("swap_mib"),
            "gpu_main_process_mib": peak("gpu_main_mib"),
            "gpu_child_processes_mib": peak("gpu_children_mib"),
            "gpu_combined_mib": peak_gpu_mib,
            "torch_allocator_yolo_peak_reserved_mib": internal_cuda["yolo_peak_reserved_bytes"] / MIB,
            "torch_allocator_yolo_persistent_reserved_mib": internal_cuda["yolo_persistent_reserved_bytes"] / MIB,
            "torch_allocator_vggt_peak_reserved_mib": internal_cuda["vggt_peak_reserved_bytes"] / MIB,
            "torch_allocator_combined_reserved_mib": internal_combined_mib,
            "sizing_peak_mib": sizing_peak_mib,
        },
        "internal_cuda_metrics": internal_cuda,
        "recommendation": {
            "gpu_measurement_valid": gpu_measurement_valid,
            "measured_peak_vram_gib": round(measured_gib, 3) if gpu_measurement_valid else None,
            "headroom_percent": 20,
            "peak_with_headroom_gib": round(measured_gib * 1.20, 3) if gpu_measurement_valid else None,
            "recommended_standard_gpu_vram_gib": recommended_gib,
            "scope": "Same model, max views, image preprocessing, precision and software stack as tested.",
        },
        "interpretation": {
            "gpu_main_process": "Pipeline parent; normally persistent YOLO allocations.",
            "gpu_child_processes": "Subprocesses; normally VGGT generation.",
            "gpu_combined": "nvidia-smi process-tree peak; it may be unavailable under CUDA MPS.",
            "sizing_peak": "Maximum of nvidia-smi and internal PyTorch concurrent estimates; used for sizing.",
        },
    }

    report_path = output_dir / "benchmark_report.json"
    _write_samples(samples_path, samples)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("BENCHMARK RESULT")
    print(f"Status:                   {report['status']} (exit {process.returncode})")
    print(f"Elapsed:                  {elapsed:.1f} s")
    print(f"Peak process-tree RAM:    {report['peaks']['process_tree_rss_mib'] / 1024:.2f} GiB")
    print(f"Peak YOLO/main VRAM:      {report['peaks']['gpu_main_process_mib'] / 1024:.2f} GiB")
    print(f"Peak VGGT/children VRAM:  {report['peaks']['gpu_child_processes_mib'] / 1024:.2f} GiB")
    print(f"PyTorch combined reserve: {internal_combined_mib / 1024:.2f} GiB")
    print(f"Sizing VRAM peak:         {measured_gib:.2f} GiB" if gpu_measurement_valid else "Sizing VRAM peak:         n/d")
    print(
        f"Recommended VRAM (+20%): {recommended_gib} GiB standard capacity"
        if recommended_gib is not None
        else "Recommended VRAM (+20%): n/d (nessun processo CUDA campionato)"
    )
    print(f"Report:                   {report_path}")
    print(f"Samples:                  {samples_path}")
    print("=" * 72)
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
