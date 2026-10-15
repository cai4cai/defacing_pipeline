"""Process-isolated GPU batch execution with one output directory per volume."""

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def resolve_gpu_devices(devices, environ=None):
    """Resolve indices within an inherited CUDA visibility list without widening it."""
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    allowed = None if visible is None else [x.strip() for x in visible.split(",") if x.strip()]
    resolved = []
    for device in devices:
        device = str(device)
        if not re.fullmatch(r"\d+|GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device):
            raise ValueError("GPU devices must be nonnegative indices or full GPU UUIDs")
        if allowed is not None:
            if device.isdigit():
                if int(device) >= len(allowed):
                    raise ValueError("GPU index is outside CUDA_VISIBLE_DEVICES")
                device = allowed[int(device)]
            elif device not in allowed:
                raise ValueError("GPU UUID is not explicitly in CUDA_VISIBLE_DEVICES")
            if not device or device == "-1":
                raise ValueError("CUDA visibility disables the requested GPU")
        if device.isdigit():
            device = str(int(device))
        if device in resolved:
            raise ValueError("Duplicate GPU; use workers_per_gpu to share a device")
        resolved.append(device)
    if not resolved:
        raise ValueError("Select at least one GPU")
    return resolved


def _stop(process):
    """Kill the worker's process group, including HD-BET/registration children."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_parallel(input_dir, output_dir, pipeline_options, gpu_devices,
                 workers_per_gpu=1, threads_per_worker=1, job_timeout=3600):
    """Run full MRI pipelines in fresh subprocesses on explicitly assigned GPUs.

    This opt-in path uses ``jobs/<input-relative-path hash>/`` for isolated
    results and logs. ``parallel_summary.json`` maps every input to its output.
    The output root must be fresh. Each input is attempted once; no automatic
    retry or implicit CPU fallback is performed. CPU thread settings are hints;
    use a container/job scheduler to impose hard CPU and memory limits.
    """
    if os.name != "posix":
        raise ValueError("Parallel execution currently requires POSIX process groups")
    if (type(workers_per_gpu) is not int or workers_per_gpu < 1 or
            type(threads_per_worker) is not int or threads_per_worker < 1):
        raise ValueError("Worker and thread counts must be positive integers")
    if not math.isfinite(job_timeout) or job_timeout <= 0:
        raise ValueError("Job timeout must be positive and finite")
    if pipeline_options.get("modality", "mri") != "mri":
        raise ValueError("Parallel execution currently supports MRI only")
    if pipeline_options.get("device") not in (None, "cuda"):
        raise ValueError("GPU workers require CUDA inference")
    devices = resolve_gpu_devices(gpu_devices)
    source = Path(input_dir).resolve(strict=True)
    destination = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("Input must be a directory")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Input and output roots must not overlap")
    files = sorted(p for p in source.rglob("*.nii.gz") if p.is_file())
    if not files:
        raise ValueError("No NIfTI inputs found")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    (destination / "jobs").mkdir(mode=0o700)
    options = {**pipeline_options, "device": "cuda"}
    pending = deque(files)
    slots = [{"gpu": gpu, "active": None} for gpu in devices for _ in range(workers_per_gpu)]
    rows = []
    started = time.monotonic()

    def receipt():
        summary = {"workers": len(slots), "threads_per_worker": threads_per_worker,
                   "gpu_devices": devices, "elapsed_seconds": time.monotonic() - started,
                   "inputs": len(files), "completed": len(rows),
                   "successful": sum(row["success"] for row in rows), "jobs": rows}
        tmp = destination / "parallel_summary.tmp"
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(destination / "parallel_summary.json")
        return summary

    try:
        while pending or any(slot["active"] for slot in slots):
            for slot in slots:
                active = slot["active"]
                if active:
                    process = active["process"]
                    expired = time.monotonic() - active["start"] > job_timeout
                    if process.poll() is None and not expired:
                        continue
                    if expired and process.poll() is None:
                        _stop(process)
                    _stop(process)
                    active["log"].close()
                    result_path = active["root"] / "result.json"
                    result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    rows.append({"input": active["relative"],
                                 "output": str(active["root"].relative_to(destination) / "output"),
                                 "gpu": slot["gpu"], "exit_code": process.returncode,
                                 "timed_out": expired,
                                 "stage_seconds": result.get("stage_seconds", {}),
                                 "success": process.returncode == 0 and not expired and result.get("success") is True,
                                 "elapsed_seconds": time.monotonic() - active["start"]})
                    slot["active"] = None
                    receipt()
                if not pending:
                    continue
                path = pending.popleft()
                relative = str(path.relative_to(source))
                job = destination / "jobs" / hashlib.sha256(relative.encode()).hexdigest()
                job.mkdir(mode=0o700)
                (job / "input").mkdir(mode=0o700)
                (job / "input" / path.name).symlink_to(path)
                config = {"input": str(job / "input"), "output": str(job / "output"),
                          "pipeline_options": options}
                (job / "job.json").write_text(json.dumps(config))
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": slot["gpu"],
                       "TORCHINDUCTOR_CACHE_DIR": str(job / "cache" / "torchinductor"),
                       "TRITON_CACHE_DIR": str(job / "cache" / "triton")}
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
                    env[key] = str(threads_per_worker)
                log = (job / "worker.log").open("x")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "caideface._batch_worker", str(job / "job.json")],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                slot["active"] = {"process": process, "root": job, "relative": relative,
                                  "start": time.monotonic(), "log": log}
            time.sleep(0.05)
    finally:
        for slot in slots:
            active = slot["active"]
            if active:
                # Kill descendants even if the top-level worker has just exited.
                _stop(active["process"])
                active["log"].close()
        receipt()
    return receipt()
