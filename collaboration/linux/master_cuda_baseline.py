#!/usr/bin/env python3
"""Measure the master-rooted FP64 CUDA Kerr control for the 5x experiment.

2026-09-04 14:35 CST (linux): Add a Linux-owned, synchronized benchmark for
the clean ``origin/master`` baseline after the required CPU/GPU frequency-
spline compatibility fix.  It records cold construction, warm
all-mode amplitude interpolation, representative end-to-end Kerr waveforms,
memory use, provenance, and a Python-level profile.  It does not modify FEW's
runtime behavior and must be run against an isolated wheel built from HEAD.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import os
import platform
import pstats
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np

import few
from few import get_backend, get_file_manager
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILENAME = "ZNAmps_l10_m10_n55_DS2Outer.h5"
EXPECTED_MASTER = "47e4fea4bb3e5fbe1ba34e5e399ed37c97814191"
DEFAULT_DURATIONS = (0.001, 0.01, 1.0)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_memory_mib() -> dict[str, float]:
    values: dict[str, float] = {}
    with Path("/proc/self/status").open(encoding="utf-8") as stream:
        for line in stream:
            key, _, raw = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                values[key] = float(raw.split()[0]) / 1024.0
    return values


def synchronize(xp: Any) -> None:
    xp.cuda.Stream.null.synchronize()


def timing_summary(
    xp: Any,
    function: Callable[[], Any],
    repetitions: int,
    warmups: int = 2,
) -> tuple[Any, dict[str, Any]]:
    output = None
    for _ in range(warmups):
        output = function()
        synchronize(xp)

    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = function()
        synchronize(xp)
        samples.append(time.perf_counter() - start)

    values = np.asarray(samples, dtype=np.float64)
    return output, {
        "samples_ms": [float(value * 1.0e3) for value in values],
        "median_ms": float(np.median(values) * 1.0e3),
        "minimum_ms": float(np.min(values) * 1.0e3),
        "p90_ms": float(np.quantile(values, 0.9) * 1.0e3),
    }


def python_profile(
    xp: Any, function: Callable[[], Any], limit: int = 40
) -> list[dict[str, Any]]:
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    synchronize(xp)
    profiler.disable()
    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, name), (primitive, calls, self_s, cumulative_s, _) in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:limit]:
        entries.append(
            {
                "file": filename,
                "line": int(line),
                "function": name,
                "primitive_calls": int(primitive),
                "calls": int(calls),
                "self_ms": float(self_s * 1.0e3),
                "cumulative_ms": float(cumulative_s * 1.0e3),
            }
        )
    return entries


def run_baseline(
    repetitions: int, durations: tuple[float, ...], profile_duration: float
) -> dict[str, Any]:
    commit = git_output("rev-parse", "HEAD")
    if commit != EXPECTED_MASTER:
        raise RuntimeError(
            f"baseline requires clean master {EXPECTED_MASTER}, found {commit}"
        )
    if "few-cuda5x-master" not in str(Path(few.__file__).resolve()):
        raise RuntimeError(
            "FEW was not imported from the isolated current-master wheel runtime"
        )

    backend = get_backend("cuda12x")
    if not backend.uses_cupy:
        raise RuntimeError("cuda12x did not expose a CuPy backend")
    xp = backend.xp
    synchronize(xp)
    properties = xp.cuda.runtime.getDeviceProperties(0)
    data_path = Path(get_file_manager().get_file(DATA_FILENAME)).resolve()

    pool = xp.get_default_memory_pool()
    memory_before = process_memory_mib()
    construct_start = time.perf_counter()
    amplitude = AmpInterpKerrEccEq(force_backend="cuda12x")
    synchronize(xp)
    construct_s = time.perf_counter() - construct_start
    memory_after_amplitude = process_memory_mib()

    amplitude_cases = []
    for points in (1, 32, 128):
        p = xp.full(points, 8.0, dtype=xp.float64)
        e = xp.full(points, 0.3, dtype=xp.float64)
        x_i = xp.ones(points, dtype=xp.float64)
        output, timing = timing_summary(
            xp,
            lambda p=p, e=e, x_i=x_i: amplitude(0.6, p, e, x_i),
            repetitions,
        )
        amplitude_cases.append(
            {
                "trajectory_points": points,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "timing": timing,
            }
        )

    # 2026-09-04 14:38 CST (linux): Release the standalone amplitude model
    # before constructing the full waveform model.  Both own the same ~5 GB
    # table, so retaining both would measure an artificial duplicate and can
    # exhaust an 11 GB RTX 2080 Ti.
    del amplitude
    gc.collect()
    pool.free_all_blocks()

    waveform_construct_start = time.perf_counter()
    waveform = FastKerrEccentricEquatorialFlux(force_backend="cuda12x")
    synchronize(xp)
    waveform_construct_s = time.perf_counter() - waveform_construct_start
    memory_after_waveform = process_memory_mib()

    waveform_args = (1.0e6, 1.0e1, 0.6, 8.0, 0.3, 1.0, np.pi / 3, np.pi / 4)
    waveform_cases = []
    for duration in durations:
        output, timing = timing_summary(
            xp,
            lambda duration=duration: waveform(
                *waveform_args, T=duration, dt=15.0, dist=1.0
            ),
            repetitions,
        )
        waveform_cases.append(
            {
                "duration_years": duration,
                "samples": int(output.size),
                "output_dtype": str(output.dtype),
                "modes_kept": int(waveform.num_modes_kept),
                "timing": timing,
            }
        )

    profile = python_profile(
        xp,
        lambda: waveform(
            *waveform_args, T=profile_duration, dt=15.0, dist=1.0
        ),
    )
    synchronize(xp)
    return {
        "schema": 1,
        "collaboration_note": (
            "2026-09-04 14:37 CST (linux): Linux-owned master-rooted FP64 "
            "control for the codex 5x mixed-compute branch. The only source "
            "change is the required host/device frequency-spline compatibility "
            "fix; no precision or arithmetic path is changed."
        ),
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "branch": git_output("branch", "--show-current"),
        "project_commit": commit,
        "working_tree_status": git_output("status", "--short"),
        "few_version": few.__version__,
        "few_module": str(Path(few.__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": xp.__version__,
        "cuda_runtime": int(xp.cuda.runtime.runtimeGetVersion()),
        "gpu": {
            "name": properties["name"].decode(),
            "compute_capability": [
                int(properties["major"]),
                int(properties["minor"]),
            ],
            "total_memory_mib": float(properties["totalGlobalMem"] / 2**20),
        },
        "data": {
            "path": str(data_path),
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "measurement_contract": {
            "reference": (
                "origin/master FP64 CUDA path plus the required host-eval "
                "frequency-spline compatibility fix"
            ),
            "timing_scope": "warm synchronized end-to-end wall time",
            "warmups": 2,
            "repetitions": repetitions,
            "dt_seconds": 15.0,
            "durations_years": list(durations),
            "five_x_gate": (
                "candidate median <= 0.20 * this baseline median for each "
                "declared representative workload"
            ),
            "profile_warning": (
                "cProfile reports Python/C-extension wall time and cannot "
                "attribute individual asynchronous GPU kernels"
            ),
        },
        "construction": {
            "standalone_amplitude_seconds": construct_s,
            "waveform_seconds": waveform_construct_s,
        },
        "amplitude_cases": amplitude_cases,
        "waveform_cases": waveform_cases,
        "python_profile_duration_years": profile_duration,
        "python_profile_top_cumulative": profile,
        "memory": {
            "process_before_amplitude_mib": memory_before,
            "process_after_amplitude_mib": memory_after_amplitude,
            "process_after_waveform_mib": memory_after_waveform,
            "cupy_pool_used_mib": float(pool.used_bytes() / 2**20),
            "cupy_pool_reserved_mib": float(pool.total_bytes() / 2**20),
        },
        "environment": {
            "FEW_FILE_EXTRA_PATHS": os.environ.get("FEW_FILE_EXTRA_PATHS"),
            "FEW_FILE_ALLOW_DOWNLOAD": os.environ.get("FEW_FILE_ALLOW_DOWNLOAD"),
            "FEW_FILE_INTEGRITY_CHECK": os.environ.get("FEW_FILE_INTEGRITY_CHECK"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--durations", nargs="+", type=float, default=list(DEFAULT_DURATIONS)
    )
    parser.add_argument("--profile-duration", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 3:
        raise ValueError("--repetitions must be at least 3")
    if any(duration <= 0.0 for duration in args.durations):
        raise ValueError("durations must be positive")
    report = run_baseline(
        args.repetitions, tuple(args.durations), args.profile_duration
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
