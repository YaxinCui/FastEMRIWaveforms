#!/usr/bin/env python3
"""Validate the installed opt-in Metal backend on real Kerr waveforms.

2026-09-02 14:02 CST (mac): Exercise the public ``force_backend="metal"``
path, compare it with the CPU summation object inside one shared full-table
generator, enforce the accepted waveform gates, and record reproducible
performance/provenance evidence without modifying global backend callables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import few
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "collaboration/mac/metal_backend_end_to_end.json"
NORMALIZED_LIMIT = 5.0e-10
RELATIVE_L2_LIMIT = 5.0e-10
MISMATCH_LIMIT = 1.0e-10
WAVEFORM_ARGS = (1.0e6, 1.0e1, 0.7, 11.0, 0.4, 1.0, np.pi / 3, np.pi / 4)
BASE_KWARGS = {
    "dist": 1.0,
    "Phi_phi0": 0.3,
    "Phi_theta0": 0.0,
    "Phi_r0": 0.7,
    "dt": 15.0,
}
CASES = (
    {"key": "baseline_short", "a": 0.7, "p0": 11.0, "e0": 0.4, "xI0": 1.0, "T": 0.001},
    {"key": "baseline_one_year", "a": 0.7, "p0": 11.0, "e0": 0.4, "xI0": 1.0, "T": 1.0},
    {
        "key": "positive_spin_retrograde",
        "a": 0.7,
        "p0": 11.0,
        "e0": 0.4,
        "xI0": -1.0,
        "T": 0.01,
    },
    {"key": "inner_orbit", "a": 0.6, "p0": 8.0, "e0": 0.3, "xI0": 1.0, "T": 0.01},
    {"key": "zero_spin", "a": 0.0, "p0": 11.0, "e0": 0.4, "xI0": 1.0, "T": 0.01},
)
DATA_PATHS = (
    PROJECT_ROOT / "src/few/data/KerrEccEqFluxData.h5",
    PROJECT_ROOT / "src/few/data/ZNAmps_l10_m10_n55_DS2Outer.h5",
)
SOURCE_PATHS = (
    PROJECT_ROOT / "CMakeLists.txt",
    PROJECT_ROOT / "src/few/CMakeLists.txt",
    PROJECT_ROOT / "src/few/amplitude/ampinterp2d.py",
    PROJECT_ROOT / "src/few/cutils/CMakeLists.txt",
    PROJECT_ROOT / "src/few/cutils/__init__.py",
    PROJECT_ROOT / "src/few/cutils/metal_sum.hh",
    PROJECT_ROOT / "src/few/cutils/metal_sum.mm",
    PROJECT_ROOT / "src/few/cutils/pymetal.pyx",
    PROJECT_ROOT / "src/few/utils/baseclasses.py",
    Path(__file__).resolve(),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def waveform_args(case: dict[str, Any]) -> tuple[float, ...]:
    return (
        WAVEFORM_ARGS[0],
        WAVEFORM_ARGS[1],
        case["a"],
        case["p0"],
        case["e0"],
        case["xI0"],
        WAVEFORM_ARGS[6],
        WAVEFORM_ARGS[7],
    )


def waveform_kwargs(case: dict[str, Any]) -> dict[str, float]:
    return {**BASE_KWARGS, "T": case["T"]}


def timed_repeated(
    generator: Any,
    args: tuple[float, ...],
    kwargs: dict[str, float],
    repetitions: int,
    *,
    record_gpu: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    first_output = None
    last_output = None
    seconds = []
    gpu_seconds = []
    repeatable = True
    for _ in range(repetitions + 1):
        start = time.perf_counter()
        output = np.ascontiguousarray(
            np.asarray(generator(*args, **kwargs), dtype=np.complex128)
        )
        seconds.append(time.perf_counter() - start)
        if record_gpu:
            gpu_seconds.append(generator.create_waveform.backend.last_gpu_seconds)
        if first_output is None:
            first_output = output.copy()
        else:
            repeatable = repeatable and np.array_equal(first_output, output)
        last_output = output.copy()
    return last_output, {
        "cold_seconds": seconds[0],
        "warm_seconds": seconds[1:],
        "warm_median_seconds": statistics.median(seconds[1:]),
        "gpu_seconds": gpu_seconds,
        "bitwise_repeatable": repeatable,
    }


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = candidate - reference
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    result = {
        "normalized_max": float(np.max(np.abs(delta)) / scale),
        "relative_l2": float(np.linalg.norm(delta) / norm),
        "flat_mismatch": float(
            max(0.0, get_mismatch(reference, candidate, use_gpu=False))
        ),
    }
    checks = {
        "normalized_max": result["normalized_max"] <= NORMALIZED_LIMIT,
        "relative_l2": result["relative_l2"] <= RELATIVE_L2_LIMIT,
        "flat_mismatch": result["flat_mismatch"] <= MISMATCH_LIMIT,
    }
    result.update(
        {
            "limits": {
                "normalized_max": NORMALIZED_LIMIT,
                "relative_l2": RELATIVE_L2_LIMIT,
                "flat_mismatch": MISMATCH_LIMIT,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    )
    return result


def run_case(
    generator: Any,
    cpu_sum: InterpolatedModeSum,
    metal_sum: InterpolatedModeSum,
    case: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    args = waveform_args(case)
    kwargs = waveform_kwargs(case)
    try:
        generator.create_waveform = cpu_sum
        cpu_output, cpu_timing = timed_repeated(generator, args, kwargs, repetitions)
        generator.create_waveform = metal_sum
        metal_output, metal_timing = timed_repeated(
            generator, args, kwargs, repetitions, record_gpu=True
        )
        generator.create_waveform = cpu_sum
        cpu_after = np.ascontiguousarray(
            np.asarray(generator(*args, **kwargs), dtype=np.complex128)
        )
    finally:
        generator.create_waveform = metal_sum

    comparison = metrics(cpu_output, metal_output)
    cpu_before_after_bitwise = bool(np.array_equal(cpu_output, cpu_after))
    finite = bool(
        np.all(np.isfinite(metal_output.real))
        and np.all(np.isfinite(metal_output.imag))
    )
    passed = bool(
        comparison["passed"]
        and cpu_timing["bitwise_repeatable"]
        and metal_timing["bitwise_repeatable"]
        and cpu_before_after_bitwise
        and finite
    )
    return {
        "key": case["key"],
        "inputs": {**case, **BASE_KWARGS},
        "shape": list(metal_output.shape),
        "dtype": str(metal_output.dtype),
        "modes_kept": int(generator.num_modes_kept),
        "finite": finite,
        "cpu": {
            **cpu_timing,
            "before_after_bitwise": cpu_before_after_bitwise,
        },
        "metal": metal_timing,
        "metal_vs_cpu": comparison,
        "warm_end_to_end_speedup": (
            cpu_timing["warm_median_seconds"] / metal_timing["warm_median_seconds"]
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("The Metal backend acceptance requires Apple Silicon")
    if args.repetitions < 2:
        raise ValueError("At least two warm repetitions are required")

    default_backend = InterpolatedModeSum().backend_name
    if default_backend == "metal":
        raise RuntimeError("Metal must not replace FEW's default backend selection")
    load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend="metal",
    )
    model_load_seconds = time.perf_counter() - load_start
    metal_sum = generator.create_waveform
    cpu_sum = InterpolatedModeSum(force_backend="cpu", pad_output=False)
    case_reports = [
        run_case(generator, cpu_sum, metal_sum, case, args.repetitions)
        for case in CASES
    ]

    report = {
        "schema": 1,
        "collaboration_note": (
            "2026-09-02 14:02 CST (mac): installed explicit Metal backend, "
            "shared-generator CPU comparison, no global callable injection"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "few": few.__version__,
            "numpy": np.__version__,
        },
        "repository": {
            "branch": subprocess.run(
                ["git", "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            ).stdout.strip(),
            "base_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            ).stdout.strip(),
        },
        "selection": {
            "requested_backend": generator.backend_name,
            "amplitude_backend": generator.amplitude_generator.backend_name,
            "summation_backend": metal_sum.backend_name,
            "default_backend": default_backend,
            "numpy_host_storage": metal_sum.backend.uses_numpy,
            "metal_feature": metal_sum.backend.uses_metal,
        },
        "model_load_seconds": model_load_seconds,
        "data_files": {path.name: file_identity(path) for path in DATA_PATHS},
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): file_identity(path)
            for path in SOURCE_PATHS
        },
        "cases": case_reports,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024),
        "passed": all(case["passed"] for case in case_reports),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
