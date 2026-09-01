#!/usr/bin/env python3
"""Generate the five-case strict Metal artifact requested by Ubuntu.

2026-09-01 23:54 CST (mac): Add a deterministic, Mac-only generator for the
unrounded complex128 strict-Metal waveform arrays and structured provenance
report specified in the Ubuntu 23:45 CST handoff. Production backends and
registered data are never modified.
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
from benchmark_metal_hybrid import install_amplitude_holders
from benchmark_metal_interp import MetalContext, error_metrics
from benchmark_metal_interp import build_library as build_interp_library
from benchmark_metal_interp import load_api as load_interp_api
from benchmark_metal_sum import BASE_KWARGS, WAVEFORM_ARGS, MetalSummation
from benchmark_metal_sum import build_library as build_sum_library
from benchmark_metal_sum import load_api as load_sum_api

import few
from few import get_backend
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

SCHEMA = 1
SEED = 20260901
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = PROJECT_ROOT / "collaboration/mac/strict_metal_ds_reference.npz"
DEFAULT_REPORT = PROJECT_ROOT / "collaboration/mac/strict_metal_ds_report.json"
DATA_PATHS = (
    PROJECT_ROOT / "src/few/data/KerrEccEqFluxData.h5",
    PROJECT_ROOT / "src/few/data/ZNAmps_l10_m10_n55_DS2Outer.h5",
    PROJECT_ROOT / "src/few/data/LPA.txt",
)

CASES = (
    {
        "key": "baseline_short",
        "a": 0.7,
        "p0": 11.0,
        "e0": 0.4,
        "xI0": 1.0,
        "T": 0.001,
    },
    {
        "key": "baseline_one_year",
        "a": 0.7,
        "p0": 11.0,
        "e0": 0.4,
        "xI0": 1.0,
        "T": 1.0,
    },
    {
        "key": "positive_spin_retrograde",
        "a": 0.7,
        "p0": 11.0,
        "e0": 0.4,
        "xI0": -1.0,
        "T": 0.01,
    },
    {
        "key": "inner_orbit",
        "a": 0.6,
        "p0": 8.0,
        "e0": 0.3,
        "xI0": 1.0,
        "T": 0.01,
    },
    {
        "key": "zero_spin",
        "a": 0.0,
        "p0": 11.0,
        "e0": 0.4,
        "xI0": 1.0,
        "T": 0.01,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required strict-Metal input is missing: {path}")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def waveform_arguments(case: dict[str, Any]) -> tuple[float, ...]:
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
    kwargs = dict(BASE_KWARGS)
    kwargs["T"] = case["T"]
    kwargs["dt"] = 15.0
    return kwargs


def timed_waveform(
    generator: Any, args: tuple[float, ...], kwargs: dict[str, float]
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    waveform = np.ascontiguousarray(
        np.asarray(generator(*args, **kwargs), dtype=np.complex128)
    )
    return waveform, time.perf_counter() - start


def exact_inputs(case: dict[str, Any]) -> dict[str, float]:
    return {
        "M": WAVEFORM_ARGS[0],
        "mu": WAVEFORM_ARGS[1],
        "a": case["a"],
        "p0": case["p0"],
        "e0": case["e0"],
        "xI0": case["xI0"],
        "theta": WAVEFORM_ARGS[6],
        "phi": WAVEFORM_ARGS[7],
        "dist": BASE_KWARGS["dist"],
        "Phi_phi0": BASE_KWARGS["Phi_phi0"],
        "Phi_theta0": BASE_KWARGS["Phi_theta0"],
        "Phi_r0": BASE_KWARGS["Phi_r0"],
        "T": case["T"],
        "dt": 15.0,
    }


def run_case(
    generator: Any,
    interpolation_context: MetalContext,
    sum_engine: MetalSummation,
    case: dict[str, Any],
    repetitions: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    args = waveform_arguments(case)
    kwargs = waveform_kwargs(case)

    cpu_first, cpu_first_seconds = timed_waveform(generator, args, kwargs)
    cpu_warm_outputs = []
    cpu_warm_seconds = []
    for _ in range(repetitions):
        output, seconds = timed_waveform(generator, args, kwargs)
        cpu_warm_outputs.append(output)
        cpu_warm_seconds.append(seconds)
    cpu_reference = cpu_warm_outputs[-1]
    cpu_repeatable = np.array_equal(cpu_first, cpu_reference) and all(
        np.array_equal(cpu_warm_outputs[0], output) for output in cpu_warm_outputs[1:]
    )

    plan_start = time.perf_counter()
    replacements = install_amplitude_holders(
        generator, interpolation_context, waveform_args=args
    )
    plan_creation_seconds = time.perf_counter() - plan_start
    backend = get_backend("cpu")
    original_sum = backend.get_waveform_wrap
    sum_call_start = len(sum_engine.calls)
    metal_outputs: list[np.ndarray] = []
    metal_first_seconds = 0.0
    metal_warm_seconds: list[float] = []
    try:
        backend.get_waveform_wrap = sum_engine
        metal_first, metal_first_seconds = timed_waveform(generator, args, kwargs)
        metal_outputs.append(metal_first)
        for _ in range(repetitions):
            output, seconds = timed_waveform(generator, args, kwargs)
            metal_outputs.append(output)
            metal_warm_seconds.append(seconds)
    finally:
        backend.get_waveform_wrap = original_sum
        for holders, index, original, _replacement in replacements:
            holders[index] = original

    cpu_after, cpu_after_seconds = timed_waveform(generator, args, kwargs)
    holder_reports = []
    for _holders, index, _original, replacement in replacements:
        holder_reports.append(
            {
                "z_index": index,
                "plan_upload_seconds": replacement.plan.upload_seconds,
                "fallback_calls": replacement.fallback_calls,
                "calls": replacement.calls,
            }
        )
        replacement.close()

    metal_reference = metal_outputs[-1]
    metal_repeatable = all(
        np.array_equal(metal_outputs[0], output) for output in metal_outputs[1:]
    )
    finite = bool(
        np.all(np.isfinite(metal_reference.real))
        and np.all(np.isfinite(metal_reference.imag))
    )
    metrics = error_metrics(cpu_reference, metal_reference)
    metrics["flat_mismatch"] = float(
        max(0.0, get_mismatch(cpu_reference, metal_reference))
    )
    report = {
        "key": case["key"],
        "inputs": exact_inputs(case),
        "dtype": str(metal_reference.dtype),
        "shape": list(metal_reference.shape),
        "modes_kept": int(generator.num_modes_kept),
        "array_sha256": sha256_array(metal_reference),
        "finite": finite,
        "cpu": {
            "first_seconds": cpu_first_seconds,
            "warm_seconds": cpu_warm_seconds,
            "warm_median_seconds": statistics.median(cpu_warm_seconds),
            "repeatable": cpu_repeatable,
            "after_seconds": cpu_after_seconds,
            "before_after_bitwise": bool(np.array_equal(cpu_reference, cpu_after)),
        },
        "metal": {
            "amplitude_plan_creation_seconds": plan_creation_seconds,
            "amplitude_holders": holder_reports,
            "first_seconds": metal_first_seconds,
            "warm_seconds": metal_warm_seconds,
            "warm_median_seconds": statistics.median(metal_warm_seconds),
            "repeatable": metal_repeatable,
            "sum_calls": sum_engine.calls[sum_call_start:],
        },
        "metal_vs_cpu": metrics,
        "warm_end_to_end_speedup": statistics.median(cpu_warm_seconds)
        / statistics.median(metal_warm_seconds),
        "peak_process_rss_mib_after_case": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        / (1024 * 1024),
    }
    if not finite or not metal_repeatable or not report["cpu"]["before_after_bitwise"]:
        raise RuntimeError(f"Strict-Metal integrity check failed for {case['key']}")
    return metal_reference, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Strict-Metal reference generation requires Apple Silicon")
    if args.repetitions < 2:
        raise ValueError("At least two warm repetitions are required")

    data_files = {path.name: file_metadata(path) for path in DATA_PATHS}
    model_load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend="cpu",
    )
    model_load_seconds = time.perf_counter() - model_load_start

    interpolation_context = MetalContext(load_interp_api(build_interp_library()))
    sum_engine = MetalSummation(load_sum_api(build_sum_library("ds")))
    outputs: dict[str, np.ndarray] = {}
    case_reports = []
    try:
        for case in CASES:
            output, case_report = run_case(
                generator,
                interpolation_context,
                sum_engine,
                case,
                args.repetitions,
            )
            outputs[case["key"]] = output
            case_reports.append(case_report)
        interpolation_metadata = interpolation_context.metadata()
    finally:
        interpolation_context.close()
        sum_engine.close()

    clang_version = subprocess.run(
        ["clang++", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    repository = {
        # 2026-09-01 23:56 CST (mac): Record the synchronized base explicitly;
        # the installed FEW distribution version can retain older local-build
        # metadata even though this generator imports the current source tree.
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
    }
    source_files = {
        name: file_metadata(Path(__file__).with_name(name))
        for name in (
            "benchmark_metal_hybrid.py",
            "benchmark_metal_interp.py",
            "benchmark_metal_sum.py",
            "benchmark_metal_waveform.py",
            "generate_strict_metal_reference.py",
            "metal_interp.mm",
            "metal_sum_ds.mm",
        )
    }
    artifact_metadata = {
        "schema": SCHEMA,
        "seed": SEED,
        "collaboration_note": (
            "2026-09-01 23:54 CST (mac): unrounded complex128 strict-Metal "
            "waveforms requested by Ubuntu handoff 0120e06c"
        ),
        "repository": repository,
        "data_files": data_files,
        "source_files": source_files,
        "cases": [
            {
                "key": case_report["key"],
                "inputs": case_report["inputs"],
                "dtype": case_report["dtype"],
                "shape": case_report["shape"],
                "modes_kept": case_report["modes_kept"],
                "array_sha256": case_report["array_sha256"],
            }
            for case_report in case_reports
        ],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        metadata_json=np.asarray(json.dumps(artifact_metadata, sort_keys=True)),
        **outputs,
    )
    artifact = {
        "path": str(args.artifact.relative_to(PROJECT_ROOT)),
        "bytes": args.artifact.stat().st_size,
        "sha256": sha256_file(args.artifact),
    }
    report = {
        "schema": SCHEMA,
        "seed": SEED,
        "collaboration_note": (
            "2026-09-01 23:54 CST (mac): structured strict-Metal report for "
            "Ubuntu CPU/CUDA and LISA-weighted comparison"
        ),
        "artifact": artifact,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "few": few.__version__,
            "numpy": np.__version__,
            "clang": clang_version,
        },
        "repository": repository,
        "metal": {
            "device": interpolation_metadata,
            "sum_pipeline_compile_seconds": sum_engine.compile_seconds,
            "math_mode": "MTLMathModeSafe",
            "floating_point_functions": "MTLMathFloatingPointFunctionsPrecise",
            "storage_mode": "MTLResourceStorageModeShared",
            "precision": "full-chain-double-single-high-low-fp32",
        },
        "model_load_seconds": model_load_seconds,
        "data_files": data_files,
        "source_files": source_files,
        "cases": case_reports,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
