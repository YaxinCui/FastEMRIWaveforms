#!/usr/bin/env python3
"""Freeze the exact prepared inputs consumed by the strict Metal mode sum.

2026-09-02 10:55 CST (mac): Add a deterministic capture around FEW's 14-
argument summation ABI. The wrapper delegates to the isolated strict Metal
kernel, verifies repeated captures and outputs bitwise, and writes only the
prepared inputs needed for a kernel-only Mac/Linux comparison.
"""

from __future__ import annotations

import argparse
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
from benchmark_metal_interp import MetalContext
from benchmark_metal_interp import build_library as build_interp_library
from benchmark_metal_interp import load_api as load_interp_api
from benchmark_metal_sum import MetalSummation
from benchmark_metal_sum import build_library as build_sum_library
from benchmark_metal_sum import load_api as load_sum_api
from generate_strict_metal_reference import (
    CASES,
    DATA_PATHS,
    PROJECT_ROOT,
    exact_inputs,
    file_metadata,
    sha256_array,
    waveform_arguments,
    waveform_kwargs,
)

import few
from few import get_backend
from few.utils.constants import MRSUN_SI, Gpc
from few.waveform import FastKerrEccentricEquatorialFlux

SCHEMA = 1
SEED = 20260902
DEFAULT_ARTIFACT = PROJECT_ROOT / "collaboration/mac/strict_metal_frozen_sum_inputs.npz"
DEFAULT_REPORT = PROJECT_ROOT / "collaboration/mac/strict_metal_frozen_sum_report.json"
STRICT_METAL_ARTIFACT = PROJECT_ROOT / "collaboration/mac/strict_metal_ds_reference.npz"
STRICT_METAL_REPORT = PROJECT_ROOT / "collaboration/mac/strict_metal_ds_report.json"
ARRAY_ARGUMENTS = (
    ("interpolation", 1, np.float64),
    ("phase_times", 2, np.float64),
    ("phase_coefficients", 3, np.float64),
    ("m_values", 4, np.int32),
    ("k_values", 5, np.int32),
    ("n_values", 6, np.int32),
    ("ylms", 10, np.complex128),
    ("trajectory_times", 12, np.float64),
)


def capture_array(value: Any, dtype: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()


class CapturingSummation:
    """Copy effective kernel inputs before delegating to strict Metal."""

    def __init__(self, delegate: MetalSummation):
        self.delegate = delegate
        self.captures: list[dict[str, Any]] = []

    def __call__(self, *args: Any) -> None:
        if len(args) != 14:
            raise RuntimeError(f"Expected 14 summation arguments, received {len(args)}")
        capture = {
            name: capture_array(args[index], dtype)
            for name, index, dtype in ARRAY_ARGUMENTS
        }
        capture["init_length"] = int(args[7])
        capture["output_length"] = int(args[8])
        capture["mode_count"] = int(args[9])
        capture["delta_t"] = float(args[11])
        capture["device"] = int(args[13])
        self.delegate(*args)
        capture["kernel_output"] = capture_array(args[0], np.complex128)
        self.captures.append(capture)


def captures_are_bitwise_equal(captures: list[dict[str, Any]]) -> bool:
    if len(captures) < 2:
        return False
    first = captures[0]
    for capture in captures[1:]:
        for name, _index, _dtype in ARRAY_ARGUMENTS:
            if not np.array_equal(first[name], capture[name]):
                return False
        for name in (
            "init_length",
            "output_length",
            "mode_count",
            "delta_t",
            "device",
        ):
            if first[name] != capture[name]:
                return False
    return True


def load_reference_report() -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(STRICT_METAL_REPORT.read_text(encoding="utf-8"))
    artifact = file_metadata(STRICT_METAL_ARTIFACT)
    report_identity = file_metadata(STRICT_METAL_REPORT)
    if artifact != report["artifact"]:
        raise RuntimeError(
            "Existing strict-Metal artifact does not match its provenance report"
        )
    return report, {"artifact": artifact, "report": report_identity}


def run_case(
    generator: Any,
    interpolation_context: MetalContext,
    capturing_sum: CapturingSummation,
    case: dict[str, Any],
    repetitions: int,
    reference_case: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    waveform_args = waveform_arguments(case)
    kwargs = waveform_kwargs(case)
    backend = get_backend("cpu")
    original_sum = backend.get_waveform_wrap
    replacements = install_amplitude_holders(
        generator, interpolation_context, waveform_args=waveform_args
    )
    capture_start = len(capturing_sum.captures)
    outputs = []
    timings = []
    try:
        backend.get_waveform_wrap = capturing_sum
        for _ in range(repetitions):
            start = time.perf_counter()
            output = np.ascontiguousarray(
                np.asarray(generator(*waveform_args, **kwargs), dtype=np.complex128)
            )
            timings.append(time.perf_counter() - start)
            outputs.append(output.copy())
    finally:
        backend.get_waveform_wrap = original_sum
        for holders, index, original, _replacement in replacements:
            holders[index] = original

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

    captures = capturing_sum.captures[capture_start:]
    inputs_repeatable = captures_are_bitwise_equal(captures)
    output_repeatable = all(
        np.array_equal(outputs[0], output) for output in outputs[1:]
    )
    output_sha256 = sha256_array(outputs[-1])
    output_matches_reference = output_sha256 == reference_case["array_sha256"]
    raw_outputs_repeatable = all(
        np.array_equal(captures[0]["kernel_output"], capture["kernel_output"])
        for capture in captures[1:]
    )
    exact = exact_inputs(case)
    reduced_mass = exact["M"] * exact["mu"] / (exact["M"] + exact["mu"])
    waveform_output_divisor = exact["dist"] * Gpc / (reduced_mass * MRSUN_SI)
    scaled_kernel_output = captures[-1]["kernel_output"] / waveform_output_divisor
    scaled_kernel_matches_generator = bool(
        np.array_equal(scaled_kernel_output, outputs[-1])
    )
    if (
        not inputs_repeatable
        or not output_repeatable
        or not raw_outputs_repeatable
        or not scaled_kernel_matches_generator
        or not output_matches_reference
    ):
        raise RuntimeError(
            f"Frozen summation integrity failed for {case['key']}: "
            f"inputs_repeatable={inputs_repeatable}, "
            f"output_repeatable={output_repeatable}, "
            f"raw_outputs_repeatable={raw_outputs_repeatable}, "
            f"scaled_kernel_matches_generator={scaled_kernel_matches_generator}, "
            f"output_matches_reference={output_matches_reference}"
        )

    frozen = captures[-1]
    arrays = {name: frozen[name] for name, _index, _dtype in ARRAY_ARGUMENTS}
    array_metadata = {
        name: {
            "artifact_key": f"{case['key']}__{name}",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "array_sha256": sha256_array(value),
        }
        for name, value in arrays.items()
    }
    case_report = {
        "key": case["key"],
        "inputs": exact_inputs(case),
        "arrays": array_metadata,
        "scalars": {
            "init_length": frozen["init_length"],
            "output_length": frozen["output_length"],
            "mode_count": frozen["mode_count"],
            "delta_t": frozen["delta_t"],
            "captured_device": frozen["device"],
        },
        "postprocessing": {
            "operation": "kernel_output / waveform_output_divisor",
            "waveform_output_divisor": waveform_output_divisor,
            "raw_kernel_array_sha256": sha256_array(frozen["kernel_output"]),
            "scaled_kernel_matches_generator_bitwise": (
                scaled_kernel_matches_generator
            ),
        },
        "capture_repetitions": repetitions,
        "inputs_bitwise_repeatable": inputs_repeatable,
        "raw_kernel_outputs_bitwise_repeatable": raw_outputs_repeatable,
        "metal_outputs_bitwise_repeatable": output_repeatable,
        "strict_metal_output": {
            "dtype": str(outputs[-1].dtype),
            "shape": list(outputs[-1].shape),
            "array_sha256": output_sha256,
            "matches_existing_reference": output_matches_reference,
        },
        "waveform_seconds": timings,
        "waveform_median_seconds": statistics.median(timings),
        "amplitude_holders": holder_reports,
    }
    return arrays, case_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Frozen strict-Metal capture requires Apple Silicon")
    if args.repetitions < 2:
        raise ValueError("At least two captures are required")

    reference_report, reference_files = load_reference_report()
    reference_cases = {case["key"]: case for case in reference_report["cases"]}
    data_files = {path.name: file_metadata(path) for path in DATA_PATHS}
    if data_files != reference_report["data_files"]:
        raise RuntimeError("Current FEW data files differ from strict-Metal reference")

    load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend="cpu",
    )
    model_load_seconds = time.perf_counter() - load_start
    interpolation_context = MetalContext(load_interp_api(build_interp_library()))
    sum_engine = MetalSummation(load_sum_api(build_sum_library("ds")))
    capturing_sum = CapturingSummation(sum_engine)

    artifact_arrays: dict[str, np.ndarray] = {}
    case_reports = []
    try:
        for case in CASES:
            arrays, case_report = run_case(
                generator,
                interpolation_context,
                capturing_sum,
                case,
                args.repetitions,
                reference_cases[case["key"]],
            )
            for name, value in arrays.items():
                artifact_arrays[f"{case['key']}__{name}"] = value
            case_reports.append(case_report)
        metal_device = interpolation_context.metadata()
    finally:
        interpolation_context.close()
        sum_engine.close()

    repository = {
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
            "generate_strict_metal_frozen_sum.py",
            "generate_strict_metal_reference.py",
            "metal_interp.mm",
            "metal_sum_ds.mm",
        )
    }
    embedded_cases = [
        {
            "key": case["key"],
            "inputs": case["inputs"],
            "arrays": case["arrays"],
            "scalars": case["scalars"],
            "postprocessing": case["postprocessing"],
            "strict_metal_output": case["strict_metal_output"],
        }
        for case in case_reports
    ]
    artifact_metadata = {
        "schema": SCHEMA,
        "seed": SEED,
        "collaboration_note": (
            "2026-09-02 10:55 CST (mac): exact prepared inputs observed at "
            "the strict Metal 14-argument summation ABI"
        ),
        "repository": repository,
        "reference_files": reference_files,
        "data_files": data_files,
        "source_files": source_files,
        "cases": embedded_cases,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.artifact,
        metadata_json=np.asarray(json.dumps(artifact_metadata, sort_keys=True)),
        **artifact_arrays,
    )
    artifact = file_metadata(args.artifact)
    report = {
        **artifact_metadata,
        "collaboration_note": (
            "2026-09-02 10:55 CST (mac): frozen strict-Metal summation inputs "
            "for kernel-only Mac/Linux CPU/CUDA validation"
        ),
        "artifact": artifact,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "few": few.__version__,
            "numpy": np.__version__,
        },
        "metal": {
            "device": metal_device,
            "sum_pipeline_compile_seconds": sum_engine.compile_seconds,
            "precision": "full-chain-double-single-high-low-fp32",
        },
        "model_load_seconds": model_load_seconds,
        "cases": case_reports,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
