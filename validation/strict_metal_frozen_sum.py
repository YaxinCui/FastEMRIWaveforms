#!/usr/bin/env python3
"""Replay Mac-prepared strict-Metal summation inputs on FEW CPU or CUDA.

2026-09-02 10:55 CST (mac): Add an integrity-bound kernel-only validator. It
does not regenerate trajectories or amplitudes, so the existing 5e-10
elementwise gate is binding across hosts as well as across Linux CPU/CUDA.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from strict_metal_cross_host import (
    comparison_metrics,
    load_lpa_asd,
    sha256_array,
    sha256_file,
    synchronize,
    to_host,
)

import few
from few import get_backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = PROJECT_ROOT / "collaboration/mac/strict_metal_frozen_sum_inputs.npz"
DEFAULT_INPUT_REPORT = (
    PROJECT_ROOT / "collaboration/mac/strict_metal_frozen_sum_report.json"
)
DEFAULT_METAL_REFERENCE = (
    PROJECT_ROOT / "collaboration/mac/strict_metal_ds_reference.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "collaboration/mac/strict_metal_frozen_sum_cpu.json"
# 2026-09-02 13:52 CST (mac): Give the production Metal backend its own Mac-
# owned evidence file so it cannot overwrite the accepted CPU replay.
DEFAULT_METAL_OUTPUT = (
    PROJECT_ROOT / "collaboration/mac/strict_metal_production_backend.json"
)
LPA_PATH = PROJECT_ROOT / "src/few/data/LPA.txt"
# 2026-09-02 11:00 CST (mac): Pin the reviewed capture and report identities so
# Ubuntu rejects partial transfers, stale regeneration, or metadata drift.
EXPECTED_FROZEN_INPUTS = {
    "bytes": 195_212,
    "sha256": "abbe058932078bda38fc4404aab1c49e6b54434f02640145b4ed9465d4cac1db",
}
EXPECTED_FROZEN_REPORT = {
    "bytes": 28_533,
    "sha256": "6064153fd5b98d1b35be2f086a6662956ec6e7d55a80b8464a9f0e3b87d3b989",
}
EXPECTED_METAL_REFERENCE = {
    "bytes": 33_254_256,
    "sha256": "42bda4811e25f94797048b7168ca55e69a26d74ca8c388374052dc42922a1851",
}
ARRAY_NAMES = (
    "interpolation",
    "phase_times",
    "phase_coefficients",
    "m_values",
    "k_values",
    "n_values",
    "ylms",
    "trajectory_times",
)
HOST_DEREFERENCED_ARRAY_NAMES = ("phase_times", "trajectory_times")


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_identity(
    label: str, actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    failures = [
        f"{field}: expected {expected[field]!r}, found {actual[field]!r}"
        for field in ("bytes", "sha256")
        if actual[field] != expected[field]
    ]
    if failures:
        raise RuntimeError(f"{label} identity failed: " + "; ".join(failures))


def load_inputs(
    artifact_path: Path, report_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    artifact_identity = file_identity(artifact_path)
    report_identity = file_identity(report_path)
    verify_identity("frozen summation input", artifact_identity, EXPECTED_FROZEN_INPUTS)
    verify_identity("frozen summation report", report_identity, EXPECTED_FROZEN_REPORT)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["artifact"] != artifact_identity:
        raise RuntimeError("Frozen input artifact differs from its Mac report")

    cases: dict[str, dict[str, np.ndarray]] = {}
    with np.load(artifact_path, allow_pickle=False) as artifact:
        embedded = json.loads(str(artifact["metadata_json"]))
        expected_keys = {"metadata_json"}
        for case in embedded["cases"]:
            expected_keys.update(
                case["arrays"][name]["artifact_key"] for name in ARRAY_NAMES
            )
        if set(artifact.files) != expected_keys:
            raise RuntimeError("Frozen input artifact keys differ from its metadata")
        for case in embedded["cases"]:
            case_arrays = {}
            for name in ARRAY_NAMES:
                metadata = case["arrays"][name]
                value = np.ascontiguousarray(artifact[metadata["artifact_key"]])
                checks = (
                    list(value.shape) == metadata["shape"],
                    str(value.dtype) == metadata["dtype"],
                    sha256_array(value) == metadata["array_sha256"],
                    np.all(np.isfinite(value)),
                )
                if not all(checks):
                    raise RuntimeError(
                        f"Frozen array integrity failed: {case['key']}/{name}"
                    )
                case_arrays[name] = value
            cases[case["key"]] = case_arrays

    report_cases = [
        {
            "key": case["key"],
            "inputs": case["inputs"],
            "arrays": case["arrays"],
            "scalars": case["scalars"],
            "postprocessing": case["postprocessing"],
            "strict_metal_output": case["strict_metal_output"],
        }
        for case in report["cases"]
    ]
    metadata_checks = {
        "schema": embedded["schema"] == report["schema"],
        "seed": embedded["seed"] == report["seed"],
        "repository": embedded["repository"] == report["repository"],
        "reference_files": embedded["reference_files"] == report["reference_files"],
        "data_files": embedded["data_files"] == report["data_files"],
        "source_files": embedded["source_files"] == report["source_files"],
        "cases": embedded["cases"] == report_cases,
    }
    if not all(metadata_checks.values()):
        raise RuntimeError(
            f"Frozen artifact/report metadata mismatch: {metadata_checks}"
        )
    return (
        embedded,
        cases,
        {
            "artifact": artifact_identity,
            "report": report_identity,
        },
    )


def load_metal_outputs(
    path: Path, embedded: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    identity = file_identity(path)
    verify_identity(
        "strict-Metal waveform reference", identity, EXPECTED_METAL_REFERENCE
    )
    outputs = {}
    with np.load(path, allow_pickle=False) as reference:
        for case in embedded["cases"]:
            key = case["key"]
            output = np.ascontiguousarray(reference[key])
            expected = case["strict_metal_output"]
            if (
                list(output.shape) != expected["shape"]
                or str(output.dtype) != expected["dtype"]
                or sha256_array(output) != expected["array_sha256"]
            ):
                raise RuntimeError(f"Strict-Metal output integrity failed: {key}")
            outputs[key] = output
    return outputs, identity


def replay(
    backend: Any,
    arrays: dict[str, np.ndarray],
    scalars: dict[str, Any],
    repetitions: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    xp = backend.xp
    transfer_start = time.perf_counter()
    # 2026-09-02 13:30 CST (linux): Match InterpolatedModeSum's mixed CUDA
    # pointer ABI. get_waveform dereferences both knot-time arrays in host C++
    # before launching device kernels; copying them to CuPy caused the first
    # frozen CUDA replay to segfault. The remaining arrays are kernel inputs.
    prepared = {
        name: (
            np.ascontiguousarray(arrays[name])
            if backend.uses_cuda and name in HOST_DEREFERENCED_ARRAY_NAMES
            else xp.asarray(arrays[name])
        )
        for name in ARRAY_NAMES
    }
    synchronize(backend)
    transfer_seconds = time.perf_counter() - transfer_start
    outputs = []
    timings = []
    device = int(xp.cuda.runtime.getDevice()) if backend.uses_cupy else 0
    for _ in range(repetitions):
        waveform = xp.zeros(scalars["output_length"], dtype=xp.complex128)
        synchronize(backend)
        start = time.perf_counter()
        backend.get_waveform_wrap(
            waveform,
            prepared["interpolation"],
            prepared["phase_times"],
            prepared["phase_coefficients"],
            prepared["m_values"],
            prepared["k_values"],
            prepared["n_values"],
            scalars["init_length"],
            scalars["output_length"],
            scalars["mode_count"],
            prepared["ylms"],
            scalars["delta_t"],
            prepared["trajectory_times"],
            device,
        )
        synchronize(backend)
        timings.append(time.perf_counter() - start)
        outputs.append(np.ascontiguousarray(to_host(waveform), dtype=np.complex128))
    repeatable = all(np.array_equal(outputs[0], output) for output in outputs[1:])
    if not repeatable:
        raise RuntimeError("Frozen summation replay is not bitwise repeatable")
    return outputs[-1], {
        "input_transfer_seconds": transfer_seconds,
        "kernel_seconds": timings,
        "bitwise_repeatable": repeatable,
        "array_sha256": sha256_array(outputs[-1]),
    }


def memory_snapshot(backend: Any) -> dict[str, float]:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    snapshot = {"peak_process_rss_mib": float(peak_rss / divisor)}
    if backend.uses_cupy:
        pool = backend.xp.get_default_memory_pool()
        snapshot.update(
            {
                "cupy_pool_used_mib": float(pool.used_bytes() / (1024 * 1024)),
                "cupy_pool_total_mib": float(pool.total_bytes() / (1024 * 1024)),
            }
        )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    # 2026-09-02 13:52 CST (mac): Reuse the integrity-bound frozen contract to
    # accept the installed Metal backend, not only the earlier isolated PoC.
    parser.add_argument("--backend", choices=("cpu", "cuda12x", "metal"), default="cpu")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--metal-reference", type=Path, default=DEFAULT_METAL_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("At least two kernel repetitions are required")
    output_path = (
        DEFAULT_METAL_OUTPUT
        if args.output is None and args.backend == "metal"
        else DEFAULT_OUTPUT
        if args.output is None
        else args.output
    )

    embedded, frozen_cases, frozen_identities = load_inputs(
        args.inputs, args.input_report
    )
    metal_outputs, metal_identity = load_metal_outputs(args.metal_reference, embedded)
    backend = get_backend(args.backend)
    lpa_frequency, lpa_asd = load_lpa_asd(LPA_PATH)
    case_reports = []
    for case in embedded["cases"]:
        key = case["key"]
        output, replay_report = replay(
            backend, frozen_cases[key], case["scalars"], args.repetitions
        )
        output = output / case["postprocessing"]["waveform_output_divisor"]
        replay_report["physical_array_sha256"] = sha256_array(output)
        metrics = comparison_metrics(
            output,
            metal_outputs[key],
            case["scalars"]["delta_t"],
            lpa_frequency,
            lpa_asd,
            enforce_elementwise=True,
        )
        metrics["local_elementwise_gate"]["reason"] = (
            "binding kernel-only gate: both paths consume the identical "
            "Mac-captured amplitude spline, phase spline, modes, and Ylms"
        )
        case_reports.append(
            {
                "key": key,
                "scalars": case["scalars"],
                "strict_metal_array_sha256": case["strict_metal_output"][
                    "array_sha256"
                ],
                "replay": replay_report,
                "metrics": metrics,
                "passed": metrics["passed"],
            }
        )

    report = {
        "schema": 1,
        "collaboration_note": (
            "2026-09-02 10:55 CST (mac): kernel-only replay of frozen strict-"
            "Metal summation inputs; each host writes results under its own directory"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "few": few.__version__,
            "numpy": np.__version__,
            "backend": args.backend,
        },
        "inputs": frozen_identities,
        "metal_reference": metal_identity,
        "cases": case_reports,
        "memory": memory_snapshot(backend),
        "passed": all(case["passed"] for case in case_reports),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
