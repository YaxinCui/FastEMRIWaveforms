#!/usr/bin/env python3
"""Probe reversible CUDA execution/precision candidates against FEW FP64.

This is a Linux-owned research utility, not a registered FEW backend.

2026-09-03 21:48 CST (linux): Add the first outcome-driven CUDA probe. It
compares the existing FP64 ROMAN native-wrapper path with a same-FP64 CuPy
matrix path so scheduling/handle overhead is measured before reducing numeric
precision. The accepted backend and production source remain unchanged.
2026-09-03 22:13 CST (linux): Exercise the implemented opt-in
cuda_roman_mode="cupy_fp64" path directly after its fresh CUDA wheel build,
rather than duplicating that candidate implementation inside the probe.
2026-09-03 22:18 CST (linux): Add warm end-to-end Schwarzschild waveform cases
so a microkernel improvement alone cannot qualify the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np

import few
from few import get_backend
from few.amplitude.romannet import RomanAmplitude
from few.utils.utility import get_mismatch
from few.waveform import FastSchwarzschildEccentricFlux

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZES = (128, 1000, 4096)
DEFAULT_REPETITIONS = 16
DEFAULT_WAVEFORM_REPETITIONS = 8
ROMAN_NORMALIZED_LIMIT = 5.0e-12
SCHWARZSCHILD_NORMALIZED_LIMIT = 5.0e-11
WAVEFORM_MISMATCH_LIMIT = 1.0e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def synchronize(xp: Any) -> None:
    xp.cuda.Stream.null.synchronize()


def timing_summary(
    xp: Any, function: Callable[[], Any], repetitions: int
) -> tuple[Any, dict[str, float]]:
    function()
    synchronize(xp)
    samples = []
    output = None
    for _ in range(repetitions):
        start = time.perf_counter()
        output = function()
        synchronize(xp)
        samples.append(time.perf_counter() - start)
    values = np.asarray(samples)
    return output, {
        "median_ms": float(np.median(values) * 1.0e3),
        "minimum_ms": float(np.min(values) * 1.0e3),
        "p90_ms": float(np.quantile(values, 0.9) * 1.0e3),
    }


def comparison_metrics(
    xp: Any, actual: Any, reference: Any, normalized_limit: float
) -> dict[str, Any]:
    delta = actual - reference
    tiny = np.finfo(float).tiny
    scale = max(float(xp.max(xp.abs(reference)).get()), tiny)
    norm = max(float(xp.linalg.norm(reference.ravel()).get()), tiny)
    normalized_max = float(xp.max(xp.abs(delta)).get()) / scale
    relative_l2 = float(xp.linalg.norm(delta.ravel()).get()) / norm
    return {
        "shape": list(actual.shape),
        "normalized_max_abs": normalized_max,
        "relative_l2": relative_l2,
        "limit": normalized_limit,
        "passed": normalized_max <= normalized_limit
        and relative_l2 <= normalized_limit,
    }


def run_probe(
    sizes: tuple[int, ...], repetitions: int, waveform_repetitions: int
) -> dict[str, Any]:
    backend = get_backend("cuda12x")
    if not backend.uses_cupy:
        raise RuntimeError("cuda12x did not expose the required CuPy array module")
    xp = backend.xp
    max_size = max(sizes)
    native_roman = RomanAmplitude(buffer_length=max_size, force_backend="cuda12x")
    candidate_roman = RomanAmplitude(
        buffer_length=max_size,
        force_backend="cuda12x",
        cuda_roman_mode="cupy_fp64",
    )
    zero_spin = xp.zeros(1, dtype=xp.float64)
    equatorial = xp.ones(1, dtype=xp.float64)

    cases = []
    for size in sizes:
        p = xp.asarray(np.linspace(10.0, 14.0, size))
        e = xp.asarray(np.linspace(0.1, 0.6, size))

        native, native_timing = timing_summary(
            xp,
            lambda: native_roman.get_amplitudes(
                zero_spin, p, e, equatorial, renormalize_amps=False
            ),
            repetitions,
        )
        candidate, candidate_timing = timing_summary(
            xp,
            lambda: candidate_roman.get_amplitudes(
                zero_spin, p, e, equatorial, renormalize_amps=False
            ),
            repetitions,
        )
        candidate_repeat = candidate_roman.get_amplitudes(
            zero_spin, p, e, equatorial, renormalize_amps=False
        )
        synchronize(xp)
        metrics = comparison_metrics(
            xp, candidate, native, normalized_limit=ROMAN_NORMALIZED_LIMIT
        )
        cases.append(
            {
                "input_points": size,
                "native_fp64": native_timing,
                "cupy_fp64_candidate": candidate_timing,
                "median_speedup": native_timing["median_ms"]
                / candidate_timing["median_ms"],
                "candidate_repeat_bitwise": bool(
                    xp.array_equal(candidate, candidate_repeat)
                ),
                "candidate_vs_native": metrics,
            }
        )

    # The waveform models are built once and timed only after warm-up. This
    # includes trajectory, amplitude normalization/selection, and summation,
    # rather than reporting the ROMAN matrix stage as an end-to-end speedup.
    native_waveform = FastSchwarzschildEccentricFlux(force_backend="cuda12x")
    candidate_waveform = FastSchwarzschildEccentricFlux(
        amplitude_kwargs={"cuda_roman_mode": "cupy_fp64"},
        force_backend="cuda12x",
    )
    waveform_args = (1.0e6, 1.0e1, 8.0, 0.2, np.pi / 3, np.pi / 4)
    waveform_cases = []
    for duration_years in (0.001, 0.01, 1.0):
        kwargs = {"dist": 1.0, "T": duration_years, "dt": 15.0}
        native, native_timing = timing_summary(
            xp,
            lambda: native_waveform(*waveform_args, **kwargs),
            waveform_repetitions,
        )
        candidate, candidate_timing = timing_summary(
            xp,
            lambda: candidate_waveform(*waveform_args, **kwargs),
            waveform_repetitions,
        )
        metrics = comparison_metrics(
            xp,
            candidate,
            native,
            normalized_limit=SCHWARZSCHILD_NORMALIZED_LIMIT,
        )
        mismatch = float(
            max(
                0.0,
                get_mismatch(xp.asnumpy(candidate), xp.asnumpy(native), use_gpu=False),
            )
        )
        metrics["flat_weight_mismatch"] = mismatch
        metrics["mismatch_limit"] = WAVEFORM_MISMATCH_LIMIT
        metrics["passed"] = metrics["passed"] and mismatch <= WAVEFORM_MISMATCH_LIMIT
        waveform_cases.append(
            {
                "duration_years": duration_years,
                "samples": int(native.size),
                "native_fp64": native_timing,
                "cupy_fp64_candidate": candidate_timing,
                "median_speedup": native_timing["median_ms"]
                / candidate_timing["median_ms"],
                "candidate_vs_native": metrics,
            }
        )

    properties = xp.cuda.runtime.getDeviceProperties(0)
    data_path = (
        PROJECT_ROOT / "src" / "few" / "data" / "SchwarzschildEccentricInput.hdf5"
    )
    source_paths = (
        PROJECT_ROOT / "src" / "few" / "amplitude" / "romannet.py",
        PROJECT_ROOT / "src" / "few" / "cutils" / "matmul.cu",
        data_path,
    )
    pool = xp.get_default_memory_pool()
    return {
        "schema": 1,
        "collaboration_note": (
            "2026-09-03 CST (linux): Linux-only exploratory report; no "
            "production backend or acceptance threshold is changed."
        ),
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "project_commit": current_commit(),
        "few_version": few.__version__,
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
        },
        "experiment": {
            "reference": "existing FEW native ROMAN path",
            "reference_storage_compute_accumulation": "FP64/FP64/FP64",
            "candidate": "CuPy ROMAN matrix path",
            "candidate_storage_compute_accumulation": "FP64/FP64/FP64",
            "renormalization_included": False,
            "repetitions": repetitions,
            "waveform_repetitions": waveform_repetitions,
            "timing_scope": "warm synchronized wall time",
            "accuracy_limit_source": "validation/dual_host_consistency.py",
        },
        "network": {
            "layer_shapes": [
                [int(dim1), int(dim2)]
                for dim1, dim2 in zip(native_roman.dim1, native_roman.dim2)
            ],
            "break_index": int(native_roman.break_index),
            "num_teuk_modes": int(native_roman.num_teuk_modes),
            "weight_dtypes": [str(weight.dtype) for weight in native_roman.weights],
            "transform_dtype": str(native_roman.transform_matrix.dtype),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in source_paths
        },
        "cases": cases,
        "waveform_cases": waveform_cases,
        "memory_pool_mib": {
            "used": float(pool.used_bytes() / (1024 * 1024)),
            "reserved": float(pool.total_bytes() / (1024 * 1024)),
        },
        "interpretation": (
            "A faster same-FP64 candidate at small/medium batches indicates "
            "that scheduling, repeated native-wrapper synchronization, or "
            "handle lifecycle should be investigated before dtype reduction. "
            "The waveform cases test end-to-end engineering consistency but "
            "do not replace LISA/TDI or parameter-bias acceptance."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="ROMAN input batch sizes",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--waveform-repetitions", type=int, default=DEFAULT_WAVEFORM_REPETITIONS
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    if args.waveform_repetitions < 2:
        raise ValueError("--waveform-repetitions must be at least 2")
    if not args.sizes or any(size < 1 for size in args.sizes):
        raise ValueError("--sizes must contain positive integers")
    report = run_probe(tuple(args.sizes), args.repetitions, args.waveform_repetitions)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
