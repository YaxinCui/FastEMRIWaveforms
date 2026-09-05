#!/usr/bin/env python3
"""Measure the opt-in mixed32 Kerr amplitude path against FP64 in one process.

2026-09-04 14:47 CST (linux): Add the first full-table mixed-compute gate.  It
compares model construction, all-mode amplitudes, automatic mode selection, and
short/medium/science-duration waveforms using identical inputs and explicit
CUDA synchronization.  FP64 is constructed and measured first as the oracle.
"""

from __future__ import annotations

import argparse
import gc
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
from few import get_backend, get_file_manager
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "collaboration/linux/master_cuda_baseline.json"
DATA_FILENAME = "ZNAmps_l10_m10_n55_DS2Outer.h5"
DEFAULT_DURATIONS = (0.001, 0.01, 1.0)
AMPLITUDE_NORMALIZED_LIMIT = 5.0e-11
WAVEFORM_NORMALIZED_LIMIT = 5.0e-10
WAVEFORM_MISMATCH_LIMIT = 1.0e-10


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
    xp: Any, function: Callable[[], Any], repetitions: int
) -> tuple[Any, dict[str, Any]]:
    output = None
    for _ in range(2):
        output = function()
        synchronize(xp)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = function()
        synchronize(xp)
        samples.append(time.perf_counter() - start)
    values = np.asarray(samples)
    return output, {
        "samples_ms": [float(value * 1.0e3) for value in values],
        "median_ms": float(np.median(values) * 1.0e3),
        "minimum_ms": float(np.min(values) * 1.0e3),
        "p90_ms": float(np.quantile(values, 0.9) * 1.0e3),
    }


def host_array(xp: Any, value: Any) -> np.ndarray:
    return xp.asnumpy(value)


def selected_modes(xp: Any, waveform: Any) -> np.ndarray:
    return np.column_stack(
        [
            host_array(xp, getattr(waveform, name))
            for name in ("ls", "ms", "ks", "ns")
        ]
    )


def run_variant(
    xp: Any,
    label: str,
    amplitude_precision: str,
    summation_precision: str,
    repetitions: int,
    durations: tuple[float, ...],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()
    memory_before = process_memory_mib()
    start = time.perf_counter()
    waveform = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"interpolation_precision": amplitude_precision},
        sum_kwargs={"summation_precision": summation_precision},
        force_backend="cuda12x",
    )
    synchronize(xp)
    construction_s = time.perf_counter() - start
    memory_after_construction = process_memory_mib()
    amplitude = waveform.amplitude_generator

    amplitude_outputs: dict[str, np.ndarray] = {}
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
        repeat = amplitude(0.6, p, e, x_i)
        synchronize(xp)
        key = str(points)
        amplitude_outputs[key] = host_array(xp, output)
        amplitude_cases.append(
            {
                "trajectory_points": points,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "repeat_bitwise": bool(xp.array_equal(output, repeat)),
                "timing": timing,
            }
        )

    waveform_args = (1.0e6, 1.0e1, 0.6, 8.0, 0.3, 1.0, np.pi / 3, np.pi / 4)
    waveform_outputs: dict[str, np.ndarray] = {}
    mode_outputs: dict[str, np.ndarray] = {}
    waveform_cases = []
    for duration in durations:
        output, timing = timing_summary(
            xp,
            lambda duration=duration: waveform(
                *waveform_args, T=duration, dt=15.0, dist=1.0
            ),
            repetitions,
        )
        repeat = waveform(*waveform_args, T=duration, dt=15.0, dist=1.0)
        synchronize(xp)
        key = str(duration)
        waveform_outputs[key] = host_array(xp, output)
        mode_outputs[key] = selected_modes(xp, waveform)
        waveform_cases.append(
            {
                "duration_years": duration,
                "samples": int(output.size),
                "modes_kept": int(waveform.num_modes_kept),
                "output_dtype": str(output.dtype),
                "repeat_bitwise": bool(xp.array_equal(output, repeat)),
                "timing": timing,
            }
        )

    holder = amplitude.spin_information_holder_A[0]
    pool = xp.get_default_memory_pool()
    report = {
        "label": label,
        "amplitude_precision": amplitude_precision,
        "summation_precision": summation_precision,
        "construction_seconds": construction_s,
        "coefficient_dtype": str(holder.coeff.dtype),
        "knot_dtypes": [str(item.dtype) for item in holder.knots],
        "amplitude_cases": amplitude_cases,
        "waveform_cases": waveform_cases,
        "memory": {
            "process_before_mib": memory_before,
            "process_after_construction_mib": memory_after_construction,
            "cupy_pool_used_mib": float(pool.used_bytes() / 2**20),
            "cupy_pool_reserved_mib": float(pool.total_bytes() / 2**20),
        },
    }
    del waveform, amplitude, holder
    gc.collect()
    pool.free_all_blocks()
    return report, amplitude_outputs, waveform_outputs, mode_outputs


def numerical_metrics(
    actual: np.ndarray, reference: np.ndarray, normalized_limit: float
) -> dict[str, Any]:
    delta = actual - reference
    tiny = np.finfo(np.float64).tiny
    scale = max(float(np.max(np.abs(reference))), tiny)
    norm = max(float(np.linalg.norm(reference.ravel())), tiny)
    normalized_max = float(np.max(np.abs(delta))) / scale
    relative_l2 = float(np.linalg.norm(delta.ravel())) / norm
    return {
        "shape": list(actual.shape),
        "finite": bool(np.all(np.isfinite(actual))),
        "normalized_max_abs": normalized_max,
        "relative_l2": relative_l2,
        "normalized_limit": normalized_limit,
        "normalized_gate_passed": (
            normalized_max <= normalized_limit and relative_l2 <= normalized_limit
        ),
    }


def run_probe(repetitions: int, durations: tuple[float, ...]) -> dict[str, Any]:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    backend = get_backend("cuda12x")
    xp = backend.xp
    data_path = Path(get_file_manager().get_file(DATA_FILENAME)).resolve()

    fp64, fp64_amplitudes, fp64_waveforms, fp64_modes = run_variant(
        xp, "fp64", "fp64", "fp64", repetitions, durations
    )
    optimized, _, optimized_waveforms, optimized_modes = run_variant(
        xp,
        "fp64_optimized",
        "fp64",
        "fp64_optimized",
        repetitions,
        durations,
    )
    mixed32, _, mixed_waveforms, mixed32_modes = run_variant(
        xp,
        "mixed32_phase",
        "mixed32",
        "mixed32_phase",
        repetitions,
        durations,
    )
    mixed32_full, mixed32_full_amplitudes, mixed32_full_waveforms, mixed32_full_modes = (
        run_variant(
            xp,
            "mixed32_full",
            "mixed32",
            "mixed32_full",
            repetitions,
            durations,
        )
    )
    # 2026-09-04 15:31 CST (linux): Benchmark integer-phasor evaluation as a
    # distinct fifth variant so its speed and numerical cost remain auditable.
    recurrence, _, recurrence_waveforms, recurrence_modes = run_variant(
        xp,
        "mixed32_recurrence",
        "mixed32",
        "mixed32_recurrence",
        repetitions,
        durations,
    )
    # 2026-09-04 15:39 CST (linux): The final exploratory variant also narrows
    # the block-local mode reduction; keep its result separate from recurrence.
    mixed32_fast, _, fast_waveforms, fast_modes = run_variant(
        xp,
        "mixed32_fast",
        "mixed32",
        "mixed32_fast",
        repetitions,
        durations,
    )
    # 2026-09-04 17:21 CST (linux): Final 5x candidate combines range-reduced
    # fast trig, precombined +/-m symmetry, and block-local FP32 accumulation.
    intrinsic_fast, _, intrinsic_fast_waveforms, intrinsic_fast_modes = run_variant(
        xp,
        "mixed32_intrinsic_fast",
        "mixed32",
        "mixed32_intrinsic_fast",
        repetitions,
        durations,
    )

    amplitude_comparisons = []
    for fp64_case, mixed_case in zip(
        fp64["amplitude_cases"], mixed32_full["amplitude_cases"]
    ):
        key = str(fp64_case["trajectory_points"])
        metrics = numerical_metrics(
            mixed32_full_amplitudes[key],
            fp64_amplitudes[key],
            AMPLITUDE_NORMALIZED_LIMIT,
        )
        amplitude_comparisons.append(
            {
                "trajectory_points": fp64_case["trajectory_points"],
                "median_speedup": (
                    fp64_case["timing"]["median_ms"]
                    / mixed_case["timing"]["median_ms"]
                ),
                "candidate_vs_fp64": metrics,
            }
        )

    original_baseline = {
        str(item["duration_years"]): item for item in baseline["waveform_cases"]
    }
    waveform_comparisons = []
    for (
        fp64_case,
        optimized_case,
        mixed_case,
        mixed_full_case,
        recurrence_case,
        fast_case,
        intrinsic_fast_case,
    ) in zip(
        fp64["waveform_cases"],
        optimized["waveform_cases"],
        mixed32["waveform_cases"],
        mixed32_full["waveform_cases"],
        recurrence["waveform_cases"],
        mixed32_fast["waveform_cases"],
        intrinsic_fast["waveform_cases"],
    ):
        key = str(fp64_case["duration_years"])
        optimized_metrics = numerical_metrics(
            optimized_waveforms[key],
            fp64_waveforms[key],
            WAVEFORM_NORMALIZED_LIMIT,
        )
        optimized_mismatch = float(
            max(
                0.0,
                get_mismatch(optimized_waveforms[key], fp64_waveforms[key]),
            )
        )
        optimized_metrics.update(
            {
                "flat_weight_mismatch": optimized_mismatch,
                "mismatch_limit": WAVEFORM_MISMATCH_LIMIT,
                "mismatch_gate_passed": (
                    optimized_mismatch <= WAVEFORM_MISMATCH_LIMIT
                ),
            }
        )
        metrics = numerical_metrics(
            mixed_waveforms[key], fp64_waveforms[key], WAVEFORM_NORMALIZED_LIMIT
        )
        full_metrics = numerical_metrics(
            mixed32_full_waveforms[key],
            fp64_waveforms[key],
            WAVEFORM_NORMALIZED_LIMIT,
        )
        mismatch = float(
            max(0.0, get_mismatch(mixed_waveforms[key], fp64_waveforms[key]))
        )
        metrics.update(
            {
                "flat_weight_mismatch": mismatch,
                "mismatch_limit": WAVEFORM_MISMATCH_LIMIT,
                "mismatch_gate_passed": mismatch <= WAVEFORM_MISMATCH_LIMIT,
            }
        )
        full_mismatch = float(
            max(
                0.0,
                get_mismatch(
                    mixed32_full_waveforms[key], fp64_waveforms[key]
                ),
            )
        )
        full_metrics.update(
            {
                "flat_weight_mismatch": full_mismatch,
                "mismatch_limit": WAVEFORM_MISMATCH_LIMIT,
                "mismatch_gate_passed": full_mismatch
                <= WAVEFORM_MISMATCH_LIMIT,
            }
        )
        recurrence_metrics = numerical_metrics(
            recurrence_waveforms[key],
            fp64_waveforms[key],
            WAVEFORM_NORMALIZED_LIMIT,
        )
        recurrence_mismatch = float(
            max(
                0.0,
                get_mismatch(recurrence_waveforms[key], fp64_waveforms[key]),
            )
        )
        recurrence_metrics.update(
            {
                "flat_weight_mismatch": recurrence_mismatch,
                "mismatch_limit": WAVEFORM_MISMATCH_LIMIT,
                "mismatch_gate_passed": recurrence_mismatch
                <= WAVEFORM_MISMATCH_LIMIT,
            }
        )
        fast_metrics = numerical_metrics(
            fast_waveforms[key],
            fp64_waveforms[key],
            WAVEFORM_NORMALIZED_LIMIT,
        )
        fast_mismatch = float(
            max(0.0, get_mismatch(fast_waveforms[key], fp64_waveforms[key]))
        )
        fast_metrics.update(
            {
                "flat_weight_mismatch": fast_mismatch,
                "mismatch_limit": WAVEFORM_MISMATCH_LIMIT,
                "mismatch_gate_passed": fast_mismatch
                <= WAVEFORM_MISMATCH_LIMIT,
            }
        )
        intrinsic_fast_metrics = numerical_metrics(
            intrinsic_fast_waveforms[key],
            fp64_waveforms[key],
            WAVEFORM_NORMALIZED_LIMIT,
        )
        intrinsic_fast_mismatch = float(
            max(
                0.0,
                get_mismatch(
                    intrinsic_fast_waveforms[key], fp64_waveforms[key]
                ),
            )
        )
        intrinsic_fast_metrics.update(
            {
                "flat_weight_mismatch": intrinsic_fast_mismatch,
                "mismatch_limit": WAVEFORM_MISMATCH_LIMIT,
                "mismatch_gate_passed": intrinsic_fast_mismatch
                <= WAVEFORM_MISMATCH_LIMIT,
            }
        )
        mode_equal = np.array_equal(mixed32_modes[key], fp64_modes[key])
        full_mode_equal = np.array_equal(
            mixed32_full_modes[key], fp64_modes[key]
        )
        recurrence_mode_equal = np.array_equal(
            recurrence_modes[key], fp64_modes[key]
        )
        fast_mode_equal = np.array_equal(fast_modes[key], fp64_modes[key])
        intrinsic_fast_mode_equal = np.array_equal(
            intrinsic_fast_modes[key], fp64_modes[key]
        )
        optimized_mode_equal = np.array_equal(
            optimized_modes[key], fp64_modes[key]
        )
        fp64_median = fp64_case["timing"]["median_ms"]
        optimized_median = optimized_case["timing"]["median_ms"]
        mixed_median = mixed_case["timing"]["median_ms"]
        mixed_full_median = mixed_full_case["timing"]["median_ms"]
        recurrence_median = recurrence_case["timing"]["median_ms"]
        fast_median = fast_case["timing"]["median_ms"]
        intrinsic_fast_median = intrinsic_fast_case["timing"]["median_ms"]
        baseline_median = original_baseline[key]["timing"]["median_ms"]
        waveform_comparisons.append(
            {
                "duration_years": fp64_case["duration_years"],
                "fp64_optimized_median_speedup": (
                    fp64_median / optimized_median
                ),
                "mixed32_phase_median_speedup": fp64_median / mixed_median,
                "mixed32_full_median_speedup": (
                    fp64_median / mixed_full_median
                ),
                "mixed32_recurrence_median_speedup": (
                    fp64_median / recurrence_median
                ),
                "mixed32_fast_median_speedup": fp64_median / fast_median,
                "same_process_median_speedup": (
                    fp64_median / intrinsic_fast_median
                ),
                "original_master_median_speedup": (
                    baseline_median / intrinsic_fast_median
                ),
                "five_x_target_ms": fp64_median / 5.0,
                "five_x_performance_passed": (
                    intrinsic_fast_median <= fp64_median / 5.0
                ),
                "automatic_mode_selection_equal": bool(
                    intrinsic_fast_mode_equal
                ),
                "mixed32_fast_mode_selection_equal": bool(fast_mode_equal),
                "mixed32_recurrence_mode_selection_equal": bool(
                    recurrence_mode_equal
                ),
                "mixed32_full_mode_selection_equal": bool(full_mode_equal),
                "mixed32_phase_mode_selection_equal": bool(mode_equal),
                "fp64_optimized_mode_selection_equal": bool(
                    optimized_mode_equal
                ),
                "fp64_modes_kept": fp64_case["modes_kept"],
                "mixed32_modes_kept": mixed_full_case["modes_kept"],
                "fp64_optimized_vs_fp64": optimized_metrics,
                "mixed32_phase_vs_fp64": metrics,
                "mixed32_full_vs_fp64": full_metrics,
                "mixed32_recurrence_vs_fp64": recurrence_metrics,
                "mixed32_fast_vs_fp64": fast_metrics,
                "candidate_vs_fp64": intrinsic_fast_metrics,
            }
        )

    properties = xp.cuda.runtime.getDeviceProperties(0)
    source_paths = (
        PROJECT_ROOT / "src/few/amplitude/ampinterp2d.py",
        PROJECT_ROOT / "src/few/cutils/AmpInterp2D.cu",
        PROJECT_ROOT / "src/few/cutils/AmpInterp2D.hh",
        PROJECT_ROOT / "src/few/cutils/pyampinterp2D.pyx",
        PROJECT_ROOT / "src/few/cutils/interpolate.cu",
        PROJECT_ROOT / "src/few/cutils/interpolate.hh",
        PROJECT_ROOT / "src/few/cutils/pyinterp.pyx",
        PROJECT_ROOT / "src/few/summation/interpolatedmodesum.py",
    )
    return {
        "schema": 1,
        "collaboration_note": (
            "2026-09-04 15:02 CST (linux): Linux-owned opt-in mixed32 Kerr "
            "probe with a same-FP64 scheduling control. Strict inherited "
            "scientific gates are reported unchanged; a fast mode is not "
            "accepted merely because it is faster."
        ),
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "branch": git_output("branch", "--show-current"),
        "project_commit": git_output("rev-parse", "HEAD"),
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
        },
        "data": {
            "path": str(data_path),
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "experiment": {
            "fp64": "FP64 storage, spline compute, accumulation, and output",
            "mixed32": (
                "FP32 coefficient/knot/coordinate storage and spline "
                "accumulation plus FP64-range-reduced FP32 phase sin/cos; "
                "FP64 output, spin interpolation, trajectory, mode selection "
                "inputs, phase splines, and waveform accumulation"
            ),
            "mixed32_full": (
                "mixed32 amplitude interpolation plus FP32 amplitude-spline "
                "evaluation, component-reduced mode phases, phasors, and "
                "Ylms; FP64 phase-spline reconstruction and final accumulation"
            ),
            "mixed32_recurrence": (
                "mixed32_full data path plus three FP32 base phasors per "
                "sample and ordered integer recurrence for (m,k,n), replacing "
                "one FP32 sin/cos evaluation per selected mode"
            ),
            "mixed32_fast": (
                "mixed32_recurrence plus complex64 block-local mode "
                "accumulation and one complex128 promotion per mode block"
            ),
            "mixed32_intrinsic_fast": (
                "range-reduced CUDA fast sin/cos plus precombined +/-m "
                "symmetry weights and complex64 block accumulation; phase "
                "spline reconstruction and output storage remain FP64"
            ),
            "fp64_optimized": (
                "FP64 arithmetic with one final synchronization and direct "
                "stores instead of per-interval streams/synchronization and "
                "uncontended double atomics"
            ),
            "warmups": 2,
            "repetitions": repetitions,
            "timing_scope": "warm synchronized wall time",
            "dt_seconds": 15.0,
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in source_paths
        },
        "baseline_report": str(BASELINE_PATH),
        "fp64": fp64,
        "fp64_optimized": optimized,
        "mixed32_phase": mixed32,
        "mixed32_full": mixed32_full,
        "mixed32_recurrence": recurrence,
        "mixed32_fast": mixed32_fast,
        "mixed32_intrinsic_fast": intrinsic_fast,
        "amplitude_comparisons": amplitude_comparisons,
        "waveform_comparisons": waveform_comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--durations", nargs="+", type=float, default=list(DEFAULT_DURATIONS)
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 3:
        raise ValueError("--repetitions must be at least 3")
    report = run_probe(args.repetitions, tuple(args.durations))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
