#!/usr/bin/env python3
"""Benchmark an in-memory Metal replacement for FEW time-domain mode summation.

2026-09-01 22:52 CST (mac): Add a real-waveform driver that temporarily swaps
the CPU backend's summation callable, restores it in a finally block, and
reports both end-to-end performance and numerical error. No backend registry,
build configuration, or installed extension is persistently modified.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_metal_interp import error_metrics

from few import get_backend
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("metal_sum.mm")
STRICT_SOURCE = Path(__file__).with_name("metal_sum_ds.mm")
WAVEFORM_ARGS = (1.0e6, 1.0e1, 0.7, 11.0, 0.4, 1.0, np.pi / 3, np.pi / 4)
BASE_KWARGS = {
    "dist": 1.0,
    "Phi_phi0": 0.3,
    "Phi_theta0": 0.0,
    "Phi_r0": 0.7,
    "T": 0.1,
    "dt": 15.0,
}


def double_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def int_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_int):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def build_library(precision: str = "f32") -> Path:
    # 2026-09-01 23:16 CST (mac): Keep the original FP32 experiment available
    # while allowing the strict full-chain double-single kernel to use the same
    # temporary-injection benchmark and ABI.
    if precision not in {"f32", "ds"}:
        raise ValueError(f"Unknown Metal summation precision: {precision}")
    source = STRICT_SOURCE if precision == "ds" else SOURCE
    output = Path(tempfile.gettempdir()) / f"few_metal_sum_{precision}_poc.dylib"
    subprocess.run(
        [
            "clang++",
            "-std=c++17",
            "-O3",
            "-fobjc-arc",
            "-dynamiclib",
            str(source),
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            "-framework",
            "Accelerate",
            "-o",
            str(output),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return output


def load_api(path: Path) -> ctypes.CDLL:
    api = ctypes.CDLL(str(path))
    dptr = ctypes.POINTER(ctypes.c_double)
    iptr = ctypes.POINTER(ctypes.c_int)
    api.few_metal_sum_last_error.restype = ctypes.c_char_p
    api.few_metal_sum_context_create.restype = ctypes.c_void_p
    api.few_metal_sum_context_destroy.argtypes = [ctypes.c_void_p]
    api.few_metal_sum_evaluate.argtypes = [
        ctypes.c_void_p,
        dptr,
        dptr,
        dptr,
        dptr,
        iptr,
        iptr,
        iptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        dptr,
        ctypes.c_double,
        dptr,
        dptr,
    ]
    api.few_metal_sum_evaluate.restype = ctypes.c_int
    return api


class MetalSummation:
    def __init__(self, api: ctypes.CDLL):
        self.api = api
        start = time.perf_counter()
        self.context = api.few_metal_sum_context_create()
        self.compile_seconds = time.perf_counter() - start
        if not self.context:
            raise RuntimeError(api.few_metal_sum_last_error().decode())
        self.calls: list[dict[str, float | int]] = []

    def __call__(
        self,
        waveform: np.ndarray,
        interpolation: np.ndarray,
        phase_times: np.ndarray,
        phase_coefficients: np.ndarray,
        m_values: np.ndarray,
        k_values: np.ndarray,
        n_values: np.ndarray,
        init_length: int,
        output_length: int,
        mode_count: int,
        ylms: np.ndarray,
        delta_t: float,
        trajectory_times: np.ndarray,
        _device: int,
    ) -> None:
        waveform_view = np.asarray(waveform).view(np.float64)
        interpolation = np.ascontiguousarray(interpolation, dtype=np.float64)
        phase_times = np.ascontiguousarray(phase_times, dtype=np.float64)
        phase_coefficients = np.ascontiguousarray(phase_coefficients, dtype=np.float64)
        m_values = np.ascontiguousarray(m_values, dtype=np.int32)
        k_values = np.ascontiguousarray(k_values, dtype=np.int32)
        n_values = np.ascontiguousarray(n_values, dtype=np.int32)
        ylms_view = np.ascontiguousarray(ylms, dtype=np.complex128).view(np.float64)
        trajectory_times = np.ascontiguousarray(trajectory_times, dtype=np.float64)
        gpu_seconds = ctypes.c_double()
        start = time.perf_counter()
        status = self.api.few_metal_sum_evaluate(
            self.context,
            double_pointer(waveform_view),
            double_pointer(interpolation),
            double_pointer(phase_times),
            double_pointer(phase_coefficients),
            int_pointer(m_values),
            int_pointer(k_values),
            int_pointer(n_values),
            init_length,
            output_length,
            mode_count,
            double_pointer(ylms_view),
            delta_t,
            double_pointer(trajectory_times),
            ctypes.byref(gpu_seconds),
        )
        wall_seconds = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(self.api.few_metal_sum_last_error().decode())
        self.calls.append(
            {
                "init_length": init_length,
                "output_length": output_length,
                "mode_count": mode_count,
                "synchronized_wall_seconds": wall_seconds,
                "gpu_seconds": gpu_seconds.value,
            }
        )

    def close(self) -> None:
        if self.context:
            self.api.few_metal_sum_context_destroy(self.context)
            self.context = None


def timed_waveforms(
    generator: Any,
    repetitions: int,
    kwargs: dict[str, float],
    waveform_args: tuple[float, ...] = WAVEFORM_ARGS,
) -> tuple[np.ndarray, list[float], bool]:
    outputs = []
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = np.asarray(generator(*waveform_args, **kwargs)).copy()
        timings.append(time.perf_counter() - start)
        outputs.append(output)
    repeatable = all(np.array_equal(outputs[0], value) for value in outputs[1:])
    return outputs[-1], timings, repeatable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-years", type=float, default=BASE_KWARGS["T"])
    parser.add_argument("--dt-seconds", type=float, default=BASE_KWARGS["dt"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--precision", choices=("f32", "ds"), default="f32")
    # 2026-09-01 23:18 CST (mac): Expose the orbital coordinates needed for
    # strict-kernel robustness scans without changing the established default.
    parser.add_argument("--spin", type=float, default=WAVEFORM_ARGS[2])
    parser.add_argument("--p0", type=float, default=WAVEFORM_ARGS[3])
    parser.add_argument("--e0", type=float, default=WAVEFORM_ARGS[4])
    parser.add_argument("--xI0", type=float, default=WAVEFORM_ARGS[5])
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This proof of concept requires Apple Silicon macOS")
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    kwargs = dict(BASE_KWARGS)
    kwargs["T"] = args.duration_years
    kwargs["dt"] = args.dt_seconds
    waveform_args = (
        WAVEFORM_ARGS[0],
        WAVEFORM_ARGS[1],
        args.spin,
        args.p0,
        args.e0,
        args.xI0,
        WAVEFORM_ARGS[6],
        WAVEFORM_ARGS[7],
    )

    load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend="cpu",
    )
    model_load_seconds = time.perf_counter() - load_start
    generator(*waveform_args, **kwargs)
    cpu_before, cpu_before_times, cpu_before_repeatable = timed_waveforms(
        generator, args.repetitions, kwargs, waveform_args
    )

    api = load_api(build_library(args.precision))
    metal_sum = MetalSummation(api)
    backend = get_backend("cpu")
    original_sum = backend.get_waveform_wrap
    try:
        backend.get_waveform_wrap = metal_sum
        generator(*waveform_args, **kwargs)
        metal_waveform, metal_times, metal_repeatable = timed_waveforms(
            generator, args.repetitions, kwargs, waveform_args
        )
    finally:
        backend.get_waveform_wrap = original_sum

    cpu_after, cpu_after_times, cpu_after_repeatable = timed_waveforms(
        generator, args.repetitions, kwargs, waveform_args
    )
    metal_sum.close()
    cpu_times = cpu_before_times + cpu_after_times
    report = {
        # 2026-09-01 23:23 CST (mac): Identify which precision tier produced
        # the report now that the driver can run both isolated kernels.
        "collaboration_note": (
            "2026-09-01 23:23 CST (mac): temporary Metal "
            f"{args.precision} mode-sum injection; CPU backend callable restored"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "model_load_seconds": model_load_seconds,
        "metal_pipeline_compile_seconds": metal_sum.compile_seconds,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024),
        "waveform_kwargs": kwargs,
        "waveform_args": waveform_args,
        "metal_sum_precision": args.precision,
        "num_modes_kept": int(generator.num_modes_kept),
        "summation_calls": metal_sum.calls,
        "waveform": {
            "shape": list(cpu_before.shape),
            "cpu_before_seconds": cpu_before_times,
            "cpu_after_seconds": cpu_after_times,
            "metal_seconds": metal_times,
            "cpu_median_seconds": statistics.median(cpu_times),
            "metal_median_seconds": statistics.median(metal_times),
            "end_to_end_speedup": statistics.median(cpu_times)
            / statistics.median(metal_times),
            "cpu_before_repeatable": cpu_before_repeatable,
            "cpu_after_repeatable": cpu_after_repeatable,
            "metal_repeatable": metal_repeatable,
            "cpu_before_after_bitwise": bool(np.array_equal(cpu_before, cpu_after)),
            "metal_vs_cpu": error_metrics(cpu_before, metal_waveform),
            "flat_mismatch": float(get_mismatch(cpu_before, metal_waveform)),
        },
    }
    # 2026-09-01 22:54 CST (mac): Keep the benchmark report on stdout while
    # satisfying the repository's no-debug-print lint rule.
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
