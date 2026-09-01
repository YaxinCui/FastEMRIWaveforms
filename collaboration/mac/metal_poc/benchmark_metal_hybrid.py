#!/usr/bin/env python3
"""Benchmark both isolated Metal candidates in one real Kerr waveform.

2026-09-01 22:50 CST (mac): Add the final combined feasibility experiment.
It temporarily injects strict double-single amplitude interpolation and the
approximate time-domain Metal sum, then restores all four holders and the CPU
backend callable before reporting. Nothing is installed or registered.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np
from benchmark_metal_interp import (
    MetalContext,
    error_metrics,
)
from benchmark_metal_interp import (
    build_library as build_interp_library,
)
from benchmark_metal_interp import (
    load_api as load_interp_api,
)
from benchmark_metal_sum import (
    BASE_KWARGS,
    WAVEFORM_ARGS,
    MetalSummation,
    timed_waveforms,
)
from benchmark_metal_sum import (
    build_library as build_sum_library,
)
from benchmark_metal_sum import (
    load_api as load_sum_api,
)
from benchmark_metal_waveform import MetalAmplitudeHolder

from few import get_backend
from few.utils.mappings.kerrecceq import kerrecceq_forward_map
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux


def install_amplitude_holders(
    generator: Any, context: MetalContext
) -> list[tuple[list[Any], int, Any, MetalAmplitudeHolder]]:
    amplitude = generator.amplitude_generator
    _u, _w, _y, z, _mask = kerrecceq_forward_map(
        np.full(1, WAVEFORM_ARGS[2]),
        np.asarray([WAVEFORM_ARGS[3]]),
        np.asarray([WAVEFORM_ARGS[4]]),
        np.ones(1),
        return_mask=True,
        kind="amplitude",
    )
    z_values = np.asarray(amplitude.z_values)
    above = int(np.flatnonzero(z_values > z[0])[0])
    below = above - 1
    replacements = []
    for holders in (
        amplitude.spin_information_holder_A,
        amplitude.spin_information_holder_B,
    ):
        for index in (below, above):
            original = holders[index]
            replacement = MetalAmplitudeHolder(original, context)
            holders[index] = replacement
            replacements.append((holders, index, original, replacement))
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-years", type=float, default=1.0)
    parser.add_argument("--dt-seconds", type=float, default=BASE_KWARGS["dt"])
    parser.add_argument("--repetitions", type=int, default=2)
    # 2026-09-01 23:20 CST (mac): Allow the combined experiment to select the
    # strict full-chain DS sum without removing the original FP32 comparison.
    parser.add_argument("--sum-precision", choices=("f32", "ds"), default="f32")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This proof of concept requires Apple Silicon macOS")
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    kwargs = dict(BASE_KWARGS)
    kwargs["T"] = args.duration_years
    kwargs["dt"] = args.dt_seconds

    load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend="cpu",
    )
    model_load_seconds = time.perf_counter() - load_start
    generator(*WAVEFORM_ARGS, **kwargs)
    cpu_before, cpu_before_times, cpu_before_repeatable = timed_waveforms(
        generator, args.repetitions, kwargs
    )

    interp_context = MetalContext(load_interp_api(build_interp_library()))
    sum_engine = MetalSummation(load_sum_api(build_sum_library(args.sum_precision)))
    plan_start = time.perf_counter()
    replacements = install_amplitude_holders(generator, interp_context)
    plan_creation_seconds = time.perf_counter() - plan_start
    backend = get_backend("cpu")
    original_sum = backend.get_waveform_wrap
    try:
        backend.get_waveform_wrap = sum_engine
        generator(*WAVEFORM_ARGS, **kwargs)
        hybrid_waveform, hybrid_times, hybrid_repeatable = timed_waveforms(
            generator, args.repetitions, kwargs
        )
    finally:
        backend.get_waveform_wrap = original_sum
        for holders, index, original, _replacement in replacements:
            holders[index] = original

    cpu_after, cpu_after_times, cpu_after_repeatable = timed_waveforms(
        generator, args.repetitions, kwargs
    )
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
    interp_metadata = interp_context.metadata()
    interp_context.close()
    sum_engine.close()

    cpu_times = cpu_before_times + cpu_after_times
    report = {
        # 2026-09-01 23:23 CST (mac): Preserve the selected summation precision
        # in the collaboration note as well as the structured Metal metadata.
        "collaboration_note": (
            "2026-09-01 23:23 CST (mac): combined in-memory Metal feasibility "
            f"run with {args.sum_precision} sum; all CPU objects restored"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "model_load_seconds": model_load_seconds,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024),
        "waveform_kwargs": kwargs,
        "num_modes_kept": int(generator.num_modes_kept),
        "metal": {
            "interpolation": interp_metadata,
            "summation_pipeline_compile_seconds": sum_engine.compile_seconds,
            "summation_precision": args.sum_precision,
            "amplitude_plan_creation_seconds": plan_creation_seconds,
            "amplitude_holders": holder_reports,
            "summation_calls": sum_engine.calls,
        },
        "waveform": {
            "shape": list(cpu_before.shape),
            "cpu_before_seconds": cpu_before_times,
            "cpu_after_seconds": cpu_after_times,
            "hybrid_seconds": hybrid_times,
            "cpu_median_seconds": statistics.median(cpu_times),
            "hybrid_median_seconds": statistics.median(hybrid_times),
            "end_to_end_speedup": statistics.median(cpu_times)
            / statistics.median(hybrid_times),
            "cpu_before_repeatable": cpu_before_repeatable,
            "cpu_after_repeatable": cpu_after_repeatable,
            "hybrid_repeatable": hybrid_repeatable,
            "cpu_before_after_bitwise": bool(np.array_equal(cpu_before, cpu_after)),
            "hybrid_vs_cpu": error_metrics(cpu_before, hybrid_waveform),
            "flat_mismatch": float(get_mismatch(cpu_before, hybrid_waveform)),
        },
    }
    # 2026-09-01 22:54 CST (mac): Emit reproducible JSON through stdout using
    # the same lint-safe convention as the dual-host validators.
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
