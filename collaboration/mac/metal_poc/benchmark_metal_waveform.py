#!/usr/bin/env python3
"""Inject the isolated Metal interpolator into one real Kerr FEW waveform.

2026-09-01 22:38 CST (mac): Add an opt-in end-to-end experiment that replaces
only the four fixed-spin amplitude holders used by the validation waveform.
The replacement is in-memory, is restored before exit, and never registers or
installs a production Metal backend.
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
    MetalPlan,
    build_library,
    error_metrics,
    load_api,
    to_complex,
)

from few.utils.mappings.kerrecceq import kerrecceq_forward_map
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

WAVEFORM_ARGS = (1.0e6, 1.0e1, 0.7, 11.0, 0.4, 1.0, np.pi / 3, np.pi / 4)
WAVEFORM_KWARGS = {
    "dist": 1.0,
    "Phi_phi0": 0.3,
    "Phi_theta0": 0.0,
    "Phi_r0": 0.7,
    "T": 0.001,
    "dt": 15.0,
}


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


class MetalAmplitudeHolder:
    """Full-mode, double-single Metal replacement for one AmpInterp2D holder."""

    def __init__(self, original: Any, context: MetalContext):
        self.original = original
        self.num_modes = int(original.num_teuk_modes)
        self.plan = MetalPlan(
            context,
            np.asarray(original.knots[0]),
            np.asarray(original.knots[1]),
            np.asarray(original.coeff),
        )
        self.calls: list[dict[str, float | int]] = []
        self.fallback_calls = 0

    def __call__(
        self,
        w: np.ndarray,
        u: np.ndarray,
        *_args: object,
        mode_indexes: np.ndarray | None = None,
        **_kwargs: object,
    ) -> np.ndarray:
        if mode_indexes is not None:
            indexes = np.asarray(mode_indexes)
            expected = np.arange(self.num_modes)
            if indexes.shape != expected.shape or not np.array_equal(indexes, expected):
                self.fallback_calls += 1
                return self.original(w, u, mode_indexes=mode_indexes)

        w = np.asarray(w, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if w.shape != u.shape:
            w, u = np.broadcast_arrays(w, u)
        output, wall_seconds, gpu_seconds = self.plan.evaluate(
            w.ravel(), u.ravel(), variant="double_single"
        )
        self.calls.append(
            {
                "points": int(w.size),
                "synchronized_wall_seconds": wall_seconds,
                "gpu_seconds": gpu_seconds,
            }
        )
        return to_complex(output, w.size)

    def close(self) -> None:
        self.plan.close()


def timed_waveform(
    generator: Any, repetitions: int, waveform_kwargs: dict[str, float]
) -> tuple[np.ndarray, list[float]]:
    outputs = []
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = np.asarray(generator(*WAVEFORM_ARGS, **waveform_kwargs))
        timings.append(time.perf_counter() - start)
        outputs.append(output)
    if not all(np.array_equal(outputs[0], value) for value in outputs[1:]):
        raise RuntimeError("Repeated waveform output is not bitwise deterministic")
    return outputs[-1], timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    # 2026-09-01 22:43 CST (mac): Permit duration scaling without editing the
    # validated baseline constants or creating a separate benchmark script.
    parser.add_argument("--duration-years", type=float, default=WAVEFORM_KWARGS["T"])
    parser.add_argument("--dt-seconds", type=float, default=WAVEFORM_KWARGS["dt"])
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This proof of concept requires Apple Silicon macOS")
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    waveform_kwargs = dict(WAVEFORM_KWARGS)
    waveform_kwargs["T"] = args.duration_years
    waveform_kwargs["dt"] = args.dt_seconds

    load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend="cpu",
    )
    model_load_seconds = time.perf_counter() - load_start

    # Prime trajectory, selector, spline, and summation state before comparing
    # the two amplitude implementations.
    generator(*WAVEFORM_ARGS, **waveform_kwargs)
    cpu_before, cpu_before_times = timed_waveform(
        generator, args.repetitions, waveform_kwargs
    )

    api = load_api(build_library())
    context = MetalContext(api)
    amplitude = generator.amplitude_generator
    signed_spin = np.full(1, WAVEFORM_ARGS[2])
    _u, _w, _y, z, _mask = kerrecceq_forward_map(
        signed_spin,
        np.asarray([WAVEFORM_ARGS[3]]),
        np.asarray([WAVEFORM_ARGS[4]]),
        np.ones(1),
        return_mask=True,
        kind="amplitude",
    )
    z_values = np.asarray(amplitude.z_values)
    above = int(np.flatnonzero(z_values > z[0])[0])
    below = above - 1

    replacements: list[tuple[list[Any], int, Any, MetalAmplitudeHolder]] = []
    plan_start = time.perf_counter()
    for holders in (
        amplitude.spin_information_holder_A,
        amplitude.spin_information_holder_B,
    ):
        for index in (below, above):
            original = holders[index]
            replacement = MetalAmplitudeHolder(original, context)
            replacements.append((holders, index, original, replacement))
    plan_seconds = time.perf_counter() - plan_start

    try:
        for holders, index, _original, replacement in replacements:
            holders[index] = replacement
        generator(*WAVEFORM_ARGS, **waveform_kwargs)
        metal_waveform, metal_times = timed_waveform(
            generator, args.repetitions, waveform_kwargs
        )
    finally:
        for holders, index, original, _replacement in replacements:
            holders[index] = original

    cpu_after, cpu_after_times = timed_waveform(
        generator, args.repetitions, waveform_kwargs
    )
    proxy_calls = []
    for _holders, index, _original, replacement in replacements:
        proxy_calls.append(
            {
                "z_index": index,
                "plan_upload_seconds": replacement.plan.upload_seconds,
                "fallback_calls": replacement.fallback_calls,
                "calls": replacement.calls,
            }
        )
        replacement.close()
    context_metadata = context.metadata()
    context.close()

    cpu_times = cpu_before_times + cpu_after_times
    report = {
        "collaboration_note": (
            "2026-09-01 22:38 CST (mac): in-memory double-single Metal "
            "amplitude injection; original FEW holders restored"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "metal": context_metadata,
        "model_load_seconds": model_load_seconds,
        "waveform_kwargs": waveform_kwargs,
        "metal_plan_creation_seconds": plan_seconds,
        "peak_process_rss_mib": peak_rss_mib(),
        "waveform": {
            "shape": list(cpu_before.shape),
            "cpu_before_seconds": cpu_before_times,
            "cpu_after_seconds": cpu_after_times,
            "metal_seconds": metal_times,
            "cpu_median_seconds": statistics.median(cpu_times),
            "metal_median_seconds": statistics.median(metal_times),
            "end_to_end_speedup": statistics.median(cpu_times)
            / statistics.median(metal_times),
            "metal_vs_cpu_before": error_metrics(cpu_before, metal_waveform),
            "metal_vs_cpu_after": error_metrics(cpu_after, metal_waveform),
            "flat_mismatch_vs_cpu_before": float(
                get_mismatch(cpu_before, metal_waveform)
            ),
            "flat_mismatch_vs_cpu_after": float(
                get_mismatch(cpu_after, metal_waveform)
            ),
            "cpu_before_after_bitwise": bool(np.array_equal(cpu_before, cpu_after)),
        },
        "amplitude_holders": proxy_calls,
    }
    # 2026-09-01 22:54 CST (mac): Emit the reviewed JSON report without a
    # debug-print call, matching the shared validation scripts.
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
