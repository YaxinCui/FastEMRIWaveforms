#!/usr/bin/env python3
"""Synchronised per-stage timing for one warm CUDA Kerr waveform process."""

# 2026-09-04 15:47 CST (linux): Linux-owned diagnostic.  Each proxy
# synchronises before and after its wrapped FEW module so asynchronous CUDA
# work is charged to trajectory, mode selection, or summation explicitly.

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from few import get_backend
from few.waveform import FastKerrEccentricEquatorialFlux


class StageTimer:
    """Transparent callable proxy recording synchronised wall time."""

    def __init__(self, target: Any, xp: Any):
        self.target = target
        self.xp = xp
        self.samples_ms: list[float] = []
        self.last_output: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.xp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        output = self.target(*args, **kwargs)
        self.last_output = output
        self.xp.cuda.Stream.null.synchronize()
        self.samples_ms.append((time.perf_counter() - start) * 1000.0)
        return output


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "samples_ms": values,
    }


def run(precision: str, repetitions: int) -> dict[str, Any]:
    xp = get_backend("cuda12x").xp
    waveform = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"interpolation_precision": "mixed32"},
        sum_kwargs={"summation_precision": precision},
        force_backend="cuda12x",
    )
    trajectory = StageTimer(waveform.inspiral_generator, xp)
    selector = StageTimer(waveform.mode_selector, xp)
    # 2026-09-04 16:47 CST (linux): Instrument the spline constructor and the
    # selected compiled kernel separately.  Synchronisation before the kernel
    # leaves casts/copies in the outer summation residual instead of hiding
    # asynchronous preparation time inside the compiled call.
    spline = StageTimer(waveform.create_waveform.build_with_same_backend, xp)
    waveform.create_waveform.build_with_same_backend = spline
    kernel_method_names = {
        "mixed32_full": "get_waveform_mixed32_full_wrap",
        "mixed32_fast": "get_waveform_mixed32_fast_wrap",
        "mixed32_intrinsic": "get_waveform_mixed32_intrinsic_wrap",
        "mixed32_intrinsic_fast": "get_waveform_mixed32_intrinsic_fast_wrap",
        "mixed32_warp": "get_waveform_mixed32_warp_wrap",
    }
    kernel_name = kernel_method_names[precision]
    backend = waveform.create_waveform.backend
    original_kernel = getattr(backend, kernel_name)
    kernel = StageTimer(original_kernel, xp)
    setattr(backend, kernel_name, kernel)
    summation = StageTimer(waveform.create_waveform, xp)
    waveform.inspiral_generator = trajectory
    waveform.mode_selector = selector
    waveform.create_waveform = summation
    args = (1.0e6, 1.0e1, 0.6, 8.0, 0.3, 1.0, np.pi / 3, np.pi / 4)

    for _ in range(2):
        waveform(*args, T=1.0, dt=15.0, dist=1.0)
    trajectory.samples_ms.clear()
    selector.samples_ms.clear()
    summation.samples_ms.clear()
    spline.samples_ms.clear()
    kernel.samples_ms.clear()

    totals: list[float] = []
    for _ in range(repetitions):
        xp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        waveform(*args, T=1.0, dt=15.0, dist=1.0)
        xp.cuda.Stream.null.synchronize()
        totals.append((time.perf_counter() - start) * 1000.0)

    stage_medians = {
        "trajectory": statistics.median(trajectory.samples_ms),
        "mode_selection_including_amplitudes": statistics.median(
            selector.samples_ms
        ),
        "summation": statistics.median(summation.samples_ms),
    }
    total_median = statistics.median(totals)
    result = {
        "precision": precision,
        "samples": int(waveform.create_waveform.num_pts),
        "modes_kept": int(waveform.num_modes_kept),
        # 2026-09-04 16:28 CST (linux): Record sparse interval count to assess
        # whether per-interval CUDA launches dominate the summation stage.
        "trajectory_points": len(trajectory.last_output[0]),
        "total": summary(totals),
        "trajectory": summary(trajectory.samples_ms),
        "mode_selection_including_amplitudes": summary(selector.samples_ms),
        "summation": summary(summation.samples_ms),
        "spline_construction": summary(spline.samples_ms),
        "compiled_kernel": summary(kernel.samples_ms),
        "summation_preparation_median_ms": (
            statistics.median(summation.samples_ms)
            - statistics.median(spline.samples_ms)
            - statistics.median(kernel.samples_ms)
        ),
        "unattributed_median_ms": total_median - sum(stage_medians.values()),
    }
    # 2026-09-04 16:47 CST (linux): Backends are process-wide singletons; undo
    # diagnostic wrapping before the next precision variant is constructed.
    setattr(backend, kernel_name, original_kernel)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precisions", nargs="+", default=["mixed32_full", "mixed32_fast"]
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "collaboration_note": (
            "2026-09-04 15:47 CST (linux): synchronized stage attribution; "
            "diagnostic wrappers are not part of the measured production path"
        ),
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "variants": [run(item, args.repetitions) for item in args.precisions],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
