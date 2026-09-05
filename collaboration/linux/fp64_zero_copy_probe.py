#!/usr/bin/env python3
"""Compare the all-mode FP64 coefficient view with the legacy gather/copy.

2026-09-04 14:39 CST (linux): Add a Linux-owned isolation experiment for the
first master-rooted optimization.  The reference forces the original advanced-
indexing coefficient gather; the candidate uses the contiguous all-mode view.
Both invoke the same FP64 native CUDA interpolation and waveform kernels.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import numpy as np

import few
from few import get_backend
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PROJECT_ROOT / "collaboration/linux/master_cuda_baseline.json"
DEFAULT_DURATIONS = (0.001, 0.01, 1.0)


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        import time

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


@contextlib.contextmanager
def force_legacy_identity_gather() -> Iterator[None]:
    original = AmpInterpKerrEccEq.get_amplitudes

    def gathered_get_amplitudes(
        self: AmpInterpKerrEccEq,
        a: Any,
        p: Any,
        e: Any,
        x_i: Any,
        specific_modes: Any = None,
    ) -> Any:
        if specific_modes is self.mode_indexes:
            specific_modes = specific_modes.copy()
        return original(
            self, a, p, e, x_i, specific_modes=specific_modes
        )

    # 2026-09-04 14:39 CST (linux): This process-local patch recreates the
    # legacy gather only while timing the reference.  Production source and the
    # serialized candidate model are never mutated by the probe.
    AmpInterpKerrEccEq.get_amplitudes = gathered_get_amplitudes
    try:
        yield
    finally:
        AmpInterpKerrEccEq.get_amplitudes = original


def comparison(xp: Any, candidate: Any, reference: Any) -> dict[str, Any]:
    delta = candidate - reference
    return {
        "bitwise_equal": bool(xp.array_equal(candidate, reference)),
        "max_abs": float(xp.max(xp.abs(delta)).get()),
        "relative_l2": float(
            (xp.linalg.norm(delta.ravel()) / xp.linalg.norm(reference.ravel())).get()
        ),
    }


def run_probe(repetitions: int, durations: tuple[float, ...]) -> dict[str, Any]:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    backend = get_backend("cuda12x")
    xp = backend.xp
    waveform = FastKerrEccentricEquatorialFlux(force_backend="cuda12x")
    synchronize(xp)
    amplitude = waveform.amplitude_generator

    amplitude_cases = []
    for points in (1, 32, 128):
        p = xp.full(points, 8.0, dtype=xp.float64)
        e = xp.full(points, 0.3, dtype=xp.float64)
        x_i = xp.ones(points, dtype=xp.float64)
        with force_legacy_identity_gather():
            reference, reference_timing = timing_summary(
                xp,
                lambda p=p, e=e, x_i=x_i: amplitude(0.6, p, e, x_i),
                repetitions,
            )
        candidate, candidate_timing = timing_summary(
            xp,
            lambda p=p, e=e, x_i=x_i: amplitude(0.6, p, e, x_i),
            repetitions,
        )
        amplitude_cases.append(
            {
                "trajectory_points": points,
                "legacy_gather_fp64": reference_timing,
                "zero_copy_fp64": candidate_timing,
                "median_speedup": (
                    reference_timing["median_ms"] / candidate_timing["median_ms"]
                ),
                "candidate_vs_reference": comparison(xp, candidate, reference),
            }
        )

    waveform_args = (1.0e6, 1.0e1, 0.6, 8.0, 0.3, 1.0, np.pi / 3, np.pi / 4)
    baseline_by_duration = {
        item["duration_years"]: item for item in baseline["waveform_cases"]
    }
    waveform_cases = []
    for duration in durations:
        with force_legacy_identity_gather():
            reference, reference_timing = timing_summary(
                xp,
                lambda duration=duration: waveform(
                    *waveform_args, T=duration, dt=15.0, dist=1.0
                ),
                repetitions,
            )
        candidate, candidate_timing = timing_summary(
            xp,
            lambda duration=duration: waveform(
                *waveform_args, T=duration, dt=15.0, dist=1.0
            ),
            repetitions,
        )
        original_baseline = baseline_by_duration[duration]["timing"]["median_ms"]
        waveform_cases.append(
            {
                "duration_years": duration,
                "samples": int(candidate.size),
                "legacy_gather_fp64": reference_timing,
                "zero_copy_fp64": candidate_timing,
                "same_process_median_speedup": (
                    reference_timing["median_ms"] / candidate_timing["median_ms"]
                ),
                "original_baseline_median_speedup": (
                    original_baseline / candidate_timing["median_ms"]
                ),
                "five_x_target_ms": original_baseline / 5.0,
                "candidate_vs_reference": comparison(xp, candidate, reference),
            }
        )

    source_path = PROJECT_ROOT / "src/few/amplitude/ampinterp2d.py"
    return {
        "schema": 1,
        "collaboration_note": (
            "2026-09-04 14:40 CST (linux): Linux-owned FP64 zero-copy "
            "experiment on the codex 5x branch; default numerical precision "
            "and native kernels remain unchanged."
        ),
        "timestamp_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "branch": git_output("branch", "--show-current"),
        "project_commit": git_output("rev-parse", "HEAD"),
        "few_version": few.__version__,
        "few_module": str(Path(few.__file__).resolve()),
        "experiment": {
            "reference": "legacy identity advanced-index gather, FP64 kernel",
            "candidate": "contiguous identity coefficient view, FP64 kernel",
            "precision_changed": False,
            "warmups": 2,
            "repetitions": repetitions,
            "timing_scope": "warm synchronized wall time",
        },
        "source_sha256": {str(source_path): sha256_file(source_path)},
        "baseline_report": str(BASELINE_PATH),
        "amplitude_cases": amplitude_cases,
        "waveform_cases": waveform_cases,
        "memory_pool_mib": {
            "used": float(xp.get_default_memory_pool().used_bytes() / 2**20),
            "reserved": float(xp.get_default_memory_pool().total_bytes() / 2**20),
        },
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
