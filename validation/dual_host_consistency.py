#!/usr/bin/env python3
"""Generate and compare deterministic FEW dual-host artifacts.

2026-09-01 18:59 CST (mac): Add a shared Mac-CPU/Ubuntu-CPU-CUDA consistency
runner with fixed inputs, scale-aware FP64 tolerances, and data-file checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import few
from few import get_backend
from few.amplitude.ampinterp2d import AmpInterpSchwarzEcc
from few.amplitude.romannet import RomanAmplitude
from few.utils.utility import get_mismatch
from few.waveform import FastSchwarzschildEccentricFlux, Pn5AAKWaveform

SEED = 20260901
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "src" / "few" / "data"
DATA_FILES = (
    "AmplitudeVectorNorm.dat",
    "FluxNewMinusPNScaled_fixed_y_order.dat",
    "SchwarzschildEccentricInput.hdf5",
    "Teuk_amps_a0.0_lmax_10_nmax_30_new.h5",
)

# These are scale-aware cross-backend limits, not waveform-model error budgets.
NORMALIZED_LIMITS = {
    "neural_layer": 5.0e-13,
    "complex_projection": 5.0e-13,
    "roman_amplitudes": 5.0e-12,
    "bicubic_amplitudes": 5.0e-12,
    "schwarzschild_waveform": 5.0e-11,
    # 2026-09-01 19:57 CST (linux): The CPU AAK kernel uses its historical
    # Numerical Recipes Bessel approximation while CUDA uses libdevice jn.
    # Preserve a strict shape/scale bound for that known implementation split;
    # waveform mismatch remains independently limited below.
    "aak_waveform": 5.0e-9,
}
WAVEFORM_MISMATCH_LIMIT = 1.0e-10
WAVEFORM_KEYS = {"schwarzschild_waveform", "aak_waveform"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_json(value: Any) -> None:
    # 2026-09-01 19:02 CST (mac): Keep CLI reports on stdout while satisfying
    # the project's no-debug-print lint rule.
    sys.stdout.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def to_host(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def synchronize(backend: Any) -> None:
    if backend.uses_cupy:
        backend.xp.cuda.Stream.null.synchronize()


def timed(backend: Any, function: Callable[[], Any]) -> tuple[np.ndarray, float]:
    synchronize(backend)
    start = time.perf_counter()
    output = function()
    synchronize(backend)
    elapsed = time.perf_counter() - start
    return to_host(output), elapsed


def low_level_outputs(backend: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    xp = backend.xp

    m, k, n = 257, 37, 61
    matrix = rng.normal(size=(m, k))
    weights = rng.normal(size=(k, n))
    bias = rng.normal(size=n)
    matrix_device = xp.asarray(np.asfortranarray(matrix).ravel(order="F"))
    weights_device = xp.asarray(np.asfortranarray(weights).ravel(order="F"))
    bias_device = xp.asarray(bias)
    neural_device = xp.empty(m * n, dtype=xp.float64)
    backend.neural_layer_wrap(
        neural_device,
        matrix_device,
        weights_device,
        bias_device,
        m,
        k,
        n,
        1,
    )
    synchronize(backend)
    neural = to_host(neural_device).reshape((m, n), order="F")

    projection_m, projection_k, projection_n = 193, 29, 73
    network_output = rng.normal(size=(projection_m, 2 * projection_k))
    transform = rng.normal(size=(projection_k, projection_n)) + 1j * rng.normal(
        size=(projection_k, projection_n)
    )
    network_device = xp.asarray(np.asfortranarray(network_output).ravel(order="F"))
    transform_device = xp.asarray(np.asfortranarray(transform).ravel(order="F"))
    network_complex_device = xp.empty(projection_m * projection_k, dtype=xp.complex128)
    projection_device = xp.empty(projection_m * projection_n, dtype=xp.complex128)
    backend.transform_output_wrap(
        projection_device,
        transform_device,
        network_complex_device,
        network_device,
        projection_m,
        projection_k,
        0.001,
        projection_n,
    )
    synchronize(backend)
    projection = to_host(projection_device).reshape(
        (projection_m, projection_n), order="F"
    )
    return {"neural_layer": neural, "complex_projection": projection}


def collect_outputs(
    backend_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    backend = get_backend(backend_name)
    rng = np.random.default_rng(SEED)
    outputs = low_level_outputs(backend, rng)
    timings: dict[str, float] = {}

    p_grid, e_grid = np.meshgrid(np.linspace(10.0, 14.0, 16), np.linspace(0.1, 0.6, 8))
    p = p_grid.ravel()
    e = e_grid.ravel()
    x = np.ones_like(p)

    roman = RomanAmplitude(buffer_length=256, force_backend=backend_name)
    outputs["roman_amplitudes"], timings["roman_amplitudes"] = timed(
        backend, lambda: roman(0.0, p, e, x)
    )

    bicubic = AmpInterpSchwarzEcc(force_backend=backend_name)
    outputs["bicubic_amplitudes"], timings["bicubic_amplitudes"] = timed(
        backend, lambda: bicubic(0.0, p, e, x)
    )

    schwarzschild = FastSchwarzschildEccentricFlux(force_backend=backend_name)
    schwarzschild_args = (1e6, 1e1, 8.0, 0.2, np.pi / 3, np.pi / 4)
    outputs["schwarzschild_waveform"], timings["schwarzschild_waveform"] = timed(
        backend,
        lambda: schwarzschild(*schwarzschild_args, dist=1.0, T=0.01, dt=15.0),
    )

    inspiral_kwargs = {"DENSE_STEPPING": 0, "buffer_length": 1000}
    aak = Pn5AAKWaveform(
        inspiral_kwargs,
        {"pad_output": False},
        force_backend=backend_name,
    )
    aak_args = (
        1e6,
        1e1,
        0.2,
        14.0,
        0.2,
        np.cos(0.1),
        0.2,
        0.2,
        0.8,
        0.8,
        1.0,
    )
    outputs["aak_waveform"], timings["aak_waveform"] = timed(
        backend,
        lambda: aak(*aak_args, mich=False, dt=10.0, T=0.001),
    )

    for key, output in outputs.items():
        if not np.all(np.isfinite(output)):
            raise RuntimeError(f"Workload '{key}' produced a non-finite value")
    return outputs, timings


def file_metadata() -> list[dict[str, Any]]:
    metadata = []
    for name in DATA_FILES:
        path = DATA_DIRECTORY / name
        if path.is_file():
            metadata.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return metadata


def runtime_metadata(backend_name: str, timings: dict[str, float]) -> dict[str, Any]:
    return {
        "schema": 1,
        "seed": SEED,
        "backend": backend_name,
        "few_version": few.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timings_seconds": timings,
        "data_files": file_metadata(),
    }


def generate(backend_name: str, output_path: Path) -> int:
    outputs, timings = collect_outputs(backend_name)
    metadata = runtime_metadata(backend_name, timings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **outputs,
    )
    emit_json(metadata)
    emit_json(
        {
            "artifact": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    )
    return 0


def comparison_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "reason": "shape mismatch",
        }
    delta = actual - expected
    reference_scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    reference_norm = max(float(np.linalg.norm(expected.ravel())), np.finfo(float).tiny)
    return {
        "shape": list(actual.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "normalized_max_abs": float(np.max(np.abs(delta)) / reference_scale),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / reference_norm),
    }


def compare(backend_name: str, reference_path: Path, report_path: Path | None) -> int:
    outputs, timings = collect_outputs(backend_name)
    runtime = runtime_metadata(backend_name, timings)
    report: dict[str, Any] = {
        "reference": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "runtime": runtime,
        "results": {},
        "passed": True,
    }
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_metadata = json.loads(str(reference["metadata_json"]))
        report["reference_metadata"] = reference_metadata
        # 2026-09-01 19:18 CST (mac): Treat schema, seed, and input-data identity
        # as validation requirements, not merely informational report fields.
        metadata_checks = {
            "schema": reference_metadata.get("schema") == 1,
            "seed": reference_metadata.get("seed") == SEED,
            "data_files": reference_metadata.get("data_files") == runtime["data_files"],
        }
        report["metadata_checks"] = metadata_checks
        report["passed"] = all(metadata_checks.values())
        for key, limit in NORMALIZED_LIMITS.items():
            metrics = comparison_metrics(outputs[key], reference[key])
            passed = (
                metrics.get("normalized_max_abs", float("inf")) <= limit
                and metrics.get("relative_l2", float("inf")) <= limit
            )
            if key in WAVEFORM_KEYS and outputs[key].shape == reference[key].shape:
                mismatch = float(
                    max(0.0, get_mismatch(outputs[key], reference[key], use_gpu=False))
                )
                metrics["flat_weight_mismatch"] = mismatch
                passed = passed and mismatch <= WAVEFORM_MISMATCH_LIMIT
            metrics["normalized_limit"] = limit
            metrics["passed"] = passed
            report["results"][key] = metrics
            report["passed"] = report["passed"] and passed

    rendered = json.dumps(report, indent=2, sort_keys=True)
    sys.stdout.write(rendered + "\n")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--backend", default="cpu")
    generate_parser.add_argument("--output", required=True, type=Path)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--backend", required=True)
    compare_parser.add_argument("--reference", required=True, type=Path)
    compare_parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return generate(args.backend, args.output)
    return compare(args.backend, args.reference, args.report)


if __name__ == "__main__":
    sys.exit(main())
