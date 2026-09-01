#!/usr/bin/env python3
"""Generate and compare deterministic full-table Kerr FEW artifacts.

2026-09-01 21:24 CST (mac): Add the opt-in, dual-host high-memory Kerr
acceptance runner after the registered 5.09 GB amplitude table was transferred
out of band and verified on Apple Silicon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import few
from few import get_backend
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

SCHEMA = 1
SEED = 20260901
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "src" / "few" / "data"
EXPECTED_DATA = {
    "KerrEccEqFluxData.h5": {
        "bytes": 9_857_632,
        "sha256": "db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014",
    },
    "ZNAmps_l10_m10_n55_DS2Outer.h5": {
        "bytes": 5_089_095_248,
        "sha256": "3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834",
    },
}

# These limits detect cross-backend regressions; they are not astrophysical
# waveform-model error budgets.
NORMALIZED_LIMITS = {
    "full_kerr_amplitudes": 5.0e-11,
    "known_kerr_amplitudes": 5.0e-11,
    "short_kerr_waveform": 5.0e-10,
}
WAVEFORM_MISMATCH_LIMIT = 1.0e-10
KNOWN_AMPLITUDE_ATOL = 1.0e-9

BROAD_AMPLITUDE_INPUTS = {
    "a": 0.7,
    "p": np.asarray([8.0, 10.0, 12.0, 14.0]),
    "e": np.asarray([0.1, 0.3, 0.5, 0.7]),
    "xI": np.ones(4),
}
KNOWN_AMPLITUDE_POINTS = (
    (
        (3, 2, 0, 4),
        (0.94071396, 3.03167504, 0.23666111),
        0.002753336140076555 + 0.004143260615912325j,
    ),
    (
        (2, -2, 0, 5),
        (0.77593827, 5.44830674, 0.39950367),
        -0.0001975310651632842 + 4.806588136412258e-05j,
    ),
    (
        (2, 2, 0, 0),
        (0.99090693, 1.81708312, 0.25313127),
        -0.09712303568244392 + 0.0004771647068539275j,
    ),
    (
        (4, -3, 0, 1),
        (0.87600927, 43.70360964, 0.478125),
        -9.89242565478408e-07 + 1.750469365836169e-05j,
    ),
    (
        (3, 3, 0, 0),
        (0.36158837, 72.94420732, 0.39375),
        5.869887391976143e-05 + 0.00136010620358342j,
    ),
)
# The third upstream fixture is not enforced: its expected value corresponds
# closely to (2,-2,0,5), not its declared (2,2,0,0) mode. In addition, the
# upstream assertion is outside the loop and therefore tests only the fifth
# point. Keep all five runtime outputs in the cross-host artifact, but do not
# turn that known fixture defect into a backend failure.
KNOWN_REFERENCE_ENFORCED = np.asarray([True, True, False, True, True])
WAVEFORM_ARGS = (1.0e6, 1.0e1, 0.7, 11.0, 0.4, 1.0, np.pi / 3, np.pi / 4)
WAVEFORM_KWARGS = {
    "dist": 1.0,
    "Phi_phi0": 0.3,
    "Phi_theta0": 0.0,
    "Phi_r0": 0.7,
    "T": 0.001,
    "dt": 15.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_json(value: Any) -> None:
    # 2026-09-01 21:24 CST (mac): Keep CLI reports on stdout while satisfying
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
    return to_host(output), time.perf_counter() - start


def memory_snapshot(backend: Any) -> dict[str, Any]:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    peak_rss_mib = (
        peak_rss / (1024 * 1024) if sys.platform == "darwin" else peak_rss / 1024
    )
    snapshot: dict[str, Any] = {"peak_process_rss_mib": float(peak_rss_mib)}
    if backend.uses_cupy:
        pool = backend.xp.get_default_memory_pool()
        snapshot["cupy_pool_used_mib"] = float(pool.used_bytes() / (1024 * 1024))
        snapshot["cupy_pool_total_mib"] = float(pool.total_bytes() / (1024 * 1024))
    return snapshot


def verified_file_metadata() -> list[dict[str, Any]]:
    metadata = []
    failures = []
    for name, expected in EXPECTED_DATA.items():
        path = DATA_DIRECTORY / name
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        actual = {
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        metadata.append(actual)
        if actual["bytes"] != expected["bytes"]:
            failures.append(
                f"{name}: expected {expected['bytes']} bytes, found {actual['bytes']}"
            )
        if actual["sha256"] != expected["sha256"]:
            failures.append(
                f"{name}: expected SHA256 {expected['sha256']}, found {actual['sha256']}"
            )
    if failures:
        raise RuntimeError("Kerr data preflight failed: " + "; ".join(failures))
    return metadata


def known_amplitude_outputs(amplitude: AmpInterpKerrEccEq) -> np.ndarray:
    output = []
    for mode, (a, p, e), _expected in KNOWN_AMPLITUDE_POINTS:
        mode_output = amplitude(a, p, e, 1.0, specific_modes=[mode])[mode]
        output.append(to_host(mode_output).item())
    return np.asarray(output)


def known_expected_amplitudes() -> np.ndarray:
    return np.asarray([point[2] for point in KNOWN_AMPLITUDE_POINTS])


def collect_outputs(
    backend_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    backend = get_backend(backend_name)
    np.random.default_rng(SEED)  # Reserve the schema seed for future input growth.
    diagnostics: dict[str, Any] = {
        "timings_seconds": {},
        "memory": {"before_models": memory_snapshot(backend)},
    }

    load_start = time.perf_counter()
    waveform_generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend=backend_name,
    )
    synchronize(backend)
    diagnostics["timings_seconds"]["kerr_model_load"] = time.perf_counter() - load_start
    diagnostics["memory"]["after_kerr_model_load"] = memory_snapshot(backend)
    amplitude = waveform_generator.amplitude_generator

    full_amplitudes, elapsed = timed(
        backend,
        lambda: amplitude(
            BROAD_AMPLITUDE_INPUTS["a"],
            BROAD_AMPLITUDE_INPUTS["p"],
            BROAD_AMPLITUDE_INPUTS["e"],
            BROAD_AMPLITUDE_INPUTS["xI"],
        ),
    )
    diagnostics["timings_seconds"]["full_kerr_amplitudes"] = elapsed
    full_repeat, repeat_elapsed = timed(
        backend,
        lambda: amplitude(
            BROAD_AMPLITUDE_INPUTS["a"],
            BROAD_AMPLITUDE_INPUTS["p"],
            BROAD_AMPLITUDE_INPUTS["e"],
            BROAD_AMPLITUDE_INPUTS["xI"],
        ),
    )
    diagnostics["timings_seconds"]["full_kerr_amplitudes_repeat"] = repeat_elapsed
    known_amplitudes, known_elapsed = timed(
        backend, lambda: known_amplitude_outputs(amplitude)
    )
    diagnostics["timings_seconds"]["known_kerr_amplitudes"] = known_elapsed
    diagnostics["memory"]["after_amplitudes"] = memory_snapshot(backend)

    known_delta = np.abs(known_amplitudes - known_expected_amplitudes())
    known_reference_points = []
    for index, (mode, inputs, expected) in enumerate(KNOWN_AMPLITUDE_POINTS):
        actual = known_amplitudes[index]
        known_reference_points.append(
            {
                "index": index,
                "mode": list(mode),
                "ape": list(inputs),
                "actual": [float(actual.real), float(actual.imag)],
                "expected": [float(expected.real), float(expected.imag)],
                "abs_error": float(known_delta[index]),
                "enforced": bool(KNOWN_REFERENCE_ENFORCED[index]),
                "within_atol": bool(known_delta[index] <= KNOWN_AMPLITUDE_ATOL),
            }
        )
    diagnostics["checks"] = {
        "full_amplitudes_finite": bool(np.all(np.isfinite(full_amplitudes))),
        "full_amplitudes_repeat_bitwise": bool(
            np.array_equal(full_amplitudes, full_repeat)
        ),
        "known_amplitudes_finite": bool(np.all(np.isfinite(known_amplitudes))),
        "known_amplitudes_max_abs_error": float(np.max(known_delta)),
        "known_amplitudes_within_atol": bool(
            np.all(known_delta[KNOWN_REFERENCE_ENFORCED] <= KNOWN_AMPLITUDE_ATOL)
        ),
        "known_amplitude_reference_points": known_reference_points,
        "known_amplitude_excluded_fixture_indices": [2],
    }

    waveform, elapsed = timed(
        backend, lambda: waveform_generator(*WAVEFORM_ARGS, **WAVEFORM_KWARGS)
    )
    diagnostics["timings_seconds"]["short_kerr_waveform"] = elapsed
    waveform_repeat, repeat_elapsed = timed(
        backend, lambda: waveform_generator(*WAVEFORM_ARGS, **WAVEFORM_KWARGS)
    )
    diagnostics["timings_seconds"]["short_kerr_waveform_repeat"] = repeat_elapsed
    diagnostics["memory"]["after_waveforms"] = memory_snapshot(backend)
    diagnostics["checks"].update(
        {
            "short_waveform_finite": bool(np.all(np.isfinite(waveform))),
            "short_waveform_repeat_bitwise": bool(
                np.array_equal(waveform, waveform_repeat)
            ),
            "short_waveform_repeat_mismatch": float(
                max(0.0, get_mismatch(waveform, waveform_repeat, use_gpu=False))
            ),
        }
    )

    if not all(
        value
        for key, value in diagnostics["checks"].items()
        if key.endswith(("_finite", "_bitwise", "_within_atol"))
    ):
        raise RuntimeError(
            f"High-memory Kerr runtime check failed: {diagnostics['checks']}"
        )

    outputs = {
        "full_kerr_amplitudes": full_amplitudes,
        "known_kerr_amplitudes": known_amplitudes,
        "short_kerr_waveform": waveform,
    }
    return outputs, diagnostics


def runtime_metadata(
    backend_name: str,
    data_files: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "seed": SEED,
        "backend": backend_name,
        "few_version": few.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "inputs": {
            "broad_amplitudes": {
                key: np.asarray(value).tolist()
                for key, value in BROAD_AMPLITUDE_INPUTS.items()
            },
            "known_amplitude_modes": [
                list(point[0]) for point in KNOWN_AMPLITUDE_POINTS
            ],
            "known_amplitude_ape": [list(point[1]) for point in KNOWN_AMPLITUDE_POINTS],
            "known_expected_amplitudes": [
                [float(point[2].real), float(point[2].imag)]
                for point in KNOWN_AMPLITUDE_POINTS
            ],
            "known_reference_enforced": KNOWN_REFERENCE_ENFORCED.tolist(),
            "waveform_args": list(WAVEFORM_ARGS),
            "waveform_kwargs": WAVEFORM_KWARGS,
        },
        "data_files": data_files,
        **diagnostics,
    }


def generate(backend_name: str, output_path: Path) -> int:
    data_files = verified_file_metadata()
    outputs, diagnostics = collect_outputs(backend_name)
    metadata = runtime_metadata(backend_name, data_files, diagnostics)
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
    data_files = verified_file_metadata()
    outputs, diagnostics = collect_outputs(backend_name)
    runtime = runtime_metadata(backend_name, data_files, diagnostics)
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
        metadata_checks = {
            "schema": reference_metadata.get("schema") == SCHEMA,
            "seed": reference_metadata.get("seed") == SEED,
            "inputs": reference_metadata.get("inputs") == runtime["inputs"],
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
            if (
                key == "short_kerr_waveform"
                and outputs[key].shape == reference[key].shape
            ):
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
