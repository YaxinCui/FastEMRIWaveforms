#!/usr/bin/env python3
"""Capture cross-host Kerr trajectory and DOP853 decision evidence.

2026-09-02 17:42 CST (mac): Add an opt-in, production-neutral diagnostic for
the isolated ``_p_to_u`` fast-math hypothesis and the first adaptive-step
divergence between Apple arm64 and Ubuntu x86_64. Generated NPZ metadata and
JSON reports carry their own host/CST collaboration annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from math import log
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numba
import numpy as np
import scipy
from numba import njit

import few
import few.trajectory.ode.flux as flux_module
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "collaboration/mac/trajectory_reproducibility"
SCHEMA = "few-trajectory-reproducibility-v1"
ARRAY_NAMES = (
    "trajectory",
    "integrator_times",
    "spline_coefficients",
    "p_to_u_probe_inputs",
    "p_to_u_probe_outputs",
    "attempt_t_before",
    "attempt_h",
    "attempt_error",
    "attempt_error_old",
    "attempt_previous_reject",
    "attempt_accepted",
    "attempt_h_next",
    "attempt_t_after",
    "attempt_state_before",
    "attempt_state_after",
)
VARIANTS = ("fastmath", "strict")
INPUTS = {
    "M": 1_000_000.0,
    "mu": 10.0,
    "a": 0.7,
    "p0": 11.0,
    "e0": 0.4,
    "xI0": 1.0,
    "Phi_phi0": 0.3,
    "Phi_theta0": 0.0,
    "Phi_r0": 0.7,
    "T": 1.0,
    "dt": 15.0,
    "err": 1.0e-11,
}
SOURCE_PATHS = (
    Path("validation/trajectory_reproducibility.py"),
    Path("src/few/trajectory/ode/flux.py"),
    Path("src/few/trajectory/dopr853.py"),
    Path("src/few/trajectory/integrate.py"),
    Path("src/few/trajectory/inspiral.py"),
    Path("src/few/utils/elliptic.py"),
)
DATA_PATHS = (Path("src/few/data/KerrEccEqFluxData.h5"),)
PROBE_INPUTS = np.asarray(
    [
        [5.051, 5.0],
        [5.1, 5.0],
        [6.0, 5.0],
        [8.0, 5.0],
        [11.0, 5.0],
        [20.0, 5.0],
        [200.0, 5.0],
    ],
    dtype=np.float64,
)


@njit(fastmath=False, cache=True)
def strict_p_to_u(p: float, p_sep: float) -> float:
    """Evaluate the production mapping without LLVM fast-math permissions."""
    return log((p - p_sep + 4.0 - 0.05) / 4.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = PROJECT_ROOT / path
    return {
        "path": str(path),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def display_path(path: Path) -> str:
    """Prefer a repository-relative path while supporting temporary outputs."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def host_label() -> str:
    return "mac" if platform.system() == "Darwin" else "linux"


def cst_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")


def scalar(array: np.ndarray) -> float:
    return float(np.asarray(array).ravel()[0])


def attach_step_trace(trajectory: EMRIInspiral) -> list[dict[str, Any]]:
    """Attach instance-local wrappers without changing production classes."""
    stepper = trajectory.inspiral_generator.dopr
    events: list[dict[str, Any]] = []
    original_controller = stepper.controllerSuccess
    original_take_step_single = stepper.take_step_single

    def traced_controller(
        flag_success,
        err,
        err_old,
        previous_reject,
        h,
        x,
    ):
        event = {
            "t_before": scalar(x),
            "h": scalar(h),
            "error": scalar(err),
            "error_old": scalar(err_old),
            "previous_reject": bool(np.asarray(previous_reject).ravel()[0]),
        }
        original_controller(
            flag_success,
            err,
            err_old,
            previous_reject,
            h,
            x,
        )
        event.update(
            {
                "accepted": bool(np.asarray(flag_success).ravel()[0]),
                "h_next": scalar(h),
            }
        )
        events.append(event)

    def traced_take_step_single(x, h, y, additional_args):
        state_before = np.asarray(y, dtype=np.float64).copy()
        event_count = len(events)
        result = original_take_step_single(x, h, y, additional_args)
        if len(events) != event_count + 1:
            raise RuntimeError("DOP853 trace/controller event count diverged")
        events[-1]["t_after"] = float(result[1])
        events[-1]["state_before"] = state_before
        events[-1]["state_after"] = np.asarray(y, dtype=np.float64).copy()
        return result

    stepper.controllerSuccess = traced_controller
    stepper.take_step_single = traced_take_step_single
    return events


def events_to_arrays(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not events:
        raise RuntimeError("The adaptive DOP853 run produced no trace events")
    return {
        "attempt_t_before": np.asarray(
            [event["t_before"] for event in events], dtype=np.float64
        ),
        "attempt_h": np.asarray([event["h"] for event in events], dtype=np.float64),
        "attempt_error": np.asarray(
            [event["error"] for event in events], dtype=np.float64
        ),
        "attempt_error_old": np.asarray(
            [event["error_old"] for event in events], dtype=np.float64
        ),
        "attempt_previous_reject": np.asarray(
            [event["previous_reject"] for event in events], dtype=np.bool_
        ),
        "attempt_accepted": np.asarray(
            [event["accepted"] for event in events], dtype=np.bool_
        ),
        "attempt_h_next": np.asarray(
            [event["h_next"] for event in events], dtype=np.float64
        ),
        "attempt_t_after": np.asarray(
            [event["t_after"] for event in events], dtype=np.float64
        ),
        "attempt_state_before": np.stack([event["state_before"] for event in events]),
        "attempt_state_after": np.stack([event["state_after"] for event in events]),
    }


def run_variant(
    variant: str,
    mapping: Callable[[float, float], float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    flux_module._p_to_u = mapping
    # Exclude one-time Numba compilation from the diagnostic wall time.
    probe_outputs = np.asarray(
        [mapping(p, p_sep) for p, p_sep in PROBE_INPUTS], dtype=np.float64
    )

    def call_trajectory(trajectory: EMRIInspiral):
        return trajectory(
            INPUTS["M"],
            INPUTS["mu"],
            INPUTS["a"],
            INPUTS["p0"],
            INPUTS["e0"],
            INPUTS["xI0"],
            Phi_phi0=INPUTS["Phi_phi0"],
            Phi_theta0=INPUTS["Phi_theta0"],
            Phi_r0=INPUTS["Phi_r0"],
            T=INPUTS["T"],
            dt=INPUTS["dt"],
            err=INPUTS["err"],
        )

    # 2026-09-02 17:46 CST (mac): Warm the complete variant before recording
    # elapsed time so the fast-math/strict diagnostic does not present Numba
    # compilation as a numerical-variant performance difference.
    call_trajectory(EMRIInspiral(func=KerrEccEqFlux))
    trajectory = EMRIInspiral(func=KerrEccEqFlux)
    events = attach_step_trace(trajectory)
    start = time.perf_counter()
    output = call_trajectory(trajectory)
    elapsed = time.perf_counter() - start

    arrays = {
        "trajectory": np.column_stack(output),
        "integrator_times": np.asarray(
            trajectory.integrator_spline_t, dtype=np.float64
        ),
        "spline_coefficients": np.asarray(
            trajectory.integrator_spline_coeff, dtype=np.float64
        ),
        "p_to_u_probe_inputs": PROBE_INPUTS.copy(),
        "p_to_u_probe_outputs": probe_outputs,
        **events_to_arrays(events),
    }
    info = {
        "variant": variant,
        "elapsed_seconds": elapsed,
        "trajectory_points": int(arrays["trajectory"].shape[0]),
        "attempts": len(events),
        "accepted_attempts": int(np.count_nonzero(arrays["attempt_accepted"])),
        "rejected_attempts": int(np.count_nonzero(~arrays["attempt_accepted"])),
        "minimum_distance_from_acceptance_boundary": float(
            np.min(np.abs(arrays["attempt_error"] - 1.0))
        ),
        "p_sep_cache": float(trajectory.func.p_sep_cache),
    }
    return arrays, info


def numeric_comparison(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if reference.dtype != actual.dtype:
        return {
            "compatible": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "reference_dtype": str(reference.dtype),
            "actual_dtype": str(actual.dtype),
        }

    if reference.shape != actual.shape:
        result: dict[str, Any] = {
            "compatible": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "reference_dtype": str(reference.dtype),
            "actual_dtype": str(actual.dtype),
        }
        if (
            reference.ndim == actual.ndim
            and reference.ndim > 0
            and reference.shape[1:] == actual.shape[1:]
        ):
            common_length = min(reference.shape[0], actual.shape[0])
            result["common_first_axis_length"] = common_length
            if common_length:
                result["common_prefix"] = numeric_comparison(
                    reference[:common_length], actual[:common_length]
                )
        return result

    # 2026-09-02 17:46 CST (mac): Compare element bytes rather than NumPy value
    # equality so signed zero and distinct NaN payloads cannot be called
    # bitwise-equal in cross-host evidence.
    reference_bytes = (
        np.ascontiguousarray(reference)
        .view(np.uint8)
        .reshape(reference.size, reference.dtype.itemsize)
    )
    actual_bytes = (
        np.ascontiguousarray(actual)
        .view(np.uint8)
        .reshape(actual.size, actual.dtype.itemsize)
    )
    differing = np.flatnonzero(np.any(reference_bytes != actual_bytes, axis=1))
    equal = differing.size == 0
    first_flat_index = None if equal else int(differing[0])
    result: dict[str, Any] = {
        "compatible": True,
        "bitwise_equal": bool(equal),
        "first_differing_flat_index": first_flat_index,
        "first_differing_index": (
            None
            if first_flat_index is None
            # 2026-09-03 16:09 CST (linux): Convert NumPy scalar indices to
            # built-in ints so a cross-host difference remains JSON serializable.
            else [
                int(index)
                for index in np.unravel_index(first_flat_index, reference.shape)
            ]
        ),
    }
    if np.issubdtype(reference.dtype, np.number) and not np.issubdtype(
        reference.dtype, np.bool_
    ):
        difference = actual.astype(np.float64) - reference.astype(np.float64)
        reference_scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
        reference_norm = max(
            float(np.linalg.norm(reference.ravel())), np.finfo(float).tiny
        )
        result.update(
            {
                "max_absolute": float(np.max(np.abs(difference))),
                "normalized_max": float(np.max(np.abs(difference)) / reference_scale),
                "relative_l2": float(
                    np.linalg.norm(difference.ravel()) / reference_norm
                ),
            }
        )
    return result


def compare_array_sets(
    reference: dict[str, np.ndarray], actual: dict[str, np.ndarray]
) -> dict[str, Any]:
    reference_keys = set(reference)
    actual_keys = set(actual)
    common = sorted(reference_keys & actual_keys)
    comparisons = {
        key: numeric_comparison(reference[key], actual[key]) for key in common
    }
    return {
        "reference_only": sorted(reference_keys - actual_keys),
        "actual_only": sorted(actual_keys - reference_keys),
        "arrays": comparisons,
        "all_bitwise_equal": bool(
            reference_keys == actual_keys
            and all(value.get("bitwise_equal", False) for value in comparisons.values())
        ),
    }


def prefixed_arrays(
    variants: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        f"{variant}__{name}": array
        for variant, arrays in variants.items()
        for name, array in arrays.items()
    }


def read_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {
            key: np.asarray(artifact[key])
            for key in artifact.files
            if key != "metadata_json"
        }


def load_verified_reference(
    artifact_path: Path, report_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify a reference against its host report before comparing arrays."""
    report = json.loads(report_path.read_text())
    if report.get("schema") != SCHEMA:
        raise RuntimeError("Reference report schema does not match this validator")
    identity = report.get("artifact", {})
    if identity.get("bytes") != artifact_path.stat().st_size or identity.get(
        "sha256"
    ) != sha256_file(artifact_path):
        raise RuntimeError("Reference artifact identity does not match its report")

    arrays = read_artifact(artifact_path)
    expected_arrays = identity.get("arrays", {})
    if set(arrays) != set(expected_arrays):
        raise RuntimeError("Reference artifact/report array names differ")
    for key, array in arrays.items():
        expected = expected_arrays[key]
        if (
            expected.get("shape") != list(array.shape)
            or expected.get("dtype") != str(array.dtype)
            or expected.get("sha256") != sha256_array(array)
        ):
            raise RuntimeError(f"Reference array identity mismatch: {key}")

    with np.load(artifact_path, allow_pickle=False) as artifact:
        embedded = json.loads(str(artifact["metadata_json"]))
    for key in ("schema", "inputs", "source_files", "data_files"):
        if embedded.get(key) != report.get(key):
            raise RuntimeError(f"Reference embedded/report metadata mismatch: {key}")
    return arrays, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without the .npz/.json suffix.",
    )
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        help="Optional Mac NPZ to compare while producing the Ubuntu report.",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        help="Mac JSON that authenticates --reference-artifact.",
    )
    args = parser.parse_args()
    if (args.reference_artifact is None) != (args.reference_report is None):
        parser.error(
            "--reference-artifact and --reference-report must be used together"
        )
    return args


def main() -> int:
    args = parse_args()
    output_prefix = args.output_prefix.resolve()
    artifact_path = output_prefix.with_suffix(".npz")
    report_path = output_prefix.with_suffix(".json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    original_mapping = flux_module._p_to_u
    try:
        variants: dict[str, dict[str, np.ndarray]] = {}
        variant_info: dict[str, dict[str, Any]] = {}
        variants["fastmath"], variant_info["fastmath"] = run_variant(
            "fastmath", original_mapping
        )
        variants["strict"], variant_info["strict"] = run_variant(
            "strict", strict_p_to_u
        )
    finally:
        flux_module._p_to_u = original_mapping

    arrays = prefixed_arrays(variants)
    annotation = (
        f"{cst_timestamp()} ({host_label()}): generated the opt-in trajectory "
        "reproducibility artifact; binary NPZ carries this embedded annotation"
    )
    metadata = {
        "schema": SCHEMA,
        "collaboration_note": annotation,
        "inputs": INPUTS,
        "variants": variant_info,
        "source_files": [file_identity(path) for path in SOURCE_PATHS],
        "data_files": [file_identity(path) for path in DATA_PATHS],
    }
    np.savez_compressed(
        artifact_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )

    report: dict[str, Any] = {
        **metadata,
        "repository": {
            "branch": git_output("branch", "--show-current"),
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
        },
        "runtime": {
            "python": platform.python_version(),
            "few": few.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "artifact": {
            # 2026-09-02 17:46 CST (mac): Permit an external temporary prefix
            # so repeatability tests do not overwrite accepted host evidence.
            "path": display_path(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "arrays": {
                key: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": sha256_array(array),
                }
                for key, array in arrays.items()
            },
        },
        "same_host_fastmath_vs_strict": compare_array_sets(
            variants["fastmath"], variants["strict"]
        ),
    }

    if args.reference_artifact is not None and args.reference_report is not None:
        reference_path = args.reference_artifact.resolve()
        reference_report_path = args.reference_report.resolve()
        # 2026-09-02 17:51 CST (mac): Bind cross-host comparison to the Mac
        # report, embedded metadata, per-array hashes, and matching source/data
        # manifests so stale or partial transfers fail before interpretation.
        reference_arrays, reference_report = load_verified_reference(
            reference_path, reference_report_path
        )
        for key in ("inputs", "data_files"):
            if reference_report[key] != metadata[key]:
                raise RuntimeError(f"Current/reference provenance mismatch: {key}")
        # 2026-09-03 16:13 CST (linux): The first cross-host comparison exposed
        # that NumPy integer indices were not JSON serializable. Authenticate
        # every scientific source and data input exactly, while permitting and
        # recording only this validator's portability-only source difference.
        reference_sources = {
            item["path"]: item for item in reference_report["source_files"]
        }
        current_sources = {item["path"]: item for item in metadata["source_files"]}
        differing_source_paths = sorted(
            path
            for path in reference_sources.keys() | current_sources.keys()
            if reference_sources.get(path) != current_sources.get(path)
        )
        allowed_source_differences = {"validation/trajectory_reproducibility.py"}
        unexpected_source_differences = sorted(
            set(differing_source_paths) - allowed_source_differences
        )
        if unexpected_source_differences:
            raise RuntimeError(
                "Current/reference provenance mismatch: source_files "
                f"{unexpected_source_differences}"
            )
        report["cross_host_reference"] = {
            "artifact": {
                "path": display_path(reference_path),
                "bytes": reference_path.stat().st_size,
                "sha256": sha256_file(reference_path),
            },
            "report": {
                "path": display_path(reference_report_path),
                "bytes": reference_report_path.stat().st_size,
                "sha256": sha256_file(reference_report_path),
            },
            "integrity": "passed",
            "source_provenance": {
                "exact_match": not differing_source_paths,
                "allowed_portability_only_differences": {
                    path: {
                        "reference": reference_sources.get(path),
                        "current": current_sources.get(path),
                    }
                    for path in differing_source_paths
                },
                "scientific_sources_exact": not unexpected_source_differences,
            },
            "comparison": compare_array_sets(reference_arrays, arrays),
        }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
    print(f"artifact={artifact_path}")  # noqa: T201
    print(f"report={report_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
