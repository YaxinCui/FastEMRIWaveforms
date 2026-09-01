#!/usr/bin/env python3
"""Validate unrounded strict-Metal waveforms against Linux CPU and CUDA.

2026-09-02 00:08 CST (linux): Add the integrity-bound five-case comparison
requested by the Mac strict-Metal handoff, including flat, vector-phase, and
LPA-noise-weighted engineering metrics.
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
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARTIFACT = {
    "bytes": 33_254_256,
    "sha256": "42bda4811e25f94797048b7168ca55e69a26d74ca8c388374052dc42922a1851",
}
EXPECTED_MAC_REPORT = {
    "bytes": 24_317,
    "sha256": "efe9779425c4a64470b3239f92ee950c12efbd309020a7dfd5c0acd0c5da0086",
}
SCHEMA = 1
SEED = 20260901

# The elementwise limits are the existing same-host Metal-vs-CPU kernel gate.
# They remain visible as diagnostics, but an end-to-end waveform regenerated on
# another CPU architecture also contains trajectory/spline differences and
# therefore cannot isolate that kernel. Cross-host acceptance is mismatch-based.
LOCAL_NORMALIZED_LIMIT = 5.0e-10
LOCAL_RELATIVE_L2_LIMIT = 5.0e-10
FLAT_MISMATCH_LIMIT = 1.0e-10
PHASE_MISMATCH_LIMIT = 1.0e-10
LPA_MISMATCH_LIMIT = 1.0e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def external_file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_identity(
    label: str, actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    failures = []
    for field in ("bytes", "sha256"):
        if actual[field] != expected[field]:
            failures.append(
                f"{field}: expected {expected[field]!r}, found {actual[field]!r}"
            )
    if failures:
        raise RuntimeError(f"{label} identity failed: " + "; ".join(failures))


def resolve_project_path(relative_path: str) -> Path:
    # 2026-09-02 00:10 CST (linux): Validate the manifest's lexical project
    # path but allow the ignored 5.09 GB table to be a project-local symlink to
    # Ubuntu's verified FEW cache, avoiding a duplicate allocation on disk.
    path = PROJECT_ROOT / relative_path
    try:
        path.absolute().relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"path escapes project root: {relative_path}") from error
    return path


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    verified = {}
    failures = []
    for name, expected in manifest.items():
        path = resolve_project_path(expected["path"])
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        verified[name] = actual
        for field in ("bytes", "sha256"):
            if actual[field] != expected[field]:
                failures.append(
                    f"{name} {field}: expected {expected[field]!r}, "
                    f"found {actual[field]!r}"
                )
    if failures:
        raise RuntimeError("manifest verification failed: " + "; ".join(failures))
    return verified


def embedded_case_metadata(case: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "key",
        "inputs",
        "shape",
        "dtype",
        "modes_kept",
        "array_sha256",
    )
    return {field: case[field] for field in fields}


def load_verified_reference(
    artifact_path: Path, mac_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    artifact_identity = file_identity(artifact_path)
    report_identity = file_identity(mac_report_path)
    verify_identity("strict-Metal artifact", artifact_identity, EXPECTED_ARTIFACT)
    verify_identity("strict-Metal report", report_identity, EXPECTED_MAC_REPORT)

    mac_report = json.loads(mac_report_path.read_text(encoding="utf-8"))
    verify_identity(
        "artifact recorded by Mac", artifact_identity, mac_report["artifact"]
    )

    arrays: dict[str, np.ndarray] = {}
    with np.load(artifact_path, allow_pickle=False) as reference:
        embedded = json.loads(str(reference["metadata_json"]))
        expected_keys = {"metadata_json", *(case["key"] for case in embedded["cases"])}
        if set(reference.files) != expected_keys:
            raise RuntimeError(
                "artifact keys differ: "
                f"expected {sorted(expected_keys)}, found {sorted(reference.files)}"
            )
        for case in embedded["cases"]:
            key = case["key"]
            array = np.asarray(reference[key])
            failures = []
            if list(array.shape) != case["shape"]:
                failures.append(f"shape {array.shape!r} != {case['shape']!r}")
            if str(array.dtype) != case["dtype"] or array.dtype != np.complex128:
                failures.append(f"dtype {array.dtype!r} != complex128")
            if not np.all(np.isfinite(array)):
                failures.append("non-finite values")
            actual_hash = sha256_array(array)
            if actual_hash != case["array_sha256"]:
                failures.append(f"SHA256 {actual_hash} != {case['array_sha256']}")
            if failures:
                raise RuntimeError(
                    f"artifact array {key} failed: " + "; ".join(failures)
                )
            arrays[key] = array

    report_cases = [embedded_case_metadata(case) for case in mac_report["cases"]]
    metadata_checks = {
        "schema": embedded.get("schema") == mac_report.get("schema") == SCHEMA,
        "seed": embedded.get("seed") == mac_report.get("seed") == SEED,
        "repository": embedded.get("repository") == mac_report.get("repository"),
        "source_files": embedded.get("source_files") == mac_report.get("source_files"),
        "data_files": embedded.get("data_files") == mac_report.get("data_files"),
        "cases": embedded.get("cases") == report_cases,
    }
    if not all(metadata_checks.values()):
        raise RuntimeError(f"Mac artifact/report metadata mismatch: {metadata_checks}")
    return embedded, mac_report, arrays


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


def memory_snapshot(backend: Any) -> dict[str, float]:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = (
        peak_rss / (1024 * 1024) if sys.platform == "darwin" else peak_rss / 1024
    )
    snapshot = {"peak_process_rss_mib": float(peak_rss_mib)}
    if backend.uses_cupy:
        pool = backend.xp.get_default_memory_pool()
        snapshot.update(
            {
                "cupy_pool_used_mib": float(pool.used_bytes() / (1024 * 1024)),
                "cupy_pool_total_mib": float(pool.total_bytes() / (1024 * 1024)),
            }
        )
    return snapshot


def waveform_call(generator: Any, inputs: dict[str, float]) -> Any:
    args = (
        inputs["M"],
        inputs["mu"],
        inputs["a"],
        inputs["p0"],
        inputs["e0"],
        inputs["xI0"],
        inputs["theta"],
        inputs["phi"],
    )
    kwargs = {
        "dist": inputs["dist"],
        "Phi_phi0": inputs["Phi_phi0"],
        "Phi_theta0": inputs["Phi_theta0"],
        "Phi_r0": inputs["Phi_r0"],
        "T": inputs["T"],
        "dt": inputs["dt"],
    }
    return generator(*args, **kwargs)


def phase_optimized_vector_mismatch(expected: np.ndarray, actual: np.ndarray) -> float:
    denominator = np.sqrt(
        float(np.vdot(expected, expected).real) * float(np.vdot(actual, actual).real)
    )
    overlap = float(abs(np.vdot(expected, actual)) / denominator)
    return float(max(0.0, 1.0 - min(overlap, 1.0)))


def load_lpa_asd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, names=True)
    frequency = np.asarray(table["f"], dtype=np.float64)
    asd = np.asarray(table["ASD"], dtype=np.float64)
    if not (
        np.all(np.isfinite(frequency))
        and np.all(np.isfinite(asd))
        and np.all(frequency > 0.0)
        and np.all(asd > 0.0)
        and np.all(np.diff(frequency) > 0.0)
    ):
        raise RuntimeError("LPA.txt must contain finite, positive, ordered f/ASD")
    return frequency, asd


def lpa_weighted_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    dt: float,
    lpa_frequency: np.ndarray,
    lpa_asd: np.ndarray,
) -> dict[str, Any]:
    """Return a transparent two-sided complex-strain engineering diagnostic."""
    sample_frequency = np.fft.fftfreq(expected.size, d=dt)
    absolute_frequency = np.abs(sample_frequency)
    mask = (absolute_frequency >= lpa_frequency[0]) & (
        absolute_frequency <= lpa_frequency[-1]
    )
    if not np.any(mask):
        raise RuntimeError("waveform has no FFT bins inside the LPA table")

    log_psd = 2.0 * np.log(lpa_asd)
    psd = np.exp(
        np.interp(
            np.log(absolute_frequency[mask]),
            np.log(lpa_frequency),
            log_psd,
        )
    )
    expected_fft = np.fft.fft(expected)
    actual_fft = np.fft.fft(actual)
    expected_band = expected_fft[mask]
    actual_band = actual_fft[mask]
    inverse_psd = 1.0 / psd
    expected_power = float(
        np.sum(np.abs(expected_band) ** 2 * inverse_psd, dtype=np.float64)
    )
    actual_power = float(
        np.sum(np.abs(actual_band) ** 2 * inverse_psd, dtype=np.float64)
    )
    denominator = np.sqrt(expected_power * actual_power)
    cross = np.vdot(expected_band, actual_band * inverse_psd)
    zero_lag_overlap_raw = float(abs(cross) / denominator)
    zero_lag_overlap = min(max(zero_lag_overlap_raw, 0.0), 1.0)
    delta_power = float(
        np.sum(
            np.abs(actual_band - expected_band) ** 2 * inverse_psd,
            dtype=np.float64,
        )
    )

    cross_spectrum = np.zeros(expected.size, dtype=np.complex128)
    cross_spectrum[mask] = np.conj(expected_band) * actual_band * inverse_psd
    correlation = np.fft.ifft(cross_spectrum)
    best_index = int(np.argmax(np.abs(correlation)))
    best_lag_samples = (
        best_index if best_index <= expected.size // 2 else best_index - expected.size
    )
    optimized_overlap_raw = float(
        expected.size * abs(correlation[best_index]) / denominator
    )
    optimized_overlap = min(max(optimized_overlap_raw, 0.0), 1.0)

    return {
        "interpretation": "engineering diagnostic; not a LISA TDI response",
        "strain_convention": "two-sided DFT of complex h_plus-minus-i-h_cross",
        "spectral_convention": "raw rectangular-window DFT; PSD=LPA ASD squared",
        "psd_interpolation": "log-log linear within the tabulated frequency band",
        "optimization": "global complex phase; optional circular discrete time lag",
        "frequency_band_hz": [
            float(absolute_frequency[mask].min()),
            float(absolute_frequency[mask].max()),
        ],
        "frequency_bins": int(np.count_nonzero(mask)),
        "weighted_relative_error": float(np.sqrt(delta_power / expected_power)),
        "zero_lag_phase_optimized_overlap_raw": zero_lag_overlap_raw,
        "zero_lag_phase_optimized_mismatch": float(1.0 - zero_lag_overlap),
        "time_phase_optimized_overlap_raw": optimized_overlap_raw,
        "time_phase_optimized_mismatch": float(1.0 - optimized_overlap),
        "best_circular_lag_samples": best_lag_samples,
        "best_circular_lag_seconds": float(best_lag_samples * dt),
    }


def comparison_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    dt: float,
    lpa_frequency: np.ndarray,
    lpa_asd: np.ndarray,
    *,
    enforce_elementwise: bool = False,
) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "reason": "shape mismatch",
        }
    delta = actual - expected
    reference_scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    reference_norm = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    metrics: dict[str, Any] = {
        "shape": list(actual.shape),
        "max_absolute": float(np.max(np.abs(delta))),
        "normalized_max": float(np.max(np.abs(delta)) / reference_scale),
        "relative_l2": float(np.linalg.norm(delta) / reference_norm),
        "flat_mismatch": float(max(0.0, get_mismatch(expected, actual, use_gpu=False))),
        "phase_optimized_vector_mismatch": phase_optimized_vector_mismatch(
            expected, actual
        ),
        "lpa_weighted": lpa_weighted_metrics(
            expected, actual, dt, lpa_frequency, lpa_asd
        ),
    }
    local_elementwise_gate = {
        "applicable_to_acceptance": enforce_elementwise,
        "reason": (
            "enforced because Linux CPU and CUDA use the same x86_64 trajectory inputs"
            if enforce_elementwise
            else "same-host kernel gate; cross-host regeneration also changes "
            "trajectory integration and spline preparation"
        ),
        "limits": {
            "normalized_max": LOCAL_NORMALIZED_LIMIT,
            "relative_l2": LOCAL_RELATIVE_L2_LIMIT,
        },
        "checks": {
            "normalized_max": metrics["normalized_max"] <= LOCAL_NORMALIZED_LIMIT,
            "relative_l2": metrics["relative_l2"] <= LOCAL_RELATIVE_L2_LIMIT,
        },
    }
    acceptance_limits = {
        "flat_mismatch": FLAT_MISMATCH_LIMIT,
        "phase_optimized_vector_mismatch": PHASE_MISMATCH_LIMIT,
        "lpa_zero_lag_phase_optimized_mismatch": LPA_MISMATCH_LIMIT,
        "lpa_time_phase_optimized_mismatch": LPA_MISMATCH_LIMIT,
    }
    acceptance_checks = {
        "flat_mismatch": metrics["flat_mismatch"] <= FLAT_MISMATCH_LIMIT,
        "phase_optimized_vector_mismatch": (
            metrics["phase_optimized_vector_mismatch"] <= PHASE_MISMATCH_LIMIT
        ),
        "lpa_zero_lag_phase_optimized_mismatch": (
            metrics["lpa_weighted"]["zero_lag_phase_optimized_mismatch"]
            <= LPA_MISMATCH_LIMIT
        ),
        "lpa_time_phase_optimized_mismatch": (
            metrics["lpa_weighted"]["time_phase_optimized_mismatch"]
            <= LPA_MISMATCH_LIMIT
        ),
    }
    metrics.update(
        {
            "local_elementwise_gate": local_elementwise_gate,
            "cross_host_acceptance_limits": acceptance_limits,
            "cross_host_acceptance_checks": acceptance_checks,
            "passed": all(acceptance_checks.values())
            and (
                not enforce_elementwise
                or all(local_elementwise_gate["checks"].values())
            ),
        }
    )
    return metrics


def write_runtime_artifact(
    path: Path,
    backend_name: str,
    embedded: dict[str, Any],
    outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    # 2026-09-02 00:16 CST (linux): Keep the direct Linux CPU-to-CUDA bridge
    # outside Git while embedding enough identity data for the CUDA process to
    # reject stale, reordered, or corrupted arrays.
    cases = []
    for case in embedded["cases"]:
        key = case["key"]
        case_metadata = embedded_case_metadata(case)
        case_metadata["runtime_array_sha256"] = sha256_array(outputs[key])
        cases.append(case_metadata)
    metadata = {
        "collaboration_note": (
            "2026-09-02 00:16 CST (linux): ephemeral same-host CPU/CUDA bridge"
        ),
        "schema": SCHEMA,
        "seed": SEED,
        "backend": backend_name,
        "reference_artifact_sha256": EXPECTED_ARTIFACT["sha256"],
        "repository": embedded["repository"],
        "data_files": embedded["data_files"],
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **outputs,
    )
    return external_file_identity(path)


def load_runtime_peer(
    path: Path, embedded: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = {}
    with np.load(path, allow_pickle=False) as peer:
        metadata = json.loads(str(peer["metadata_json"]))
        expected_keys = {
            "metadata_json",
            *(case["key"] for case in metadata["cases"]),
        }
        if set(peer.files) != expected_keys:
            raise RuntimeError("Linux peer artifact has unexpected keys")
        for case in metadata["cases"]:
            key = case["key"]
            array = np.asarray(peer[key])
            if sha256_array(array) != case["runtime_array_sha256"]:
                raise RuntimeError(f"Linux peer array {key} failed its SHA256")
            arrays[key] = array

    peer_cases = []
    for case in metadata["cases"]:
        without_runtime_hash = dict(case)
        without_runtime_hash.pop("runtime_array_sha256")
        peer_cases.append(without_runtime_hash)
    checks = {
        "schema": metadata.get("schema") == SCHEMA,
        "seed": metadata.get("seed") == SEED,
        "backend": metadata.get("backend") == "cpu",
        "reference_artifact": (
            metadata.get("reference_artifact_sha256") == EXPECTED_ARTIFACT["sha256"]
        ),
        "repository": metadata.get("repository") == embedded["repository"],
        "data_files": metadata.get("data_files") == embedded["data_files"],
        "cases": peer_cases == embedded["cases"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Linux peer metadata failed: {checks}")
    return metadata, arrays


def runtime_metadata(backend_name: str, backend: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "backend": backend_name,
        "few_version": few.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if backend.uses_cupy:
        properties = backend.xp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        metadata["cuda"] = {
            "device": name,
            "runtime_version": int(backend.xp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(backend.xp.cuda.runtime.driverGetVersion()),
            "cupy_version": backend.xp.__version__,
        }
    return metadata


def compare(
    backend_name: str,
    artifact_path: Path,
    mac_report_path: Path,
    output_path: Path,
    runtime_artifact_path: Path | None,
    cpu_peer_path: Path | None,
) -> int:
    embedded, mac_report, references = load_verified_reference(
        artifact_path, mac_report_path
    )
    verified_sources = verify_manifest(embedded["source_files"])
    verified_data = verify_manifest(embedded["data_files"])
    lpa_path = resolve_project_path(embedded["data_files"]["LPA.txt"]["path"])
    lpa_frequency, lpa_asd = load_lpa_asd(lpa_path)

    backend = get_backend(backend_name)
    model_load_start = time.perf_counter()
    generator = FastKerrEccentricEquatorialFlux(
        amplitude_kwargs={"downsample_Z": 1},
        sum_kwargs={"pad_output": False},
        force_backend=backend_name,
    )
    synchronize(backend)
    model_load_seconds = time.perf_counter() - model_load_start

    report: dict[str, Any] = {
        "collaboration_note": (
            "2026-09-02 CST (linux): generated strict-Metal cross-host report; "
            "JSON cannot contain source comments"
        ),
        "schema": SCHEMA,
        "seed": SEED,
        "backend": backend_name,
        "passed": True,
        "reference": {
            "artifact": file_identity(artifact_path),
            "mac_report": file_identity(mac_report_path),
            "repository": embedded["repository"],
        },
        "integrity": {
            "metadata_equal": True,
            "source_files": verified_sources,
            "data_files": verified_data,
            "arrays": {
                case["key"]: {
                    "shape": case["shape"],
                    "dtype": case["dtype"],
                    "sha256": case["array_sha256"],
                }
                for case in embedded["cases"]
            },
        },
        "runtime": runtime_metadata(backend_name, backend),
        "model_load_seconds": model_load_seconds,
        "memory": {"after_model_load": memory_snapshot(backend)},
        "metric_scope": {
            "flat": "FEW get_mismatch with no phase or time maximization",
            "lpa": (
                "two-sided complex-strain engineering diagnostic using LPA.txt; "
                "not a LISA TDI response or parameter-estimation error budget"
            ),
        },
        "cases": [],
    }

    runtime_outputs = {}

    for case in embedded["cases"]:
        key = case["key"]
        first, first_seconds = timed(
            backend, lambda case=case: waveform_call(generator, case["inputs"])
        )
        modes_kept = int(generator.num_modes_kept)
        repeat, repeat_seconds = timed(
            backend, lambda case=case: waveform_call(generator, case["inputs"])
        )
        metrics = comparison_metrics(
            first,
            references[key],
            float(case["inputs"]["dt"]),
            lpa_frequency,
            lpa_asd,
        )
        checks = {
            "finite": bool(np.all(np.isfinite(first))),
            "shape": list(first.shape) == case["shape"],
            "dtype": first.dtype == np.complex128,
            "modes_kept": modes_kept == case["modes_kept"],
            "repeat_bitwise": bool(np.array_equal(first, repeat)),
            "metrics": bool(metrics["passed"]),
        }
        result = {
            "key": key,
            "inputs": case["inputs"],
            "modes_kept": modes_kept,
            "expected_modes_kept": case["modes_kept"],
            "dtype": str(first.dtype),
            "shape": list(first.shape),
            "array_sha256": sha256_array(first),
            "timings_seconds": {
                "first": first_seconds,
                "repeat": repeat_seconds,
            },
            "checks": checks,
            "metrics": metrics,
            "memory_after_case": memory_snapshot(backend),
            "passed": all(checks.values()),
        }
        report["cases"].append(result)
        report["passed"] = report["passed"] and result["passed"]
        runtime_outputs[key] = first

    report["memory"]["final"] = memory_snapshot(backend)
    report["mac_local_metrics"] = {
        case["key"]: case["metal_vs_cpu"] for case in mac_report["cases"]
    }
    if runtime_artifact_path is not None:
        report["ephemeral_runtime_artifact"] = write_runtime_artifact(
            runtime_artifact_path, backend_name, embedded, runtime_outputs
        )
    if cpu_peer_path is not None:
        peer_metadata, peer_arrays = load_runtime_peer(cpu_peer_path, embedded)
        peer_results = {}
        peer_passed = True
        for case in embedded["cases"]:
            key = case["key"]
            peer_metrics = comparison_metrics(
                runtime_outputs[key],
                peer_arrays[key],
                float(case["inputs"]["dt"]),
                lpa_frequency,
                lpa_asd,
                enforce_elementwise=True,
            )
            peer_results[key] = peer_metrics
            peer_passed = peer_passed and peer_metrics["passed"]
        report["linux_cpu_peer"] = {
            "artifact": external_file_identity(cpu_peer_path),
            "metadata": peer_metadata,
            "results": peer_results,
            "passed": peer_passed,
        }
        report["passed"] = report["passed"] and peer_passed
    rendered = json.dumps(report, indent=2, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    sys.stdout.write(rendered + "\n")
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda12x"))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "collaboration/mac/strict_metal_ds_reference.npz",
    )
    parser.add_argument(
        "--mac-report",
        type=Path,
        default=PROJECT_ROOT / "collaboration/mac/strict_metal_ds_report.json",
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--runtime-artifact",
        type=Path,
        help="write an integrity-bound runtime NPZ, normally under /tmp",
    )
    parser.add_argument(
        "--cpu-peer",
        type=Path,
        help="also compare directly against an ephemeral Linux CPU runtime NPZ",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return compare(
        args.backend,
        args.artifact,
        args.mac_report,
        args.report,
        args.runtime_artifact,
        args.cpu_peer,
    )


if __name__ == "__main__":
    sys.exit(main())
