#!/usr/bin/env python3
"""Sweep the CUDA mixed-compute candidate across representative Kerr regimes.

2026-09-04 17:50 CST (linux): Add a Linux-owned, read-only validation sweep for
the 5x candidate.  The sweep constructs FP64 and mixed models sequentially to
fit the RTX 2080 Ti, checks waveform/mode-selection agreement across physical
regimes, and checks all five amplitude reference points used by FEW's tests.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

import few
from few import get_backend, get_file_manager
from few.utils.mappings.kerrecceq import kerrecceq_forward_map
from few.utils.utility import get_mismatch
from few.waveform import FastKerrEccentricEquatorialFlux

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "collaboration/linux/mixed32_accuracy_sweep.json"
DATA_FILENAME = "ZNAmps_l10_m10_n55_DS2Outer.h5"
POINTWISE_NORMALIZED_LIMIT = 5.0e-10
FLAT_MISMATCH_LIMIT = 1.0e-10

# 2026-09-04 17:50 CST (linux): These are copied from tests/test_amplitudes.py
# so the installed wheel, not the checkout, remains the executable under test.
AMPLITUDE_REFERENCE_POINTS = (
    ((3, 2, 0, 4), (0.94071396, 3.03167504, 0.23666111),
     0.002753336140076555 + 0.004143260615912325j),
    ((2, -2, 0, 5), (0.77593827, 5.44830674, 0.39950367),
     -0.0001975310651632842 + 4.806588136412258e-05j),
    ((2, 2, 0, 0), (0.99090693, 1.81708312, 0.25313127),
     -0.09712303568244392 + 0.0004771647068539275j),
    ((4, -3, 0, 1), (0.87600927, 43.70360964, 0.478125),
     -9.89242565478408e-07 + 1.750469365836169e-05j),
    ((3, 3, 0, 0), (0.36158837, 72.94420732, 0.39375),
     5.869887391976143e-05 + 0.00136010620358342j),
)

# 2026-09-04 17:50 CST (linux): Cases deliberately cover the Schwarzschild
# limit, high spin/eccentricity, retrograde motion, both interpolation regions,
# and a near-separatrix orbit.  Shorter T limits storage; mixed computation is
# local to amplitude/summation, while the trajectory and phases remain FP64.
WAVEFORM_CASES = (
    {
        "name": "nominal",
        "a": 0.6,
        "p0": 8.0,
        "e0": 0.3,
        "xI": 1.0,
        "T": 0.1,
    },
    {
        "name": "schwarzschild_limit",
        "a": 0.0,
        "p0": 10.0,
        "e0": 0.4,
        "xI": 1.0,
        "T": 0.1,
    },
    {
        "name": "high_spin_prograde",
        "a": 0.99,
        "p0": 9.0,
        "e0": 0.2,
        "xI": 1.0,
        "T": 0.1,
    },
    {
        "name": "high_eccentricity",
        "a": 0.5,
        "p0": 12.0,
        "e0": 0.8,
        "xI": 1.0,
        "T": 0.1,
    },
    {
        "name": "retrograde",
        "a": 0.7,
        "p0": 11.0,
        "e0": 0.4,
        "xI": -1.0,
        "T": 0.1,
    },
    {
        "name": "weak_field_region_b",
        "a": 0.36158837,
        "p0": 72.94420732,
        "e0": 0.39375,
        "xI": 1.0,
        "T": 0.1,
    },
    {
        "name": "near_separatrix",
        "a": 0.94071396,
        "p0": 3.03167504,
        "e0": 0.23666111,
        "xI": 1.0,
        "T": 0.001,
    },
)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def command_output(*args: str) -> str:
    result = subprocess.run(
        list(args), check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_array(xp: Any, value: Any) -> np.ndarray:
    return np.asarray(xp.asnumpy(value))


def synchronize(xp: Any) -> None:
    xp.cuda.Stream.null.synchronize()


def selected_modes(xp: Any, waveform: Any) -> np.ndarray:
    return np.column_stack(
        [
            host_array(xp, getattr(waveform, name))
            for name in ("ls", "ms", "ks", "ns")
        ]
    )


def model_kwargs(candidate: bool) -> dict[str, Any]:
    if not candidate:
        return {
            "amplitude_kwargs": {"interpolation_precision": "fp64"},
            "sum_kwargs": {"summation_precision": "fp64"},
        }
    return {
        "amplitude_kwargs": {"interpolation_precision": "mixed32"},
        "sum_kwargs": {"summation_precision": "mixed32_intrinsic_fast"},
    }


def run_model(
    xp: Any, candidate: bool
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()
    started = time.perf_counter()
    waveform = FastKerrEccentricEquatorialFlux(
        **model_kwargs(candidate), force_backend="cuda12x"
    )
    synchronize(xp)
    result: dict[str, Any] = {
        "label": "mixed32_intrinsic_fast" if candidate else "fp64",
        "construction_seconds": time.perf_counter() - started,
        "waveform_cases": [],
        "amplitude_reference_points": [],
    }
    waveforms: dict[str, np.ndarray] = {}
    modes: dict[str, np.ndarray] = {}

    waveform_args = (1.0e6, 1.0e1)
    angles = (np.pi / 3.0, np.pi / 4.0)
    for case in WAVEFORM_CASES:
        entry = {key: value for key, value in case.items()}
        try:
            # 2026-09-04 18:01 CST (linux): Match AmpInterpKerrEccEq's
            # canonicalization for retrograde equatorial motion: encode the
            # orientation in signed spin and map with positive xI.
            _, _, _, _, region = kerrecceq_forward_map(
                case["a"] * case["xI"],
                case["p0"],
                case["e0"],
                1.0,
                return_mask=True,
                kind="amplitude",
            )
            entry["initial_amplitude_region"] = "A" if bool(region[0]) else "B"
            started = time.perf_counter()
            output = waveform(
                *waveform_args,
                case["a"],
                case["p0"],
                case["e0"],
                case["xI"],
                *angles,
                T=case["T"],
                dt=15.0,
                dist=1.0,
            )
            synchronize(xp)
            host_output = host_array(xp, output)
            entry.update(
                {
                    "elapsed_ms": (time.perf_counter() - started) * 1.0e3,
                    "samples": int(host_output.size),
                    "modes_kept": int(waveform.num_modes_kept),
                    "finite": bool(np.all(np.isfinite(host_output))),
                }
            )
            waveforms[case["name"]] = host_output
            modes[case["name"]] = selected_modes(xp, waveform)
        except Exception as error:  # keep the rest of the regime sweep useful
            entry["error"] = f"{type(error).__name__}: {error}"
        result["waveform_cases"].append(entry)

    amplitude = waveform.amplitude_generator
    for lmkn, ape, expected in AMPLITUDE_REFERENCE_POINTS:
        a, p, e = ape
        entry = {"lmkn": list(lmkn), "ape": list(ape)}
        try:
            output = amplitude(a, p, e, 1.0, specific_modes=[lmkn])[lmkn]
            value = complex(host_array(xp, output).item())
            entry.update(
                {
                    "value": [value.real, value.imag],
                    "tabulated_value": [expected.real, expected.imag],
                    "absolute_error_vs_tabulated": float(abs(value - expected)),
                    "tabulated_atol_1e-9_passed": bool(
                        abs(value - expected) <= 1.0e-9
                    ),
                }
            )
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        result["amplitude_reference_points"].append(entry)

    del waveform, amplitude
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()
    return result, waveforms, modes


def numerical_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    delta = actual - reference
    tiny = np.finfo(np.float64).tiny
    scale = max(float(np.max(np.abs(reference))), tiny)
    norm = max(float(np.linalg.norm(reference.ravel())), tiny)
    normalized_max = float(np.max(np.abs(delta))) / scale
    relative_l2 = float(np.linalg.norm(delta.ravel())) / norm
    mismatch = float(max(0.0, get_mismatch(actual, reference)))
    return {
        "normalized_max_abs": normalized_max,
        "relative_l2": relative_l2,
        "flat_weight_mismatch": mismatch,
        "pointwise_limit": POINTWISE_NORMALIZED_LIMIT,
        "pointwise_gate_passed": bool(
            normalized_max <= POINTWISE_NORMALIZED_LIMIT
            and relative_l2 <= POINTWISE_NORMALIZED_LIMIT
        ),
        "mismatch_limit": FLAT_MISMATCH_LIMIT,
        "mismatch_gate_passed": bool(mismatch <= FLAT_MISMATCH_LIMIT),
    }


def compare_results(
    fp64: dict[str, Any],
    candidate: dict[str, Any],
    fp64_waveforms: dict[str, np.ndarray],
    candidate_waveforms: dict[str, np.ndarray],
    fp64_modes: dict[str, np.ndarray],
    candidate_modes: dict[str, np.ndarray],
) -> dict[str, Any]:
    waveform_comparisons = []
    for case in WAVEFORM_CASES:
        name = case["name"]
        entry: dict[str, Any] = {"name": name}
        if name not in fp64_waveforms or name not in candidate_waveforms:
            entry["comparable"] = False
        elif fp64_waveforms[name].shape != candidate_waveforms[name].shape:
            entry.update(
                {
                    "comparable": False,
                    "fp64_shape": list(fp64_waveforms[name].shape),
                    "candidate_shape": list(candidate_waveforms[name].shape),
                }
            )
        else:
            entry.update(
                {
                    "comparable": True,
                    "automatic_mode_selection_equal": bool(
                        np.array_equal(fp64_modes[name], candidate_modes[name])
                    ),
                    "candidate_vs_fp64": numerical_metrics(
                        candidate_waveforms[name], fp64_waveforms[name]
                    ),
                }
            )
        waveform_comparisons.append(entry)

    amplitude_comparisons = []
    for fp64_entry, candidate_entry in zip(
        fp64["amplitude_reference_points"],
        candidate["amplitude_reference_points"],
    ):
        entry: dict[str, Any] = {"lmkn": fp64_entry["lmkn"]}
        if "value" not in fp64_entry or "value" not in candidate_entry:
            entry["comparable"] = False
        else:
            reference = complex(*fp64_entry["value"])
            actual = complex(*candidate_entry["value"])
            absolute = float(abs(actual - reference))
            scale = max(abs(reference), np.finfo(np.float64).tiny)
            entry.update(
                {
                    "comparable": True,
                    "absolute_error_vs_fp64": absolute,
                    "relative_error_vs_fp64": float(absolute / scale),
                }
            )
        amplitude_comparisons.append(entry)

    comparable_waveforms = [
        item for item in waveform_comparisons if item.get("comparable")
    ]
    return {
        "waveforms": waveform_comparisons,
        "amplitudes": amplitude_comparisons,
        "summary": {
            "waveform_cases_requested": len(WAVEFORM_CASES),
            "waveform_cases_comparable": len(comparable_waveforms),
            "all_mode_selections_equal": bool(
                comparable_waveforms
                and all(
                    item["automatic_mode_selection_equal"]
                    for item in comparable_waveforms
                )
            ),
            "all_flat_mismatch_gates_passed": bool(
                comparable_waveforms
                and all(
                    item["candidate_vs_fp64"]["mismatch_gate_passed"]
                    for item in comparable_waveforms
                )
            ),
            "all_strict_pointwise_gates_passed": bool(
                comparable_waveforms
                and all(
                    item["candidate_vs_fp64"]["pointwise_gate_passed"]
                    for item in comparable_waveforms
                )
            ),
        },
    }


def main() -> None:
    backend = get_backend("cuda12x")
    xp = backend.xp
    data_path = Path(get_file_manager().get_file(DATA_FILENAME)).resolve()
    fp64, fp64_waveforms, fp64_modes = run_model(xp, candidate=False)
    candidate, candidate_waveforms, candidate_modes = run_model(xp, candidate=True)
    comparisons = compare_results(
        fp64,
        candidate,
        fp64_waveforms,
        candidate_waveforms,
        fp64_modes,
        candidate_modes,
    )
    report = {
        "metadata": {
            "generated_at_cst": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "script": str(Path(__file__).resolve()),
            "git_branch": git_output("branch", "--show-current"),
            "git_head": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short"),
            "few_version": few.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "nvidia_smi": command_output(
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ),
            "data_path": str(data_path),
            "data_size_bytes": data_path.stat().st_size,
            "data_sha256": sha256_file(data_path),
        },
        "threshold_interpretation": {
            "strict_pointwise": (
                "Inherited FP64-regression threshold; failure prevents claiming "
                "drop-in numerical equivalence."
            ),
            "flat_weight_mismatch": (
                "Diagnostic unweighted overlap only; passing is not a substitute "
                "for detector-noise-weighted scientific validation."
            ),
        },
        "fp64": fp64,
        "candidate": candidate,
        "comparisons": comparisons,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons["summary"], indent=2))
    print(f"report={OUTPUT_PATH}")


if __name__ == "__main__":
    import sys

    main()
