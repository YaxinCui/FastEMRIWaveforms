#!/usr/bin/env python3
"""Reproduce low-cost cross-layer FEW failure-contract findings.

2026-09-04 18:50 CST (linux): Add a Linux-owned diagnostic that exercises
validation, policy, exception-state, preprocessing, and test-structure issues
without loading the high-memory Kerr amplitude table or modifying FEW state on
disk.  Findings are observations, not fixes or scientific-impact estimates.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

import few
from few.trajectory.integrate import get_integrator
from few.trajectory.ode import PN5
from few.utils.baseclasses import KerrEccentricEquatorial
from few.utils.modeselector import ModeSelector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "collaboration/linux/multilayer_failure_probe.json"


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def observe_call(function, *args, **kwargs) -> dict[str, Any]:
    try:
        value = function(*args, **kwargs)
        return {"accepted": True, "result_repr": repr(value)}
    except Exception as error:
        return {
            "accepted": False,
            "exception": type(error).__name__,
            "message": str(error),
        }


def json_safe_float(value: float) -> float | str:
    """Encode non-finite probe inputs without emitting non-standard JSON."""

    # 2026-09-04 18:53 CST (linux): JSON does not standardize NaN/Infinity;
    # preserve their requested semantics as strings and make serialization
    # reject any accidental non-finite result elsewhere in the report.
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def validation_probe() -> dict[str, Any]:
    model = KerrEccentricEquatorial(force_backend="cpu")
    nominal = (1.0e6, 10.0, 0.6, 8.0, 0.3, 1.0)
    cases = {
        "nominal": nominal,
        "nan_m1": (math.nan, *nominal[1:]),
        "nan_m2": (nominal[0], math.nan, *nominal[2:]),
        "equal_masses": (10.0, 10.0, *nominal[2:]),
        "negative_m2": (nominal[0], -10.0, *nominal[2:]),
    }
    return {
        name: observe_call(model.sanity_check_init, *parameters)
        for name, parameters in cases.items()
    }


class DummyAmplitude:
    """Small structural stand-in for ModeSelector construction."""

    def __init__(self) -> None:
        self.l_arr_no_mask = np.asarray([2], dtype=np.int32)
        self.m_arr_no_mask = np.asarray([2], dtype=np.int32)
        self.k_arr_no_mask = np.asarray([0], dtype=np.int32)
        self.n_arr_no_mask = np.asarray([0], dtype=np.int32)
        self.unique_l = np.asarray([2], dtype=np.int32)
        self.unique_m = np.asarray([2], dtype=np.int32)
        self.inverse_lm = np.asarray([0], dtype=np.int32)
        self.index_map_arr = np.zeros((3, 3, 1, 1), dtype=np.int32)


class DummyYlm:
    pass


def mode_policy_probe() -> dict[str, Any]:
    selector = ModeSelector(
        DummyAmplitude(), ylm_generator=DummyYlm(), force_backend="cpu"
    )
    entries = []
    for threshold in (-0.1, 0.0, 1.0e-5, 1.0, 1.1, math.nan, math.inf):
        try:
            mode, effective_threshold, effective_include, mode_array = (
                selector._set_defaults_and_check_inputs(
                    "threshold", threshold, False
                )
            )
            entries.append(
                {
                    "requested_threshold": json_safe_float(threshold),
                    "accepted": True,
                    "effective_mode_selection": mode,
                    "effective_threshold": json_safe_float(effective_threshold),
                    "requested_include_minus_mkn": False,
                    "effective_include_minus_mkn": effective_include,
                    "mode_array_is_none": mode_array is None,
                }
            )
        except Exception as error:
            entries.append(
                {
                    "requested_threshold": json_safe_float(threshold),
                    "accepted": False,
                    "exception": type(error).__name__,
                    "message": str(error),
                }
            )
    return {"cases": entries}


def exception_state_probe() -> dict[str, Any]:
    integrator = get_integrator(PN5)
    state_before = getattr(integrator, "generating_trajectory", None)
    original_integrate = integrator.integrate

    def injected_failure(*_args, **_kwargs):
        raise RuntimeError("injected integration failure")

    integrator.integrate = injected_failure
    observed = observe_call(
        integrator.run_inspiral,
        1.0e6,
        10.0,
        0.6,
        np.asarray([8.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        np.asarray([0.0]),
        T=0.001,
        dt=15.0,
    )
    state_after = getattr(integrator, "generating_trajectory", None)
    integrator.integrate = original_integrate
    return {
        "state_before": state_before,
        "injected_call": observed,
        "state_after_caught_exception": state_after,
        "expected_for_clean_reuse": False,
        "state_restored_by_run_inspiral": state_after is False,
    }


def preprocessing_probe() -> dict[str, Any]:
    try:
        from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc  # type: ignore

        return {
            "imported": True,
            "symbol_repr": repr(AmpInterpKerrEqEcc),
        }
    except Exception as error:
        return {
            "imported": False,
            "exception": type(error).__name__,
            "message": str(error),
            "source_import": (
                "from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc"
            ),
        }


def test_structure_probe() -> dict[str, Any]:
    path = PROJECT_ROOT / "tests/test_amplitudes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "test_kerrecceq"
    )
    loop = next(node for node in target.body if isinstance(node, ast.For))
    following = target.body[target.body.index(loop) + 1]
    return {
        "test_path": str(path),
        "reference_loop_lines": [loop.lineno, loop.end_lineno],
        "following_statement_type": type(following).__name__,
        "following_statement_line": following.lineno,
        "assertion_is_inside_reference_loop": bool(
            loop.lineno <= following.lineno <= loop.end_lineno
        ),
    }


def main() -> None:
    report = {
        "metadata": {
            "generated_at_cst": datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).isoformat(),
            "git_branch": git_output("branch", "--show-current"),
            "git_head": git_output("rev-parse", "HEAD"),
            "few_version": few.__version__,
            "few_module": str(Path(few.__file__).resolve()),
            "scope": (
                "Low-cost contract probes only; no high-memory table, waveform "
                "accuracy, or scientific impact is tested."
            ),
        },
        "input_validation": validation_probe(),
        "mode_policy": mode_policy_probe(),
        "exception_state": exception_state_probe(),
        "preprocessing_import": preprocessing_probe(),
        "amplitude_test_structure": test_structure_probe(),
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    OUTPUT_PATH.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    print(f"report={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
