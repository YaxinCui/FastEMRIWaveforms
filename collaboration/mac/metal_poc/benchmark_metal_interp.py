#!/usr/bin/env python3
"""Benchmark the isolated FEW FP32 Metal interpolation proof of concept.

2026-09-01 22:24 CST (mac): Add a dependency-free driver that compiles the
Objective-C++ bridge outside the repository and compares actual Kerr H5 slices
against FEW's Apple Accelerate/GCD FP64 implementation. It never changes the
installed FEW backend or writes a result file.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from few import get_backend
from few.utils.mappings.kerrecceq import kerrecceq_forward_map

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("metal_interp.mm")
DEFAULT_DATA = PROJECT_ROOT / "src/few/data/ZNAmps_l10_m10_n55_DS2Outer.h5"
# 2026-09-01 22:27 CST (mac): Preserve the H5 dataset's capital A/B spelling;
# str.title() lowercases the trailing region letter and cannot name these keys.
COEFFICIENT_DATASETS = {"regionA": "CoeffsRegionA", "regionB": "CoeffsRegionB"}
INPUTS = {
    "a": 0.7,
    "p": np.asarray([8.0, 10.0, 12.0, 14.0]),
    "e": np.asarray([0.1, 0.3, 0.5, 0.7]),
    "xI": np.ones(4),
}


def build_library() -> Path:
    """Compile into a temporary directory, keeping binaries out of Git."""
    output = Path(tempfile.gettempdir()) / "few_metal_interp_poc.dylib"
    command = [
        "clang++",
        "-std=c++17",
        "-O3",
        "-fobjc-arc",
        "-dynamiclib",
        str(SOURCE),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "Accelerate",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return output


def load_api(path: Path) -> ctypes.CDLL:
    api = ctypes.CDLL(str(path))
    double_pointer = ctypes.POINTER(ctypes.c_double)
    api.few_metal_last_error.restype = ctypes.c_char_p
    api.few_metal_context_create.restype = ctypes.c_void_p
    api.few_metal_context_destroy.argtypes = [ctypes.c_void_p]
    api.few_metal_device_name.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    api.few_metal_device_name.restype = ctypes.c_int
    api.few_metal_thread_execution_width.argtypes = [ctypes.c_void_p]
    api.few_metal_thread_execution_width.restype = ctypes.c_size_t
    api.few_metal_max_threads.argtypes = [ctypes.c_void_p]
    api.few_metal_max_threads.restype = ctypes.c_size_t
    api.few_metal_plan_create.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        ctypes.c_uint32,
        double_pointer,
        ctypes.c_uint32,
        double_pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    api.few_metal_plan_create.restype = ctypes.c_void_p
    api.few_metal_plan_destroy.argtypes = [ctypes.c_void_p]
    api.few_metal_plan_evaluate.argtypes = [
        ctypes.c_void_p,
        double_pointer,
        double_pointer,
        ctypes.c_uint32,
        double_pointer,
        double_pointer,
    ]
    api.few_metal_plan_evaluate.restype = ctypes.c_int
    # 2026-09-01 22:31 CST (mac): Compare the direct FP32 basis kernel with a
    # host-FP64 prepared-basis kernel through the same persistent Metal plan.
    api.few_metal_plan_evaluate_prepared.argtypes = api.few_metal_plan_evaluate.argtypes
    api.few_metal_plan_evaluate_prepared.restype = ctypes.c_int
    # 2026-09-01 22:32 CST (mac): Measure a two-float precision variant using
    # the same persistent plan and actual Kerr coefficients.
    api.few_metal_plan_evaluate_double_single.argtypes = (
        api.few_metal_plan_evaluate.argtypes
    )
    api.few_metal_plan_evaluate_double_single.restype = ctypes.c_int
    return api


def pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class MetalContext:
    def __init__(self, api: ctypes.CDLL):
        self.api = api
        start = time.perf_counter()
        self.handle = api.few_metal_context_create()
        self.compile_seconds = time.perf_counter() - start
        if not self.handle:
            raise RuntimeError(api.few_metal_last_error().decode())

    def close(self) -> None:
        if self.handle:
            self.api.few_metal_context_destroy(self.handle)
            self.handle = None

    def metadata(self) -> dict[str, object]:
        name = ctypes.create_string_buffer(256)
        if self.api.few_metal_device_name(self.handle, name, len(name)) != 0:
            raise RuntimeError(self.api.few_metal_last_error().decode())
        return {
            "name": name.value.decode(),
            "pipeline_compile_seconds": self.compile_seconds,
            "thread_execution_width": self.api.few_metal_thread_execution_width(
                self.handle
            ),
            "max_threads_per_threadgroup": self.api.few_metal_max_threads(self.handle),
        }

    def __enter__(self) -> MetalContext:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class MetalPlan:
    def __init__(
        self,
        context: MetalContext,
        tx: np.ndarray,
        ty: np.ndarray,
        coefficients: np.ndarray,
    ):
        self.context = context
        self.tx = np.ascontiguousarray(tx, dtype=np.float64)
        self.ty = np.ascontiguousarray(ty, dtype=np.float64)
        self.coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
        self.num_grids = self.coefficients.shape[0] * self.coefficients.shape[1]
        self.coefficients_per_grid = self.coefficients.shape[2]
        start = time.perf_counter()
        self.handle = context.api.few_metal_plan_create(
            context.handle,
            pointer(self.tx),
            self.tx.size,
            pointer(self.ty),
            self.ty.size,
            pointer(self.coefficients.ravel()),
            self.num_grids,
            self.coefficients_per_grid,
        )
        self.upload_seconds = time.perf_counter() - start
        if not self.handle:
            raise RuntimeError(context.api.few_metal_last_error().decode())

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        variant: str = "direct",
    ) -> tuple[np.ndarray, float, float]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        output = np.empty(self.num_grids * x.size, dtype=np.float64)
        gpu_seconds = ctypes.c_double()
        start = time.perf_counter()
        functions = {
            "direct": self.context.api.few_metal_plan_evaluate,
            "prepared": self.context.api.few_metal_plan_evaluate_prepared,
            "double_single": self.context.api.few_metal_plan_evaluate_double_single,
        }
        function = functions[variant]
        status = function(
            self.handle,
            pointer(x),
            pointer(y),
            x.size,
            pointer(output),
            ctypes.byref(gpu_seconds),
        )
        wall_seconds = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(self.context.api.few_metal_last_error().decode())
        return output, wall_seconds, gpu_seconds.value

    def close(self) -> None:
        if self.handle:
            self.context.api.few_metal_plan_destroy(self.handle)
            self.handle = None


def cpu_evaluate(
    tx: np.ndarray,
    ty: np.ndarray,
    coefficients: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    backend = get_backend("cpu")
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    num_grids = coefficients.shape[0] * coefficients.shape[1]
    output = np.zeros(num_grids * x.size, dtype=np.float64)
    start = time.perf_counter()
    backend.interp2D(
        output,
        tx,
        tx.size,
        ty,
        ty.size,
        coefficients.ravel(),
        3,
        3,
        x,
        x.size,
        y,
        y.size,
        num_grids,
        coefficients.shape[2],
    )
    return output, time.perf_counter() - start


def to_complex(output: np.ndarray, point_count: int) -> np.ndarray:
    return (
        output.reshape(-1, 2, point_count).transpose(2, 1, 0)[:, 0]
        + 1j * (output.reshape(-1, 2, point_count).transpose(2, 1, 0)[:, 1])
    )


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference.ravel())
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).tiny)
    denominator = np.linalg.norm(reference.ravel()) * np.linalg.norm(candidate.ravel())
    overlap = abs(np.vdot(reference.ravel(), candidate.ravel())) / denominator
    return {
        "max_absolute": float(np.max(np.abs(difference))),
        "normalized_max": float(np.max(np.abs(difference)) / scale),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / reference_norm),
        "phase_optimized_vector_mismatch": float(max(0.0, 1.0 - overlap)),
    }


def mapped_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.full(INPUTS["p"].size, INPUTS["a"])
    u, w, _y, z, region_a = kerrecceq_forward_map(
        a,
        INPUTS["p"],
        INPUTS["e"],
        INPUTS["xI"],
        return_mask=True,
        kind="amplitude",
    )
    return np.asarray(w), np.asarray(u), np.asarray(z), np.asarray(region_a)


def run_slice(
    context: MetalContext,
    h5: h5py.File,
    region: str,
    z_index: int,
    x: np.ndarray,
    y: np.ndarray,
    repetitions: int,
) -> tuple[np.ndarray, dict[str, object]]:
    group = h5[region]
    coefficients = np.ascontiguousarray(group[COEFFICIENT_DATASETS[region]][z_index])
    tx = np.ascontiguousarray(group["w_knots"][:])
    ty = np.ascontiguousarray(group["u_knots"][:])

    cpu_runs = [
        cpu_evaluate(tx, ty, coefficients, x, y) for _ in range(repetitions + 1)
    ]
    reference = cpu_runs[-1][0]
    cpu_times = [run[1] for run in cpu_runs]
    quantized = coefficients.astype(np.float32).astype(np.float64)
    quantized_reference, quantized_cpu_seconds = cpu_evaluate(tx, ty, quantized, x, y)
    del quantized

    plan = MetalPlan(context, tx, ty, coefficients)
    del coefficients
    metal_outputs = []
    wall_times = []
    gpu_times = []
    prepared_outputs = []
    prepared_wall_times = []
    prepared_gpu_times = []
    double_single_outputs = []
    double_single_wall_times = []
    double_single_gpu_times = []
    try:
        for _ in range(repetitions + 1):
            output, wall_seconds, gpu_seconds = plan.evaluate(x, y)
            metal_outputs.append(output)
            wall_times.append(wall_seconds)
            gpu_times.append(gpu_seconds)
            prepared_output, prepared_wall, prepared_gpu = plan.evaluate(
                x, y, variant="prepared"
            )
            prepared_outputs.append(prepared_output)
            prepared_wall_times.append(prepared_wall)
            prepared_gpu_times.append(prepared_gpu)
            ds_output, ds_wall, ds_gpu = plan.evaluate(x, y, variant="double_single")
            double_single_outputs.append(ds_output)
            double_single_wall_times.append(ds_wall)
            double_single_gpu_times.append(ds_gpu)
    finally:
        plan.close()

    metal_output = metal_outputs[-1]
    prepared_output = prepared_outputs[-1]
    double_single_output = double_single_outputs[-1]
    cpu_warm = statistics.median(cpu_times[1:])
    result = {
        "region": region,
        "z_index": z_index,
        "points": int(x.size),
        "grids_real_plus_imag": int(reference.size // x.size),
        "coefficient_upload_seconds": plan.upload_seconds,
        "cpu_fp64_first_seconds": cpu_times[0],
        "cpu_fp64_warm_median_seconds": cpu_warm,
        "cpu_quantized_coefficients_seconds": quantized_cpu_seconds,
        "metal_first_wall_seconds": wall_times[0],
        "metal_warm_wall_median_seconds": statistics.median(wall_times[1:]),
        "metal_first_gpu_seconds": gpu_times[0],
        "metal_warm_gpu_median_seconds": statistics.median(gpu_times[1:]),
        "cpu_over_metal_warm_speedup": cpu_warm / statistics.median(wall_times[1:]),
        "cpu_over_prepared_metal_warm_speedup": cpu_warm
        / statistics.median(prepared_wall_times[1:]),
        "coefficient_quantization_only": error_metrics(reference, quantized_reference),
        "metal_total_fp32": error_metrics(reference, metal_output),
        "metal_repeat_bitwise": all(
            np.array_equal(metal_outputs[1], value) for value in metal_outputs[2:]
        ),
        "prepared_metal_first_wall_seconds": prepared_wall_times[0],
        "prepared_metal_warm_wall_median_seconds": statistics.median(
            prepared_wall_times[1:]
        ),
        "prepared_metal_first_gpu_seconds": prepared_gpu_times[0],
        "prepared_metal_warm_gpu_median_seconds": statistics.median(
            prepared_gpu_times[1:]
        ),
        "prepared_metal_total_fp32": error_metrics(reference, prepared_output),
        "prepared_metal_repeat_bitwise": all(
            np.array_equal(prepared_outputs[1], value) for value in prepared_outputs[2:]
        ),
        "double_single_first_wall_seconds": double_single_wall_times[0],
        "double_single_warm_wall_median_seconds": statistics.median(
            double_single_wall_times[1:]
        ),
        "double_single_first_gpu_seconds": double_single_gpu_times[0],
        "double_single_warm_gpu_median_seconds": statistics.median(
            double_single_gpu_times[1:]
        ),
        "cpu_over_double_single_warm_speedup": cpu_warm
        / statistics.median(double_single_wall_times[1:]),
        "double_single_errors": error_metrics(reference, double_single_output),
        "double_single_repeat_bitwise": all(
            np.array_equal(double_single_outputs[1], value)
            for value in double_single_outputs[2:]
        ),
    }
    return to_complex(double_single_output, x.size), {
        "reference": to_complex(reference, x.size),
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This proof of concept requires Apple Silicon macOS")
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")

    library = build_library()
    api = load_api(library)
    w, u, z, region_a_mask = mapped_coordinates()
    report: dict[str, object] = {
        "collaboration_note": (
            "2026-09-01 22:24 CST (mac): runtime-generated read-only Metal "
            "feasibility measurements; no production backend changes"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "data": str(args.data),
        "inputs": {key: np.asarray(value).tolist() for key, value in INPUTS.items()},
        "mapped": {
            "w": w.tolist(),
            "u": u.tolist(),
            "z": z.tolist(),
            "region_a_mask": region_a_mask.tolist(),
        },
        "slices": [],
    }

    with MetalContext(api) as context, h5py.File(args.data, "r") as h5:
        report["metal"] = context.metadata()
        z_knots = np.asarray(h5["regionA/z_knots"][:])
        above = int(np.flatnonzero(z_knots > z[0])[0])
        below = above - 1
        reference_parts: dict[str, dict[int, np.ndarray]] = {
            "regionA": {},
            "regionB": {},
        }
        metal_parts: dict[str, dict[int, np.ndarray]] = {
            "regionA": {},
            "regionB": {},
        }

        for region, mask in (("regionA", region_a_mask), ("regionB", ~region_a_mask)):
            for z_index in (below, above):
                metal_output, details = run_slice(
                    context,
                    h5,
                    region,
                    z_index,
                    w[mask],
                    u[mask],
                    args.repetitions,
                )
                metal_parts[region][z_index] = metal_output
                reference_parts[region][z_index] = details.pop("reference")
                report["slices"].append(details["result"])

        weight = float((z[0] - z_knots[below]) / (z_knots[above] - z_knots[below]))
        reference = np.empty((w.size, 6993), dtype=np.complex128)
        metal = np.empty_like(reference)
        for region, mask in (("regionA", region_a_mask), ("regionB", ~region_a_mask)):
            reference[mask] = reference_parts[region][below] + weight * (
                reference_parts[region][above] - reference_parts[region][below]
            )
            metal[mask] = metal_parts[region][below] + weight * (
                metal_parts[region][above] - metal_parts[region][below]
            )
        report["combined_full_kerr_amplitudes"] = {
            "shape": list(reference.shape),
            "z_linear_weight": weight,
            "errors": error_metrics(reference, metal),
            "reference_max_absolute": float(np.max(np.abs(reference))),
        }

    # 2026-09-01 22:54 CST (mac): Keep machine-readable CLI results on stdout
    # while following the project's no-debug-print lint rule.
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
