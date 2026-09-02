# FEW Apple GPU interpolation proof of concept

<!-- 2026-09-01 22:24 CST (mac): Document the isolated native-Metal experiment,
its non-production boundary, reproducible command, and temporary artifact policy. -->

This directory tests one bounded question: can the M3 Pro GPU accelerate FEW's
actual Kerr bicubic amplitude interpolation enough to justify an opt-in hybrid
backend, despite Metal's lack of public FP64 arithmetic?

The Objective-C++ bridge compiles its Metal shader at runtime because this Mac
has Command Line Tools and the Metal framework, but not Xcode's offline
`metal`/`metallib` tools. It uses shared-storage `MTLBuffer` objects, converts
the H5 FP64 knots and coefficients to FP32 with Accelerate, retains coefficient
buffers in a plan, dispatches one GPU thread per real/imaginary grid and input
point, and converts the result back to FP64 for comparison.

It is deliberately not part of CMake, the wheel, or FEW's backend registry.
The CPU and CUDA paths are unchanged. The compiled dylib is created under the
system temporary directory and must not be committed.

Run from the repository root with the existing Mac virtual environment:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/benchmark_metal_interp.py
```

The driver reads only four spin slices from the ignored 5.09 GB Kerr table. It
reports runtime pipeline compilation, FP64-to-FP32 coefficient upload, GPU
timestamps, synchronized wall time, Apple CPU/GCD time, deterministic repeats,
coefficient-quantization-only error, total Metal FP32 error, and the combined
four-point/6993-mode Kerr amplitude error after spin interpolation.

<!-- 2026-09-01 22:31 CST (mac): Record the prepared-basis variant added after
the direct kernel showed that FP32 coordinate/basis arithmetic dominated pure
coefficient quantization error. -->

The driver evaluates two kernels. The direct variant mirrors CUDA by finding
the spline span and evaluating the cubic basis in every GPU thread. The
prepared variant calculates each point's span and basis once on the CPU in
FP64, converts eight weights to FP32, and leaves only the 16-term coefficient
contraction to the GPU. The latter is the intended hybrid-backend candidate.

<!-- 2026-09-01 22:32 CST (mac): Document the precision-recovery experiment;
it is not native FP64 and doubles FP32 coefficient storage. -->

A third kernel splits every FP64 coefficient and host-computed basis weight
into high/low FP32 parts, then uses `TwoSum`-style addition plus FMA-based
two-float products. Its coefficient storage is 8 bytes per value—the same as
the source FP64 table—so its purpose is to test whether strict numerical
consistency is attainable at useful GPU speed, not to save memory.

<!-- 2026-09-01 22:35 CST (mac): Record the compiler condition required for
error-recovery arithmetic after the default fast-math build invalidated it. -->

Runtime Metal compilation disables fast-math for all three comparison kernels.
This is required because the double-single FMA residual depends on explicit
rounding steps that unsafe algebraic contraction is allowed to remove.

No generated result is written automatically. A reviewed measurement must be
copied into this README or the Mac handoff with a new CST-dated collaboration
comment before synchronization.

<!-- 2026-09-01 22:38 CST (mac): Add the opt-in end-to-end waveform command;
the script restores every replaced holder and leaves the installed backend untouched. -->

After the slice benchmark succeeds, the high-memory end-to-end experiment is:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/benchmark_metal_waveform.py
```

It loads the normal CPU Kerr generator, creates double-single Metal plans only
for the two adjacent spin slices in both interpolation regions, temporarily
replaces those four in-memory holders, and compares synchronized warm waveform
time and flat mismatch before restoring the original objects.

<!-- 2026-09-01 22:43 CST (mac): Document duration scaling added after the
0.001-year waveform established correctness and short-workload speedup. -->

Longer cases use `--duration-years 0.01` or `--duration-years 0.1`; the driver
reports the exact duration and cadence in its JSON output.

<!-- 2026-09-01 22:52 CST (mac): Document the separately gated time-domain
summation experiment and its precision boundary. -->

The long-waveform hotspot experiment is separate:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/benchmark_metal_sum.py \
  --duration-years 0.1
```

It leaves amplitude generation on the existing CPU path. The CPU evaluates
the phase splines in FP64, reduces each fundamental phase modulo `2*pi`, and
splits it into high/low floats. Metal uses precise `sincos`, FP32 amplitude
polynomials, and compensated two-float accumulation. Because the trigonometric
result remains FP32, this experiment is expected to test a fast approximate
summation path, not the strict consistency achieved by the interpolation-only
double-single kernel.

<!-- 2026-09-01 23:23 CST (mac): Document the strict full-chain summation
variant added after component diagnostics showed that FP32 error entered before
the existing compensated accumulator. -->

The strict summation comparison uses:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/benchmark_metal_sum.py \
  --duration-years 1.0 --repetitions 2 --precision ds
```

`metal_sum_ds.mm` splits amplitude-spline coefficients, local time, phases,
and complex Ylms into high/low FP32 pairs. It uses FMA-based double-single
products for the amplitude polynomial and complex arithmetic, implements a
double-single sine/cosine polynomial after range reduction, and retains the
two-float mode accumulator. Safe Metal math is mandatory. The original
`--precision f32` path remains available for direct comparison.

The strict one-year run returned normalized maximum `5.817e-11`, relative L2
`1.676e-11`, bitwise repeatability, and an `8.77x` warm end-to-end speedup over
the Apple FP64 CPU path. Short robustness cases can set `--spin`, `--p0`,
`--e0`, and `--xI0`; tested prograde, retrograde, inner-orbit, and zero-spin
cases all stayed below `7.5e-13` normalized maximum at `0.01` year.

<!-- 2026-09-01 22:50 CST (mac): Document the final combined feasibility run;
both temporary injections are restored before the process reports. -->

The combined upper-bound experiment is:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/benchmark_metal_hybrid.py \
  --duration-years 1.0
```

This measures a real FEW waveform with both Metal candidates enabled. It is an
engineering feasibility result, not a backend acceptance test: the amplitude
path can meet strict FP64 cross-backend tolerances, while the mode-sum path
must still be judged using waveform-level scientific error criteria.

<!-- 2026-09-01 23:23 CST (mac): Add the strict combined command and measured
acceptance result without replacing the original FP32 upper-bound experiment. -->

Use `--sum-precision ds` to combine the strict amplitude and strict summation
paths. For the one-year baseline this returned normalized maximum `5.816e-11`,
relative L2 `1.676e-11`, flat mismatch at numerical zero, and `8.62x` warm
end-to-end speedup. This passes the current `5e-10` engineering regression
gate, but it is still an isolated Mac result rather than production or
cross-host acceptance.

<!-- 2026-09-01 23:58 CST (mac): Document the deterministic five-case artifact
generator added in response to Ubuntu handoff 0120e06c, including its tracked
outputs and explicit large-data exclusion. -->

Generate the exact cross-host comparison payload with:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/generate_strict_metal_reference.py \
  --repetitions 2
```

The generator runs the baseline short and one-year waveforms plus positive-spin
retrograde, inner-orbit, and zero-spin cases. It stores only the unrounded
strict-Metal `complex128` arrays in
`collaboration/mac/strict_metal_ds_reference.npz`; CPU arrays are regenerated
by Ubuntu. `collaboration/mac/strict_metal_ds_report.json` records exact inputs,
array/source/data hashes, safe-math/device/compiler metadata, cold/warm timing,
memory, repeatability, and local Metal-versus-CPU metrics.

The registered 5.09 GB amplitude H5 remains ignored and is never copied into
the artifact. The final NPZ is approximately 33 MB and is intended only as a
bounded cross-host validation payload, not as installed package data.

<!-- 2026-09-02 11:01 CST (mac): Document the follow-up ABI capture that turns
the existing same-input Metal gate into a portable Ubuntu CPU/CUDA test. -->

Freeze the exact strict-Metal summation inputs with:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/generate_strict_metal_frozen_sum.py \
  --repetitions 2
```

The capture wrapper copies the eight prepared arrays plus the five scalar ABI
values before delegating each call to strict Metal. It also captures the raw
internal Metal output, verifies the FEW distance divisor against the returned
physical waveform bitwise, and requires two bitwise-identical captures and
outputs per case. The resulting NPZ is only about 195 KB because the dense
waveforms remain in the existing strict-Metal reference.
