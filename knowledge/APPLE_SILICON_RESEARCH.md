# Apple Silicon acceleration research decision

<!-- 2026-09-01 18:38 CST (mac): Record the broad pre-implementation research pass so the Linux validator can audit the architectural choice and its evidence. -->

Status: engineering decision for the first portable Mac backend, based on
primary documentation, upstream source, and waveform-method papers reviewed on
2026-09-01. This document is an evidence index, not a claim that every candidate
optimization has already been implemented or measured.

## Question

Can FEW use the Apple GPU as a drop-in replacement for its CUDA backend while
retaining the numerical meaning of the existing `double` and complex-double
kernels?

Current answer: no production-quality drop-in route is available. The initial
Mac acceleration path should preserve FP64 on the Apple CPU through Accelerate
and parallelize only demonstrably independent work. A reduced-precision Metal
path can be researched later, but it must remain experimental until it passes
full waveform validation.

## Evidence matrix

| Candidate | FP64 / complex FP64 | CUDA-source reuse | Maturity | Decision |
| --- | --- | --- | --- | --- |
| Accelerate BLAS/LAPACK | Native `double`; complex-double BLAS | Small call-site substitutions | Apple system framework | Production path |
| Accelerate vForce/vDSP | Vectorized double transcendentals, arithmetic, and transforms | Requires data-layout restructuring | Apple system framework | Profile-guided candidate |
| Grand Central Dispatch | Preserves scalar FP64 operations | Wrap independent CPU iterations | Apple system framework | Production candidate with race tests |
| Metal / MPS | Apple JAX and PyTorch MPS reject `float64`; FEW also needs complex-double semantics | Manual kernel/backend rewrite | Mature API, unsuitable dtype | Do not use for the FP64 reference path |
| MLX | NumPy `float64` converts to MLX `float32` by default | Python/model rewrite | Active Apple project | Experimental precision study only |
| AdaptiveCpp Metal | Documents no `double`, no `atomic64`, and no portable-CUDA on Metal | FEW `.cu` is not directly portable to this backend | Metal backend is experimental | Not a current route |
| Software FP64 on Metal | Possible in principle | Full integration and transcendental work required | `metal-float64` is archived and calls itself unfinished | Research reference only |

Primary platform evidence:

- [Accelerate](https://developer.apple.com/documentation/accelerate),
  [BLAS/LAPACK](https://developer.apple.com/documentation/accelerate/blas-library),
  and [BLAS threading](https://developer.apple.com/documentation/accelerate/blas_threading)
  provide Apple-tuned CPU numerical kernels and explicit threading control.
- [vForce](https://developer.apple.com/documentation/accelerate/vforce-library)
  supplies vectorized double-precision trigonometric and transcendental
  functions; [vDSP](https://developer.apple.com/documentation/accelerate/vdsp)
  supplies double-precision vector and Fourier-transform operations.
- [`dispatch_apply_f`](https://developer.apple.com/documentation/dispatch/dispatch_apply_f)
  is appropriate only when each invocation is reentrant and iterations can run
  concurrently. It waits for all work before returning.
- Apple's [JAX Metal page](https://developer.apple.com/metal/jax/) lists
  `float64`, `complex64`, and `complex128` as unsupported. PyTorch's
  [MPS allocation source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/EmptyTensor.cpp)
  likewise rejects double tensors.
- [AdaptiveCpp's Metal documentation](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-metal.md)
  explicitly states that Apple Silicon GPUs lack hardware double precision,
  that its Metal backend has no `double` or 64-bit atomics, and that its
  portable-CUDA dialect is not supported there.
- The archived [`metal-float64`](https://github.com/philipturner/metal-float64)
  project describes itself as unfinished and has incomplete performance and
  precision tables. It cannot be a dependency for a scientific production
  backend.

## Why waveform-level precision is the gate

FEW tracks many harmonics over thousands to millions of cycles. A small local
phase error can accumulate or change an inference result even when a kernel's
elementwise error appears small.

The general accuracy analysis in
[Lindblom, Owen, and Brown](https://arxiv.org/abs/0809.3844) expresses the
parameter-measurement requirement in the detector-noise-weighted inner product:
the model error must satisfy `<delta h | delta h> < 1` for indistinguishability.
Its detection requirement is instead set by the application's allowed mismatch.
[Lindblom's follow-up](https://arxiv.org/abs/0907.0457) cautions against applying
these conditions without the detector and use-case context.

The current Kerr eccentric equatorial FEW work reports model mismatches around
`1e-5` over most of its domain relative to error-free adiabatic waveforms and
uses an SNR-dependent indistinguishability criterion; see
[The Fast and the Frame-Dragging](https://arxiv.org/abs/2506.09470). Therefore:

1. Existing CPU FP64 output is the Mac reference.
2. Ubuntu CUDA FP64 output is the cross-host accelerator reference.
3. Elementwise norms catch implementation mistakes but do not replace a
   detector-noise-weighted, time/phase-optimized mismatch.
4. There is no universal acceptance number for every model and SNR. Validation
   must report source parameters, duration, sample interval, detector PSD,
   optimization convention, overlap/mismatch, and maximum phase/amplitude error.
5. A future FP32/Metal experiment must be compared against the existing model's
   own error budget rather than merely declared "close" with a loose `allclose`.

## FEW hotspot mapping

The FEW architecture compiles several `.cu` sources as both C++ and CUDA. That
helps share formulas but does not make Metal a supported compiler target. The
first-stage mapping is:

| FEW work | CUDA implementation | Safe Mac direction | Research note |
| --- | --- | --- | --- |
| ROMAN dense neural layers | cuBLAS `Dgemm` | Accelerate `cblas_dgemm` | Direct FP64 library substitution |
| Complex ROM projection | cuBLAS `Zgemm` | Accelerate `cblas_zgemm` | Preserve complex-double layout and verify ABI |
| Tridiagonal spline solves | cuSPARSE batched solver | Accelerate LAPACK `dgtsv` | Native system LAPACK avoids an external Fortran runtime |
| Amplitude interpolation | GPU mode parallelism | CPU parallelism by independent mode | Confirm disjoint outputs and tune the grain size |
| Time-domain summation | GPU sample/interval parallelism | CPU parallelism by disjoint spline interval | Keep summation order within each output sample unchanged |
| AAK waveform loops | GPU kernel parallelism | CPU parallelism by independent interval | Trigonometric vectorization is a later profiling target |

The 2025 FEW paper reports, for one four-year accelerated waveform example,
roughly 0.10 s in trajectory, 0.02 s in amplitude, and 0.15 s in time-domain
summation (0.60 s for frequency-domain summation). This is model- and
hardware-specific, but it supports measuring complete module timings instead of
optimizing matrix multiplication alone.

## Implementation gates

The Mac work should proceed in this order:

1. Make the source build self-contained on Apple Silicon using system
   Accelerate while leaving Linux/CUDA behavior unchanged.
2. Replace only exact BLAS/LAPACK equivalents and compare deterministic outputs
   against the scalar CPU implementation at near-FP64 tolerances.
3. Add CPU concurrency only across ranges with provably disjoint writes. Keep
   the floating-point reduction order inside each waveform sample unchanged.
4. Benchmark complete FEW modules and representative short/long waveforms after
   warm-up. Report first-call and warm-call timing separately.
5. Consider vForce/vDSP only where profiling shows a large contiguous batch of
   double-precision transcendentals or vector operations. Avoid temporary-array
   conversion unless its end-to-end benefit is measured.
6. Validate a parameter matrix covering Schwarzschild eccentric, AAK, and Kerr
   eccentric equatorial paths, including high eccentricity and near-separatrix
   cases supported by each model.
7. On Ubuntu, compare Mac artifacts with both CPU and CUDA backends using the
   same input data and checksums. Any mismatch calculation must use the same PSD
   and time/phase optimization convention on both hosts.

## Rejected shortcuts

- Calling an FP32 Metal result "consistent" solely because samples pass a loose
  relative tolerance.
- Treating MPSMatrix, BNNS, MLX, or JAX Metal as hidden FP64 execution paths.
- Adopting an archived or unfinished software-FP64 library as a runtime
  dependency.
- Adding mandatory OpenMP to the default AppleClang build. Upstream
  [issue 85](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/issues/85)
  records an Apple Silicon installation failure caused by `-fopenmp`.
- Assuming a CUDA-to-Metal source translator is scientifically equivalent
  without dtype, complex arithmetic, reduction-order, and waveform tests.

## Decision

Continue the existing Accelerate/GCD direction, then measure and validate it.
Do not begin a production Metal backend in this adaptation. Revisit Metal only
as a separately gated experiment after the FP64 Mac backend and Ubuntu CUDA
consistency suite are complete.

## 2026-09-01 22:52 CST Metal feasibility addendum

<!-- 2026-09-01 22:52 CST (mac): Link the post-acceptance experiment that the
original decision explicitly deferred; the accepted Accelerate reference path
and CPU/CUDA defaults remain unchanged. -->

The prerequisite FP64 Mac/Ubuntu CPU/CUDA acceptance is now complete, and the
user directed a separate native-Metal feasibility study. Its implementation,
measurements, precision tiers, and revised go/no-go decision are recorded in
[`APPLE_GPU_METAL_FEASIBILITY.md`](APPLE_GPU_METAL_FEASIBILITY.md).

The original production decision remains correct for the current default
backend. New evidence supports a future opt-in double-single Metal amplitude
accelerator that meets the strict tested amplitude gate. The tested Metal mode
summation remains approximate and must not become a default until it passes a
scientific PSD/SNR-dependent validation campaign.
