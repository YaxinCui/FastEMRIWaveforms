# Mixed-precision acceleration research plan

<!-- 2026-09-01 23:15 CST (linux): Record the user-proposed FP64/FP32/FP16 hybrid-compute direction, evidence, numerical boundaries, and staged acceptance plan without treating it as an implemented feature. -->

<!-- 2026-09-01 23:37 CST (linux): Reconcile this plan with pulled Mac commit 5c872d31, which demonstrates an isolated strict double-single Metal prototype but does not yet constitute production integration. -->

<!-- 2026-09-03 18:20 CST (linux): Apply the user's clarification that the CUDA goal is not a prescribed FP64+FP32 combination. The governing objective is the fastest validated mix of precision, storage, accumulation, CPU/GPU placement, and fallback supported by the actual hardware. -->

Status: research and implementation proposal. Nothing in this document changes
the production default, which remains the accepted FP64/complex-FP64 backend on
Apple Accelerate CPU, Linux CPU, and CUDA 12.x.

## Governing objective after the 2026-09-03 clarification

The deliverable is **measurable end-to-end acceleration under a predeclared
scientific error budget**, not a mandatory FP64+FP32 implementation. FP64,
FP32, FP16, hardware-supported matrix formats, compensated accumulation,
double-single arithmetic, and CPU/GPU stage splitting are candidates rather
than requirements. Storage precision, compute precision, and accumulation
precision may differ.

The search must be evidence driven:

1. profile the FP64 reference to locate actual time, bandwidth, allocation, and
   synchronization costs;
2. test one precision or execution-boundary change at a time;
3. reject any candidate that has no repeatable end-to-end benefit, even if its
   isolated kernel is faster;
4. reject or add an adaptive high-precision fallback when the candidate crosses
   its waveform/science error budget;
5. select the fastest surviving policy for each model or workload instead of
   forcing one global dtype policy.

The local GPU's supported arithmetic determines which formats can be tried.
Unsupported or nonaccelerated formats must not be emulated merely to satisfy a
named-precision plan. Likewise, kernel fusion, batching, persistent buffers,
cache layout, fewer conversions, and CPU/GPU scheduling are in scope when they
produce a larger validated gain than dtype reduction alone.

## Post-pull feasibility update

Mac commit `5c872d31` arrived after the initial version of this plan was written.
Its isolated native-Metal proof of concept materially advances the Apple side:

- double-single Kerr amplitude interpolation reached normalized maximum error
  `2.40e-15` on the four-point/6993-mode workload, with approximately
  `3.2x`--`4.2x` warm per-slice speedup;
- an approximate FP32 one-year sum reached about `11.42x` but missed the strict
  normalized waveform gate;
- a strict double-single one-year sum reached normalized maximum
  `5.81620e-11`, relative L2 `1.67612e-11`, and `8.62x` end-to-end speedup,
  passing the current `5e-10` engineering gate.

These are feasibility results, not a registered backend. They move the next
Apple milestone from basic Metal prototyping to broader cross-host numerical,
LISA PSD-weighted, persistent-buffer, and production-integration validation.
The separate CUDA `mixed32` coefficient/interpolation experiment remains useful
for determining portable precision boundaries and performance behavior.

## Decision summary

Mixed computation is worth prototyping because the current RTX 2080 Ti is a
Turing GPU with much more FP32 capacity than FP64 capacity and with Tensor
Cores for mixed-precision matrix operations. The production-oriented design
should not cast the whole waveform model to a single lower precision. The
following is a conservative starting hypothesis, not a prescribed final mix:

- FP64 for trajectory integration, orbital frequencies, separatrix/domain
  decisions, time, phase, and final scientific comparisons;
- FP32 for opt-in Kerr coefficient storage and amplitude interpolation, subject
  to waveform-level validation;
- FP16 inputs with FP32 accumulation only for suitable dense real matrix
  multiplications, initially the ROMAN neural layers;
- FP64 or compensated accumulation for phase-sensitive mode summation.

FP16 trajectory or phase evolution is out of scope. FP16 has approximately
`2^-10` relative machine precision, FP32 `2^-23`, and FP64 `2^-52`; an EMRI's
long-duration phase evolution can turn a small local error into a significant
waveform error.

## Primary evidence

- NVIDIA's [Turing tuning guide](https://docs.nvidia.com/cuda/archive/11.8.0/turing-tuning-guide/index.html)
  documents 64 FP32 cores, only 2 FP64 cores, and 8 mixed-precision Tensor
  Cores per Turing SM.
- The [CUDA floating-point guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
  documents the range and precision of FP16, FP32, and FP64 and distinguishes
  general arithmetic from Tensor Core formats.
- The [cuBLAS documentation](https://docs.nvidia.com/cuda/archive/12.8.2/cublas/index.html)
  exposes input and accumulation precision through `cublasGemmEx` and
  `cublasLtMatmul`.
- The [FEW framework paper](https://arxiv.org/abs/2104.04582) describes the
  modular GPU-accelerated waveform architecture and the need to evaluate
  scientific waveform fidelity, not kernel output alone.
- [Model Waveform Accuracy Standards](https://arxiv.org/abs/0809.3844) and its
  [follow-up](https://arxiv.org/abs/0907.0457) show why the final accuracy gate
  must be tied to detector-noise-weighted waveform error and the intended
  detection or parameter-estimation use case.

Current source evidence is also explicit:

- `src/few/cutils/global.h` defines real and complex scalar types as `double`
  and `complex<double>`;
- `src/few/cutils/matmul.cu` calls `cublasDgemm` and `cublasZgemm`;
- `src/few/cutils/AmpInterp2D.cu` stores coordinates, coefficients, and outputs
  as `double`;
- Python/CuPy amplitude and summation buffers use `float64` and `complex128`.

## Accepted FP64 baseline

The full 5,089,095,248-byte Kerr table has already been accepted across all
three FP64 backends. These measurements are the performance and accuracy
baseline, not promised mixed-precision results:

| Backend | Model load | Warm 6993-mode amplitude | Warm 2104-sample waveform | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| Apple M3 Pro Accelerate CPU | 7.237 s | 0.02361 s | 0.03271 s | 6322.91 MiB |
| Ubuntu CPU | 10.669 s | 0.1149 s | 0.2310 s | 6473.93 MiB |
| RTX 2080 Ti CUDA 12.x | 13.421 s | 0.01188 s | 0.01839 s | 6483.68 MiB |

The CUDA run used up to 5202.91 MiB in the CuPy pool. An FP32 coefficient
payload would be approximately half the FP64 payload size before container,
metadata, alignment, or compression effects; FP16 would be approximately one
quarter. This does not mean cold-start runtime will scale by the same ratio:
HDF5 I/O, host model construction, conversion, transfers, and trajectory work
remain important.

The exact reference and reports are:

- `collaboration/mac/high_memory_kerr_reference.npz`;
- `collaboration/linux/high_memory_kerr_cpu.json`;
- `collaboration/linux/high_memory_kerr_cuda12x.json`.

## Proposed precision partition

| Component | First experimental precision | Required high-precision boundary |
| --- | --- | --- |
| Trajectory ODE and flux evolution | FP64 | Always FP64 in the initial experiment |
| Orbital frequencies, separatrix, region masks | FP64 | Decisions must match the reference path |
| Time and accumulated phases | FP64 | Do not reduce precision |
| Kerr coefficient cache | FP32 storage | Preserve original H5 and checksum provenance |
| Kerr bicubic interpolation | FP32 compute candidate | Promote output before phase-sensitive use; allow FP64 fallback |
| ROMAN real neural layers | FP16 input / FP32 accumulation candidate | Retain an FP32 and FP64 comparison path |
| ROMAN complex projection | FP32 candidate | Do not assume complex FP16 Tensor Core support |
| Per-mode amplitude/phase product | FP32 candidate | Phase arguments remain FP64 |
| Mode reduction | FP64, pairwise, or compensated | Keep a deterministic summation policy |
| Final waveform, overlap, mismatch, likelihood | complex128 / FP64 | Always compare in FP64 |

FP16 is not the first Kerr interpolation format. Its range and approximately
`1e-3` relative resolution can erase weak modes, disturb automatic mode
selection, and amplify cancellation. If FP16 coefficients are studied later,
they need per-mode or per-block scaling plus an FP64 fallback and must remain a
separate experimental option.

## Illustrative user-facing modes

- `precision="fp64"`: unchanged default and reference behavior.
- `precision="mixed32"`: FP64 dynamics/phases/final accumulation with FP32
  coefficient storage and selected amplitude kernels.
- `precision="mixed16"`: later opt-in mode, initially limited to dense ROMAN
  matrix multiplication with FP32 accumulation.

These names illustrate an auditable interface; they do not require FP32 or
FP16 to survive profiling and validation. The implementation may instead
select per-stage policies (for example, high-precision trajectory and phase,
lower-precision amplitude kernels, high-precision accumulation), adaptive
fallback, or a hardware-specific policy. An `auto` policy may be considered
only after deterministic explicit policies are reproducible.

The precision mode must be explicit in metadata, reports, cache names, and
reproducibility artifacts. A reduced-precision cache must not impersonate the
registered FP64 H5 file or reuse its checksum.

## Staged implementation

1. Profile the accepted FP64 CUDA path with kernel, memory-bandwidth, transfer,
   and CPU/GPU synchronization timings. Separate cold model construction from
   repeated waveform generation.
2. Rank candidate interventions from the profile: reduced storage precision,
   reduced compute precision, mixed accumulation, kernel fusion, batching,
   persistent buffers, fewer conversions/transfers, or a changed CPU/GPU
   boundary. Record expected benefit and risk before implementation.
3. Implement the highest-value candidate behind an explicit opt-in policy.
   If it changes coefficient storage, stream conversion from the source H5,
   preserve checksum provenance, and keep the registered file untouched.
4. Separate storage, arithmetic, and accumulation experiments. For example,
   compare reduced storage with FP64 interpolation, reduced storage plus
   reduced interpolation, and reduced multiplication with FP32 or FP64
   accumulation. This identifies the source of speed and error.
5. Apply hardware-supported matrix formats only to profiled dense operations
   such as suitable ROMAN layers. Tensor Cores do not automatically accelerate
   irregular interpolation or transcendental kernels, and an unsupported
   format is not a useful target on the local GPU.
6. Use the isolated Mac double-single Metal prototype as feasibility evidence,
   then validate it over the broader CPU/CUDA reference grid before any backend
   registration. Apple GPU support remains a separate implementation, not an
   automatic CUDA dtype switch.
7. Keep every reduced-precision route opt-in until both hosts reproduce its
   numerical and performance claims.

## Validation gates

Every experiment must use the current FP64 artifacts as the oracle and report:

1. exact source-data hashes, precision policy, compiler/backend, GPU, and seed;
2. all 6993 Kerr modes at the existing four region-A/B points and the five
   targeted fixtures, including the known upstream fixture-index-2 caveat;
3. high spin, retrograde, high eccentricity, region-boundary, near-separatrix,
   and cancellation-sensitive cases;
4. fixed-mode and automatic-mode-selection comparisons, because small
   amplitude changes can alter the selected mode set discontinuously;
5. short, medium, and science-duration waveforms rather than only the current
   0.001-year smoke waveform;
6. amplitude maximum/relative-L2 error, maximum accumulated phase error,
   detector-noise-weighted optimized mismatch, and likelihood or parameter-bias
   checks where the use case requires them;
7. repeatability, NaN/Inf checks, cold/warm timing, peak host RSS, GPU pool,
   transfer volume, and energy when measurable.

The existing full-table validator uses normalized limits `5e-11` for Kerr
amplitudes, `5e-10` for its short waveform, and `1e-10` for flat-weight
mismatch. These are the initial strict regression gates. If an experimental
fast mode cannot meet them, any relaxed threshold must be justified from the
detector PSD, source SNR, duration, and intended scientific use; it must not be
chosen merely to make FP16 or FP32 pass.

## Expected outcome and limits

Mixed precision is most likely to help repeated or batched inference workloads,
where the model is warm and GPU-resident. It can reduce coefficient memory and
bandwidth and move appropriate dense matrices onto Turing Tensor Cores. It may
offer only a modest gain for a single cold waveform because current full-table
model loading and first-call work take seconds while warm kernels take
milliseconds.

For Apple Silicon, mixed precision is strategically important because it may
allow the M-series GPU to handle FP32/FP16 amplitude work while the CPU keeps
FP64 dynamics and phases. Unified memory reduces explicit copies but does not
remove dtype conversion, scheduling, synchronization, or scientific-validation
costs.

## Non-goals and stop conditions

- Do not replace the accepted FP64 default during experimentation.
- Do not use FP16 for trajectory integration, time, or phase evolution.
- Do not claim a Tensor Core speedup for non-matrix kernels without profiling.
- Do not judge success from `allclose` alone.
- Do not keep a named precision mode merely because it was proposed; keep only
  policies that improve repeatable end-to-end performance within the error
  budget.
- Stop or add an adaptive FP64 fallback if reduced precision changes domain
  decisions, selected modes, produces unstable long-duration phase error, or
  exceeds the documented waveform error budget.

## Recommended first milestone

Proceed on two explicit experimental tracks. First, validate Mac's strict
double-single amplitude/summation prototype against a broader Ubuntu FP64/CUDA
reference grid and LISA-weighted mismatch. Second, profile the CUDA FP64 path
and implement the single highest-value opt-in candidate, comparing its storage,
compute, accumulation, transfer, and fallback variants independently. FP32
Kerr interpolation is one plausible first candidate, not a requirement. The
milestone succeeds only when at least one policy produces a repeatable,
scientifically accepted end-to-end gain; the winning precision mix is an
experimental result.
