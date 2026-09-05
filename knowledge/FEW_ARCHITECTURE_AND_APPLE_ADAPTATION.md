# FEW architecture and Apple-silicon adaptation review

<!-- 2026-09-02 15:17 CST (mac): Record a whole-pipeline architecture, error
budget, data-lifecycle, and acceleration review after the explicit Metal mode-sum
integration; production backend behavior is intentionally unchanged. -->

## Executive conclusion

The current explicit Metal backend is a valid narrow production vertical slice:
it accelerates strict time-domain mode summation on Apple silicon while keeping
NumPy host arrays and leaving the default CPU/CUDA behavior unchanged. Its
same-input error and end-to-end speed have already been measured.

The next largest risks are no longer the Metal summation formula itself:

1. independent adaptive trajectories can diverge across CPU architectures and
   mathematical libraries over a year;
2. the largest Kerr amplitude table is eagerly loaded despite an individual
   evaluation using only one spin slice, or two adjacent slices;
3. the backend abstraction treats array placement, hardware capability, and
   kernel availability as if they were one decision;
4. the Metal path still pays repeated buffer allocation, CPU preprocessing, and
   synchronous completion costs.

Therefore the next work should first localize trajectory divergence and improve
the amplitude-data lifecycle, then optimize the already-correct Metal executor.
Porting every FEW stage wholesale to Metal would hide the source of errors and
may accelerate work that need not be performed.

## Current waveform path

```text
physical parameters
    |
    v
EMRIInspiral / DOP853 trajectory on CPU (adaptive steps, dense output)
    |
    +--> orbital-frequency derivatives and phase spline coefficients
    |
    v
amplitude interpolation from HDF5 coefficients
    |
    +--> Ylm and mode selection
    |
    v
time- or frequency-domain mode summation
    |
    +--> CPU / CUDA / explicit Metal kernel
    |
    v
distance scaling, optional detector response, comparison or likelihood
```

In the present spherical waveform construction, `EMRIInspiral` deliberately
remains a CPU/NumPy operation. Trajectory arrays are converted only after the
integration. `ParallelModuleBase.build_with_same_backend` then propagates one
`force_backend` choice to several downstream modules. The Metal backend borrows
the CPU method bundle and NumPy array namespace, replacing only strict
time-domain waveform summation.

This hybrid is not a defect: Apple unified memory and a narrow kernel boundary
make it a sensible first integration. But the abstraction should eventually
describe the hybrid explicitly.

## Four separate error budgets

### 1. Physical-model error

This includes adiabatic/self-force order, flux interpolation, omitted physics,
mode truncation, and detector-response approximation. The roughly `1e-5`
LISA-weighted mismatch reported for a recent FEW model over much of its domain
belongs here. It is not a GPU rounding tolerance.

### 2. Numerical-algorithm error

This includes adaptive DOP853 tolerances, accepted/rejected step sequence,
dense-output interpolation, phase interpolation, table interpolation, and
finite mode sets. A last-bit difference can move an adaptive comparison across
its branch boundary, changing later knots and accumulating into a visible phase
difference.

### 3. Backend implementation error

This compares the same frozen numerical inputs across CPU, CUDA, and Metal. The
current one-year strict Metal sum has a maximum normalized difference of
`5.816972425119773e-11` and normalized L2 difference of
`1.676156203749705e-11` against its CPU reference, with zero measured flat-noise
mismatch in that run. This is the correct layer for evaluating double-single
arithmetic and Metal kernel ordering.

### 4. Scientific/observational error

Detector-weighted mismatch compares waveform geometry under a LISA noise model,
normally allowing physically irrelevant time and phase offsets. Parameter bias
also depends on signal-to-noise ratio and derivatives in parameter space. It
cannot be replaced by a single maximum absolute or relative sample error.

For a local perturbation

```text
h = A exp(-i phi),        delta h / h ~= delta A / A - i delta phi.
```

Thus a pointwise difference of `1.77e-5` may represent a residual phase error
near `1.77e-5` radians. It does **not** mean an error of `1.77e-5` times the total
millions of accumulated phase radians. For small phase residuals, the aligned
noise-weighted mismatch is controlled approximately by half the weighted
variance of the residual phase, not its largest unaligned sample.

The known `~1.77e-5` Mac/Linux result was produced by two independently computed
long trajectories. Since frozen-input CPU/CUDA/Metal summation agrees far more
tightly, the evidence localizes the larger discrepancy upstream of the Metal
mode-sum kernel.

## Architecture debts exposed by the port

### Backend identity is overloaded

`BackendMethods` combines an array namespace, a large kernel-method bundle, and
an accelerator label. `Backend.Feature.GPU` can describe the Metal backend even
when its public arrays remain NumPy host arrays. Consequently `uses_gpu` does
not reliably answer whether an array is device-resident, whether CUDA is in use,
or whether only one GPU kernel is available.

A future capability model should describe independently:

- array domain: host NumPy, CUDA/CuPy, or another owner;
- accelerator: none, CUDA, or Metal;
- per-kernel availability and supported precision;
- transfer/coherency policy and asynchronous behavior.

The recently added `uses_cupy`, `uses_cuda`, and `uses_metal` predicates improve
call-site clarity, but do not yet replace the monolithic model.

### Kernel routing is global rather than per operation

One backend selection is propagated to amplitude, harmonic, mode-selection, and
summation modules, although each stage has different precision, memory, and
parallelism needs. The long-term target should be a per-kernel router with an
explicit capability query and a safe CPU fallback.

### The native ABI does not describe memory

Raw pointers alone do not state memory location, dtype, length, alignment,
ownership, or lifetime. This is manageable for the present NumPy-to-Metal
boundary, but fragile for a mixed CPU/CUDA/Metal architecture. A typed buffer
descriptor should eventually carry those facts and make invalid combinations
fail before dispatch.

### The Metal executor still contains avoidable overhead

The Metal context already persists the device, queue, and compiled pipeline.
Each waveform evaluation nevertheless allocates thirteen buffers, splits FP64
inputs into high/low FP32 components, prepares phase values on the CPU,
dispatches, blocks with `waitUntilCompleted`, and reconstructs `complex128`
output.

The next optimization candidates are:

1. capacity-based reusable buffers;
2. caching static mode indices and harmonic factors;
3. parallel or fused phase preparation;
4. multiple evaluations per allocation or command buffer;
5. asynchronous execution only after ownership and lifetime are explicit.

GPU counters, occupancy, memory bandwidth, dispatch count, and end-to-end
latency should be recorded together. Kernel-only throughput is insufficient.

## The 5 GB amplitude-data opportunity

The local registered data audit found:

| File/region | Layout | Logical bytes |
| --- | --- | ---: |
| `ZNAmps_l10_m10_n55_DS2Outer.h5`, total | 10 datasets, contiguous, uncompressed | 5,089,087,744 |
| Region A coefficient array | `(33, 6993, 2, 1089)` | 4,020,919,056 |
| Region B coefficient array | `(33, 6993, 2, 289)` | 1,067,075,856 |
| `KerrEccEqFluxData.h5` | 4 contiguous datasets | about 9.86 MB |
| Teukolsky amplitude table | 3,844 contiguous datasets | about 102.88 MB |
| Schwarzschild eccentric input | 43 gzip/chunked datasets | about 12.89 MB |

`AmpInterpKerrEccEq.__init__` currently reads both full coefficient arrays with
`[()]` and constructs holders for all 33 spin grid values in both regions.
However, `get_amplitudes` requires one common spin for the batch and evaluates
only an exact slice or two adjacent spin slices. This implies a high-value,
testable optimization:

1. keep the HDF5 file open under an explicit lifetime owner;
2. lazily read the required one or two spin slices;
3. retain a small thread-safe LRU of decoded/interpolator slices;
4. measure cold start, peak RSS, repeated calls, and concurrency;
5. only then test selected-mode I/O or a new derived chunked layout.

The coefficient datasets are currently contiguous, so hyperslab reads can avoid
loading the complete array but may not be optimal for repeated slice access. A
repacked file must be a new, ignored derived artifact with its own checksum and
numerical comparison. Never mutate or replace the registered source HDF5 file.

There is also a per-call `c[mode_indexes].flatten()` allocation in
`AmpInterp2D`. Static mode-set caching can be evaluated after correctness and
the large lazy-loading benefit are established. Automatic mode selection still
needs broad amplitude information, so it cannot simply assume the final mode
set before computing its selection statistic.

## Target architecture

```text
Scientific policy
  - physical approximant, tolerances, modes, scientific acceptance metric
            |
Numerical plan
  - trajectory method, interpolation, precision, deterministic reference mode
            |
Per-kernel router
  - CPU / Accelerate | CUDA | Metal, each with explicit capabilities
            |
Typed buffers and executors
  - location, dtype, size, ownership, cache, synchronization
            |
Data providers
  - immutable registered data, lazy slices, bounded caches, derived artifacts
```

This split would allow, for example, CPU FP64 DOP853, lazy host amplitude
interpolation, and Metal double-single summation without pretending every array
or module belongs to one homogeneous backend.

For reproducibility, add a trajectory diagnostic mode rather than forcing bitwise
identity on the optimized path. It should record at selected checkpoints:

- accepted and rejected step counts and step sizes;
- state vectors and derivatives;
- orbital frequencies and accumulated phases;
- interpolation knots and phase coefficients;
- hashes plus the first numerical divergence.

Fixed-step or fixed-knot reference runs, tightened tolerances, and controlled
transcendental implementations can then be compared against the adaptive
production integrator. The objective is to understand whether host drift affects
scientific mismatch, not merely to make two files bitwise equal.

## Acceptance ladder

Every new accelerator or numerical optimization should pass increasingly broad
gates:

1. **Kernel contract:** frozen identical inputs, dtype/shape/lifetime checks,
   adversarial cancellation, and CPU reference comparison.
2. **Component:** amplitude, trajectory, interpolation, or summation tests over
   boundaries and held-out sizes.
3. **Waveform:** short and one-year cases, normalized pointwise and L2 errors,
   aligned flat-noise mismatch, and LISA-PSD-weighted mismatch.
4. **Cross-host:** Mac CPU/Metal versus Ubuntu CPU/CUDA from the same frozen
   inputs, followed separately by independent end-to-end construction.
5. **Performance:** cold and warm latency, setup, peak RSS, transfers, kernel
   time, synchronization, and batch throughput.
6. **Packaging:** source build, wheel build, isolated import, backend discovery,
   explicit fallback, and unchanged defaults.

Published physical-model mismatch, backend numerical difference, and
cross-architecture reproducibility must remain separate columns in every report.

## Proposed work order

### P0: freeze and cross-validate the current integration

- Ubuntu builds the pending explicit Metal-aware source tree with Metal disabled
  and verifies that CPU/CUDA defaults and tests are unchanged.
- Preserve the current Mac reports and their checksums as the baseline.

### P1: localize long-trajectory divergence

- Add an opt-in checkpoint tracer around DOP853 and phase construction.
- Compare Mac and Ubuntu first-divergence locations.
- Run a tolerance sweep and a fixed-step/fixed-knot reference experiment.
- Test whether paired/controlled transcendental evaluation, compensated phase
  accumulation, or double-double state is justified at the first sensitive site.

### P1: prototype lazy amplitude data

- Load one/two spin slices with a bounded cache.
- Compare every returned amplitude and waveform against eager loading.
- Benchmark cold time, warm time, RSS, multiple spins, and multiple processes.

### P1: optimize the Metal executor

- Reuse capacity-managed buffers and cache static inputs.
- Profile with Xcode captures and counters.
- Retain the same frozen-input and waveform acceptance gates.

### P2: expand only where measurements justify it

- Evaluate an explicit amplitude-interpolation plan with persistent resources.
- Study the frequency-domain FEW path with Metal-capable FFT/NUFFT technology.
- Add the standard LISA PSD and response configurations to the recurring matrix.
- Investigate reduced-order, relative-binning, or heterodyne techniques at the
  likelihood layer; do not confuse these with raw waveform-kernel acceleration.

This document is an analysis and experiment plan. It intentionally makes no
production-code change during the literature pass.
