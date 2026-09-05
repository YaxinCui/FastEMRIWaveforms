# Apple GPU / Metal feasibility study

<!-- 2026-09-01 22:52 CST (mac): Record the user-directed deep Apple GPU
investigation, reproducible native-Metal prototypes, measured precision tiers,
and production decision gates for the Linux collaborator and future work. -->

<!-- 2026-09-01 23:23 CST (mac): Supersede the original FP32-only summation
conclusion with the user-directed full-chain double-single follow-up while
retaining both precision tiers and their measurements. -->

## Outcome

Native Metal acceleration is technically viable for FEW on Apple Silicon. It
does not require PyTorch, MPS, MLX, or full Xcode at runtime. Two materially
different precision tiers were demonstrated on an Apple M3 Pro:

1. Bicubic Kerr amplitude interpolation can use high/low FP32
   ("double-single") arithmetic and match the existing FP64 CPU result near
   machine precision while retaining a measured warm speedup.
2. Time-domain mode summation can produce a large long-waveform speedup using
   CPU-FP64 phase preparation plus Metal FP32 trigonometry. Its normalized
   sample error is about `1e-7`, so it is an approximate scientific path even
   though the tested flat mismatch is near machine precision.
3. A subsequent full-chain double-single summation kernel reduced the one-year
   normalized maximum to `5.82e-11` while retaining `8.77x` end-to-end speedup.
   Combined with strict Metal amplitude interpolation, the result was
   `5.82e-11` and `8.62x`, passing the current `5e-10` engineering gate.

The evidence now supports production-oriented development of an opt-in strict
Metal amplitude and summation path. It does not yet support making either
summation tier the default or calling the strict kernel cross-host accepted;
the latter still needs broader Mac validation and Ubuntu reference comparison.

## Reproduction boundary

- Branch: `codex/apple-silicon-dual-host`; no new branch was created.
- Host: Apple M3 Pro, 18-core GPU, 36 GiB unified memory, Metal 4, macOS
  26.5.2 arm64.
- Python: the existing uv-managed CPython 3.12 virtual environment.
- Data: ignored 5,089,095,248-byte Kerr H5 table with SHA256
  `3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`.
- CPU reference: the accepted Apple Accelerate/GCD FP64 backend with
  `VECLIB_MAXIMUM_THREADS=1`.
- Source and drivers: `collaboration/mac/metal_poc/`.
- The dylibs are compiled under the system temporary directory and are not
  linked into FEW. Every injected object/callable is restored in-process.

This Mac has Command Line Tools but not Xcode's offline `metal` and `metallib`
executables. The prototypes therefore use Metal's supported runtime source
compilation API. A production wheel can either retain runtime compilation or
ship an offline metallib when the build host has full Xcode.

## Platform facts that shaped the design

- Apple documents that `MTLStorageModeShared` allocates system memory jointly
  accessible by CPU and GPU and is the default buffer mode on Apple Silicon.
  The prototypes use shared buffers and explicit command-buffer completion:
  [shared storage](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)
  and [Apple GPU storage choice](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus).
- Metal supports compiling a library from an MSL source string with explicit
  compiler options:
  [runtime library compilation](https://developer.apple.com/documentation/metal/mtldevice/makelibrary%28source%3Aoptions%3Acompletionhandler%3A%29).
- Metal's `mathMode` controls optimizations that may violate IEEE-754
  behavior, while `mathFloatingPointFunctions` selects FP32 function variants:
  [math mode](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode)
  and [FP32 math functions](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathfloatingpointfunctions).
- The public MPS data types include FP16/FP32 complex types but not complex
  FP64, and MLX documents GPU `float64` as unsupported. Therefore no framework
  wrapper provides a hidden native-FP64 route:
  [MPS data types](https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype)
  and [MLX data types](https://ml-explore.github.io/mlx/build/html/python/data_types.html).

## Experiment 1: Kerr amplitude interpolation

The source kernel evaluates the same 4-by-4 cubic B-spline coefficient window
as `AmpInterp2D.cu`. Tests used both interpolation regions, both adjacent spin
slices for `a=0.7`, all 6993 modes, and the four accepted broad Kerr points.

Three variants separated the error sources:

| Variant | Method | Representative result |
| --- | --- | --- |
| Direct FP32 | Coefficients, knots, span search, basis, and contraction on GPU | Relative L2 around `1e-7`; fails strict gate |
| Prepared FP32 | CPU FP64 span/basis; FP32 coefficients and 16-term GPU contraction | Relative L2 around `4e-8` to `7e-8`; fails strict gate |
| Prepared double-single | FP64 coefficients/weights split into high/low floats; FMA product residual and `TwoSum`-style accumulation | Combined relative L2 `2.94e-15`, normalized max `2.40e-15` |

The current full-Kerr cross-host amplitude limit is `5e-11`; the double-single
variant passed it by over four orders of magnitude on this workload. Slice
outputs were bitwise repeatable. Depending on region and point count, its warm
synchronized wall time was approximately `3.2x` to `4.2x` faster than the
Apple FP64 kernel.

<!-- 2026-09-01 22:55 CST (mac): Add the post-implementation boundary/knot
stress evidence used to challenge the initial four-point result. -->

A separate fixed-seed stress pass covered Region A and B, spin slices 0, 15,
and 32, and 128 paired coordinates per slice drawn from exact boundaries,
interior knots, and uniform points. All 6993 complex modes were compared. The
worst normalized maximum was `1.03e-14` and the worst relative L2 was
`3.99e-15` across the six cases.

Fast math is a correctness boundary. With Metal's unsafe default arithmetic,
the double-single residual was algebraically invalidated and error stayed near
FP32. Setting `MTLMathModeSafe` restored relative errors around `1e-15`. This
option must be an enforced build/runtime condition and covered by tests.

The double-single coefficient cache uses two floats per source double, so it
does not reduce table storage. A Region-A slice is about 122 MiB and a Region-B
slice about 32 MiB in this representation. Caching both adjacent spin slices
in both regions costs about 308 MiB. Caching all 66 slices would add nearly the
entire 4.85 GiB coefficient payload again and is not recommended; use a lazy
two-to-four-slice LRU keyed by region/spin slice.

## Experiment 2: amplitude-only injection into FEW

Four in-memory `AmpInterp2D` holders were replaced by persistent
double-single plans while trajectory, mode selection, spline construction,
phase evolution, and mode summation remained on the CPU FP64 backend.

| Duration | Samples | CPU warm median | Metal-amplitude median | Speedup | Relative L2 | Flat mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 yr | 2,104 | 18.60 ms | 5.83 ms | 3.19x | `3.89e-15` | `0` |
| 0.01 yr | 21,039 | 41.06 ms | 27.60 ms | 1.49x | `3.16e-15` | `0` |
| 0.1 yr | 210,388 | 292.28 ms | 278.51 ms | 1.05x | `9.50e-15` | `0` |

The diminishing end-to-end gain is expected: sparse amplitude interpolation
grows slowly with duration, while CPU mode summation grows with dense output
length. This established the need to investigate the summation kernel.

## Experiment 3: time-domain mode summation

The tested waveform retained 124 modes. For every dense output sample, the
host evaluates the three phase splines in FP64, reduces each phase modulo
`2*pi`, and splits it into high/low floats. The Metal kernel reconstructs each
integer mode phase, calls precise FP32 `sincos`, evaluates the cubic amplitude
polynomial, applies plus/minus-m spherical harmonics, and performs compensated
two-float accumulation in the original per-sample mode order.

Amplitude generation remained on the existing CPU backend in this experiment:

| Duration | Samples | CPU warm median | Metal-sum median | Speedup | Relative L2 | FEW flat mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 yr | 2,104 | 19.29 ms | 16.66 ms | 1.16x | `1.07e-7` | `5.66e-15` |
| 0.1 yr | 210,388 | 293.30 ms | 37.30 ms | 7.86x | `1.12e-7` | `5.33e-15` |
| 1.0 yr | 2,103,877 | 2.103 s | 0.1870 s | 11.25x | `1.09e-7` | `2.66e-15` |

For the one-year case the GPU command itself took about 27--32 ms, while the
synchronized native call took about 157--165 ms. CPU phase preparation,
shared-buffer allocation/conversion, synchronization, and output reconstruction
are therefore the main remaining costs. Persistent reusable buffers and
parallel phase preparation should improve throughput without changing the
kernel's precision tier.

The normalized waveform regression limit is `5e-10`; the approximately
`2.2e-7` normalized-maximum error fails that strict gate by hundreds of times.
The tiny flat mismatch is encouraging but cannot substitute for a LISA
PSD-weighted, time/phase-optimized study across parameters and SNR. Waveform
accuracy standards depend on the intended detection or inference use case; see
[Lindblom, Owen, and Brown](https://arxiv.org/abs/0809.3844) and
[the improved time-domain standards](https://arxiv.org/abs/1008.1803).

<!-- 2026-09-01 22:55 CST (mac): Add a small cross-parameter robustness check;
it is evidence for continued research, not a replacement for the required
scientific validation grid. -->

A four-case, 0.01-year robustness check retained between 113 and 148 modes and
covered baseline prograde, the corresponding retrograde orbit, an inner
`a=0.6, p=8, e=0.3` orbit, and the `a=0` limit. Relative L2 ranged from
`1.02e-7` to `1.36e-7`, normalized maximum from `1.65e-7` to `2.54e-7`, and
FEW flat mismatch from `5.0e-15` to `1.05e-14`. The stable error scale supports
further study but is far too small a grid for scientific acceptance.

## Experiment 4: strict full-chain double-single summation

<!-- 2026-09-01 23:23 CST (mac): Record the precision-source isolation, strict
kernel design, duration scaling, robustness scan, and measured performance of
the follow-up Metal summation experiment. -->

A component-level short-waveform diagnostic showed why compensated summation
alone could not improve the FP32 path. With every other stage evaluated in
FP64, rounding only the indicated stage produced these approximate normalized
maximum errors:

| Rounded stage | Normalized maximum |
| --- | ---: |
| Cubic amplitude polynomial in FP32 | `4.54e-8` |
| Final complex amplitude in FP32 | `3.24e-8` |
| Sine/cosine outputs in FP32 | `2.36e-8` |
| Complex Ylms in FP32 | `1.01e-8` |
| All FP32 modal terms, accumulated in FP64 | `5.59e-8` |
| Entire diagnostic formula in FP64 | `6.48e-14` |

The strict kernel therefore carries every modal input and intermediate as a
high/low FP32 pair. It adds FMA-residual multiplication to the existing
`TwoSum` operations, evaluates the cubic amplitude polynomial with
double-single Horner arithmetic, implements double-single sine/cosine
polynomials after phase range reduction, applies complex phase/Ylm products in
double-single arithmetic, and only then accumulates the modal contribution.
The original fast FP32 kernel remains separately selectable.

With amplitude generation left on the CPU, the strict sum produced:

| Duration | Samples | CPU warm median | Strict Metal median | Speedup | Relative L2 | Normalized maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 yr | 2,104 | 19.48 ms | 18.86 ms | 1.03x | `2.72e-14` | `4.62e-14` |
| 0.1 yr | 210,388 | 290.62 ms | 39.16 ms | 7.42x | `1.68e-12` | `6.40e-12` |
| 1.0 yr | 2,103,877 | 2.1233 s | 0.24220 s | 8.77x | `1.68e-11` | `5.82e-11` |

The one-year GPU command took approximately 83 ms. The remaining approximately
130 ms inside the synchronized native call was host phase preparation, buffer
management, and output reconstruction, so persistent buffers still have useful
headroom. Strict output was bitwise repeatable, CPU output before and after the
temporary injection remained bitwise identical, and FEW flat mismatch was at
numerical zero.

Three additional 0.01-year strict cases covered a positive-spin retrograde
orbit (`148` modes), the inner `a=0.6, p=8, e=0.3` orbit (`113` modes), and the
`a=0` limit (`136` modes). Their normalized maxima were respectively
`6.37e-13`, `7.44e-13`, and `4.81e-13`; relative L2 stayed below `2.49e-13`.
This is evidence that the precision recovery is structural, but it remains too
small a parameter grid for final acceptance.

## Combined upper-bound result

With both candidates temporarily enabled for the one-year waveform:

- output: 2,103,877 complex samples, 124 retained modes;
- CPU median: `2.0964 s`;
- hybrid Metal median: `0.18356 s`;
- warm end-to-end speedup: `11.42x`;
- relative L2: `1.09094e-7`;
- normalized maximum: `2.18326e-7`;
- FEW flat mismatch: `3.66e-15`;
- independent phase-optimized vector mismatch: `1.74e-14`;
- peak process RSS: `6698 MiB`;
- repeated hybrid output was bitwise stable;
- CPU output before and after the temporary injection was bitwise identical.

These timings exclude the approximately 7.25 s full-H5 model load. Runtime
pipeline compilation was about 43 ms total and four amplitude plan uploads
about 45 ms. For a single short waveform, cold setup can erase the GPU gain;
for repeated, batched, or long waveforms it is amortized.

<!-- 2026-09-01 23:23 CST (mac): Add the strict combined result that supersedes
the FP32 result for cross-backend engineering accuracy, while preserving the
FP32 measurement as the maximum-speed precision tier. -->

Repeating the combined one-year experiment with strict double-single
summation returned:

- CPU median: `2.11885 s`;
- strict hybrid Metal median: `0.245868 s`;
- warm end-to-end speedup: `8.62x`;
- relative L2: `1.67612e-11`;
- normalized maximum: `5.81620e-11`;
- flat and phase-optimized vector mismatch: numerical zero;
- peak process RSS: `6728 MiB`;
- strict hybrid output was bitwise repeatable.

This result passes the current `5e-10` normalized waveform regression gate by
approximately `8.6x`. It does not alter the separate requirement for
PSD-weighted scientific validation or Ubuntu CPU/CUDA comparison.

## Recommended backend architecture

Do not model Metal as a NumPy/CuPy replacement. Keep NumPy as `xp` and expose
native Metal functions only for measured kernels:

1. Add an explicit, non-default `metal` or `metal-experimental` backend whose
   array semantics remain NumPy/host based.
2. Build a macOS-only Objective-C++ extension linked to Metal, Foundation, and
   Accelerate. No PyTorch/MLX runtime dependency is needed.
3. Create one process-wide device, command queue, safe-math libraries, and
   pipelines. Encode the two spin slices/regions into one command buffer and
   synchronize once.
4. Cache double-single amplitude plans lazily by coefficient identity and
   support `mode_indexes` inside the kernel rather than copying/gathering the
   full coefficient grid in Python.
5. Reuse summation buffers by capacity. Preserve CPU-FP64 trajectory, spline,
   and phase preparation; expose the validated full-chain double-single kernel
   as the strict sum and retain the original FP32 kernel only as an explicitly
   approximate speed tier.
6. Keep strict amplitude, strict summation, and approximate FP32 summation
   behind separate capability flags. Never silently relax the FP64 validator.

One existing code detail needs care: the Schwarzschild amplitude path passes
`backend.uses_gpu` to a mapping function that currently interprets GPU as
CuPy. A NumPy-backed Metal backend should use `uses_cupy` there or introduce a
separate Metal feature; otherwise adding the generic GPU flag can select the
wrong array behavior.

## Go / no-go decision

- **Go:** production-oriented, opt-in Metal double-single amplitude and strict
  summation, followed by broader Mac and Ubuntu CPU/CUDA validation.
- **Go as research:** persistent-buffer optimization, the explicit fast-FP32
  sum tier, and a representative LISA scientific mismatch campaign.
- **No-go for default today:** neither sum kernel has completed the required
  parameter grid, packaging work, Ubuntu comparison, and PSD-weighted study;
  the FP32 tier additionally fails the strict normalized waveform gate.
- **No-go:** PyTorch/MPS or MLX as a mandatory FEW dependency; neither resolves
  FEW's precision and backend-interface requirements.

Required next validation includes a spin/eccentricity/separatrix grid,
prograde and retrograde cases, longer trajectories, explicit mode selections,
flat and LISA PSD-weighted overlaps with documented time/phase optimization,
maximum phase/amplitude error, cold/warm/batch timing, and unified-memory peak.

## 2026-09-02 production integration checkpoint

<!-- 2026-09-02 14:10 CST (mac): Supersede the PoC-only status with the
smallest cross-host-validated production boundary and preserve its limits. -->

The strict double-single time-domain sum is now integrated as an explicit
`metal` backend. It retains NumPy host storage and delegates every other FEW
method to the CPU backend, so existing CPU/CUDA automatic preference lists and
array semantics do not change. The Schwarzschild mapping call that previously
treated every GPU as CuPy now tests `uses_cupy` explicitly.

The implementation owns one runtime-compiled safe/precise Metal pipeline per
backend instance through a narrow Objective-C++/Cython ABI. CMake defaults to
building it only on Apple Silicon, rejects `FEW_WITH_METAL=ON` elsewhere, and
omits the entire Objective-C++ graph for `OFF`, CUDA-only plugin, and bare
builds. A standalone arm64 wheel build and isolated wheel import passed; the
wheel contains the compiled extension but no Metal/Cython sources or large H5
table.

Frozen identical-input validation is exact for all five accepted cases. The
full-table public-API run also passes all elementwise/L2/mismatch gates. Its
one-year case has normalized maximum `5.8169724e-11`, relative L2
`1.6761562e-11`, flat mismatch zero, and a measured warm end-to-end speedup of
`9.15x` on the M3 Pro. A separate short Schwarzschild public-API check has
normalized maximum `6.77e-14`.

This checkpoint deliberately does not integrate Metal amplitude
interpolation: persistent H5 slice-plan ownership needs a separate production
lifecycle, while summation already captures the dominant long-waveform gain.
It also does not make Metal automatic, accelerate frequency-domain summation,
reuse per-call buffers, or claim a completed LISA PSD/parameter-bias study.
Ubuntu still needs to accept the new build/registry boundary with Metal
disabled while rerunning CPU/CUDA regression checks.
