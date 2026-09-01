# macOS handoff log

<!-- 2026-09-01 18:24 CST (mac): Start the first Apple Silicon adaptation session. -->

## 2026-09-01 18:24 CST — adaptation started

- Confirmed host: Apple M3 Pro, macOS 26.5.2, arm64.
- Confirmed branch: `codex/apple-silicon-dual-host`; fork remote is
  `git@github.com:YaxinCui/FastEMRIWaveforms.git`.
- Created `.venv` with uv and CPython 3.12.12 (ignored by Git).
- Baseline build initially exposed an unresolved Homebrew/Fortran runtime
  symbol. Rebuilding with `CMAKE_PREFIX_PATH=/opt/homebrew/opt/lapack` produced
  a working CPU backend.
- Baseline, one BLAS thread, deterministic seed 20260901:
  - real neural layer, `(m,k,n)=(1000,128,128)`: median 0.011440 s;
    maximum absolute error versus NumPy: 0.
  - complex ROM projection, `(m,k,n)=(1000,32,384)`: median 0.008880 s;
    maximum absolute error versus NumPy: `2.781e-17`.
- Current task: replace the macOS-only naive matrix kernels and external
  LAPACKE requirement with Apple Accelerate while preserving CUDA/Linux paths.

## 2026-09-01 18:29 CST — first Accelerate implementation

- Added automatic `FEW_USE_APPLE_ACCELERATE` CMake selection on macOS, with an
  explicit `ON`/`OFF` override for reproducible comparison builds.
- The macOS CPU backend now links only the system Accelerate framework for the
  touched numerical module; `pyinterp` no longer links Homebrew LAPACKE.
- Deterministic regression tests pass for FP64 real GEMM, complex GEMM, and the
  cubic-spline solve against NumPy/SciPy.
- Apple Accelerate results, one BLAS thread, same shapes as the baseline:
  - real neural layer: median 0.0001455 s, maximum absolute error 0;
    approximately 78.6x faster than the original scalar loop.
  - complex ROM projection: median 0.0005763 s, maximum absolute error
    `2.082e-17`; approximately 15.4x faster than the original scalar loop.
- Next: run broader Mac tests and a real FEW waveform benchmark, then prepare a
  deterministic CUDA comparison artifact for the Ubuntu handoff.

## 2026-09-01 18:30 CST — real ROMAN amplitude comparison

- Input: 1000 evenly spaced `(p,e)` points, 21 neural layers, 99 reduced-basis
  complex coefficients, and 3843 output Teukolsky modes.
- Original scalar CPU path: median 1.4013 s over three timed runs.
- Apple Accelerate path: median 0.05558 s over seven timed runs, approximately
  25.2x faster end to end for `RomanAmplitude.get_amplitudes`.
- Accelerate versus original scalar output:
  - maximum absolute difference: `6.7132e-16`;
  - relative L2 difference: `5.9854e-16`;
  - `allclose(rtol=2e-14, atol=2e-14)`: true.
- Rebuilt the environment with the default automatic Accelerate path after the
  comparison. `pyinterp` links the system Accelerate framework and no Homebrew
  LAPACK/gfortran library.

## 2026-09-01 18:45 CST — research pass and Kerr data boundary

<!-- 2026-09-01 18:45 CST (mac): Record the research-driven architecture decision and the removal of an incomplete high-memory download before another host uses this tree. -->

- The primary-source review is recorded in
  `knowledge/APPLE_SILICON_RESEARCH.md`. The production direction is FP64
  Accelerate plus race-free GCD parallelism; Metal remains an experimental
  precision study because current Apple GPU frameworks do not support FEW's
  `float64`/complex-double semantics.
- A sequential A/B run attempted to instantiate the default Kerr eccentric
  amplitude model. Its registry entry is tagged `high_memory`; the server
  reports a 5,089,095,248-byte file. The run was stopped at 18% because this is
  outside the fast validation set and cannot be stored in ordinary GitHub Git.
- Deleted the incomplete 934,281,216-byte
  `src/few/data/ZNAmps_l10_m10_n55_DS2Outer.h5` fragment. It was not a valid HDF5
  validation artifact and is recoverable from the FEW data registry. Full Kerr
  validation needs an explicit large-data transfer/LFS decision.

## 2026-09-01 19:01 CST — deterministic parallelism and validation artifact

<!-- 2026-09-01 19:01 CST (mac): Record the GCD stack fix, final Mac fast-suite result, and binary reference identity for Ubuntu. -->

- The first GCD waveform run exposed a macOS worker-stack overflow: FEW's CPU
  mode workspace was approximately 540 KiB while a dispatch worker stack was
  approximately 512 KiB. The Apple path now uses task-owned heap storage while
  retaining the 5000-mode block and per-sample summation order.
- Sequential CPU versus Accelerate/GCD A/B results:
  - 128-point ROMAN amplitudes: 0.179722 s versus 0.010969 s (about 16.4x),
    relative L2 difference `9.80e-16`;
  - 128-point bicubic amplitudes: 0.066020 s versus 0.016865 s (about 3.9x),
    bitwise identical;
  - 0.01-year Schwarzschild waveform: 0.051205 s versus 0.035055 s (about
    1.46x), flat-weight mismatch 0;
  - short AAK waveform: 0.003352 s versus 0.002394 s (about 1.40x), bitwise
    identical.
- Accelerated warm Schwarzschild medians for `T=0.01`, `0.1`, and `0.25` years
  are 0.034149 s, 0.189133 s, and 0.316489 s. Repeated outputs were bitwise
  deterministic. Relative to the earlier pre-GCD records, the longer cases are
  about 3.1x and 5.1x faster.
- Mac fast suite: 44 tests in 60.417 s, all passed, 18 intentionally skipped
  by `--disable slow --disable high_memory`. A separate non-Accelerate reference
  build also passed the touched targeted tests.
- Added `validation/dual_host_consistency.py` and generated the binary Mac
  reference `collaboration/mac/apple_silicon_reference.npz`:
  - size: 14,997,448 bytes;
  - SHA256: `d86f5099f927a0d9c750c2904fc6531be622abf4f250a324bd21bfa6cc08e64b`;
  - Mac self-comparison passed all six workloads exactly.

## 2026-09-01 19:10 CST — final Accelerate header and reference refresh

<!-- 2026-09-01 19:10 CST (mac): Record the SDK-forward CBLAS update and identify the final Ubuntu reference, which supersedes the 19:01 artifact. -->

- Apple targets now select `ACCELERATE_NEW_LAPACK` without ILP64. This removes
  the macOS 13.3+ deprecated-CBLAS warning while retaining 32-bit dimensions;
  FEW's complex-double buffer is size/alignment checked before the typed CBLAS
  ABI bridge.
- A source-built wheel installed in an independent temporary virtual
  environment, passed the Apple regression tests, and linked all four native
  modules only to Accelerate, libc++, and libSystem.
- The final Mac reference supersedes the 19:01 artifact:
  - size: 14,997,362 bytes;
  - SHA256: `2a596c5dc11d2af7e895fe144876dd6c9846a3bee5a93a046d1f44d969cea2b2`;
  - the six-workload Mac self-comparison is exact, including waveform mismatch
    0.
- 2026-09-01 19:16 CST (mac): Final default-Accelerate fast suite passed all
  44 tests in 60.026 s with 18 tagged skips. The broader suite with only
  `high_memory` disabled passed all 44 tests in 115.473 s with 11 skips,
  including the two-year AAK, detector-frame, one-year frequency-domain, and
  slow-versus-fast checks. The final non-Accelerate build also passed both
  targeted tests and all six cross-path tolerance checks before the virtual
  environment was restored to the default Accelerate build.

## 2026-09-01 19:20 CST — Git-pull data transfer manifest

<!-- 2026-09-01 19:20 CST (mac): Document the ignored binary files force-added for exact Ubuntu reproduction; binary files cannot carry collaboration comments. -->

- The largest required file is 102,881,792 bytes (98.12 MiB), below GitHub's
  ordinary-Git 100 MiB hard limit. It is therefore committed directly so the
  Ubuntu host receives all present validation data with `git pull`; Git LFS is
  neither installed nor required for this transfer.
- Files staged from the public FEW data registry:
  - `AmplitudeVectorNorm.dat`: 49,517 bytes,
    SHA256 `df04fd629353703d140fad80b6877e0b51e0944f1ba60c50412d66df422752e1`;
  - `FluxNewMinusPNScaled_fixed_y_order.dat`: 61,494 bytes,
    SHA256 `c551ace42fbc79febbda132a78868e20900a0a97a06fa4f4d4bf23b12cc1517c`;
  - `SchwarzschildEccentricInput.hdf5`: 12,885,136 bytes,
    SHA256 `edb90365f3a5ee39b92ea1ebeb96c916d059f0835234b768087d9b7e80d1bede`;
  - `Teuk_amps_a0.0_lmax_10_nmax_30_new.h5`: 102,881,792 bytes,
    SHA256 `b3e7d7b215dd2ac472b598e6fee762935dd9909a23a321b894c7b372bc0ca247`;
  - `KerrEccEqFluxData.h5`: 9,857,632 bytes,
    SHA256 `db332e617223a650eb9f890c610e928afd095edea1454b544ad06651c03a5014`.
- The absent 5,089,095,248-byte high-memory Kerr amplitude table remains
  excluded. It exceeds ordinary Git and GitHub Free/Pro's 2 GB LFS per-file
  limit, and is not used by the six-workload handoff validator.

## 2026-09-01 19:27 CST — Mac lock released for user-mediated switch

<!-- 2026-09-01 19:27 CST (mac): Record the accepted remote implementation commit and stop Mac-side editing before Ubuntu acquires the lock. -->

- GitHub accepted implementation/data commit `131c612f` on
  `codex/apple-silicon-dual-host`; the only server notice was the non-blocking
  warning that the 98.12 MiB data file exceeds GitHub's recommended 50 MiB.
- The Mac virtual environment remains a uv-managed CPython 3.12.12 environment
  with the default automatic Accelerate build installed.
- Mac editing is now released. After the user pulls the same branch on Ubuntu,
  the Linux collaborator should first record its hardware/CUDA/Python state and
  acquire `collaboration/LOCK.md`, then follow `validation/README.md` without
  modifying files under `collaboration/mac/`.

## 2026-09-01 20:38 CST — accepted Ubuntu fixes on final Mac build

<!-- 2026-09-01 20:38 CST (mac): Record the post-CUDA Mac rebuild, regression evidence, and final cross-host acceptance. -->

- Fast-forwarded to Ubuntu handoff `5526b336` and reviewed implementation
  commit `3b030762`. Its two source changes explicitly bridge CuPy arrays to
  host-only SciPy/NumPy spline operations and return results to the active
  backend; the CPU branches preserve NumPy behavior.
- Rebuilt the pulled source in the existing uv CPython 3.12.12 `.venv` with
  default automatic Apple Accelerate selection. All four CPU native modules
  link only Accelerate, libc++, and libSystem.
- Direct Apple regression tests passed; the CUDA-only boundary test skipped as
  designed on macOS. The six-workload Mac self-comparison passed exactly,
  including schema/seed/data SHA checks and zero mismatch for both waveforms.
- Final Mac fast suite: 44 tests in 63.881 s, all passed, 19 skipped by the
  `slow`, `high_memory`, or CUDA-only gates.
- Final Mac suite with only `high_memory` disabled: 44 tests in 119.454 s, all
  passed, 12 skipped. This includes two-year AAK, detector-frame, one-year
  frequency-domain, and slow-versus-fast coverage.
- Reverified the Ubuntu report identities:
  - CPU report SHA256
    `bbe313c07f327f4d6b1835b3d7f57217649acb1a3572820433ae61cfd8b50e6f`;
  - CUDA report SHA256
    `2d0a2dd07906d02e0376292f1ba37e5568675801d14b3c69a8b98a1f436ce1a4`.
- Applied only pinned Ruff formatting to Ubuntu's new CUDA regression test and
  recorded that mechanical edit in its module docstring. Ruff checks, format
  checks, Python compilation, and `git diff --check` pass.
- Acceptance: the Apple Accelerate CPU backend, Linux CPU backend, and CUDA
  12.x backend are consistent for all six transferred workloads. The explicit
  remaining scope boundary is the absent 5,089,095,248-byte high-memory Kerr
  amplitude table.
- 2026-09-01 20:38 CST (mac): Final Mac acceptance and regression evidence were
  committed as `b1dce6b2`; the following lock-state commit closes the
  dual-host edit session for remote synchronization.

## 2026-09-01 21:29 CST — full-table Kerr Mac acceptance

<!-- 2026-09-01 21:29 CST (mac): Record the out-of-band Kerr table identity,
full-mode/short-waveform Apple results, upstream fixture caveat, and the exact
small artifact handed to Ubuntu. -->

- Verified the out-of-band file at
  `src/few/data/ZNAmps_l10_m10_n55_DS2Outer.h5` before every run:
  - size: 5,089,095,248 bytes;
  - SHA256: `3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`;
  - Git status: ignored by `src/few/data/.gitignore`; it remains untracked and
    must not be pushed.
- Added `validation/high_memory_kerr_consistency.py`. One shared full-table
  model checks all 6993 modes at four parameter points spanning interpolation
  regions A/B, five targeted mode outputs, and a deterministic 2104-sample
  `FastKerrEccentricEquatorialFlux` waveform.
- Final one-thread Apple Accelerate generation:
  - model load: 7.237 s;
  - first/warm full-mode amplitudes: 0.8591 s / 0.02361 s;
  - first/warm short Kerr waveforms: 6.622 s / 0.03271 s;
  - maximum resident set: 6,630,047,744 bytes (6322.91 MiB);
  - full amplitudes and waveform were bitwise repeatable; repeat waveform
    mismatch was `1.11e-16`.
- The final binary reference is
  `collaboration/mac/high_memory_kerr_reference.npz`:
  - size: 449,418 bytes;
  - SHA256: `b7728a81e2f566d7db503320804234296b1fb1f8d230908b39482171fbc834b3`;
  - Mac self-comparison passed schema/seed/input/data identity and returned
    zero array differences for shapes `(4,6993)`, `(5,)`, and `(2104,)`.
- Direct upstream regression results, each isolated in its own process:
  - `AmplitudesTest.test_kerrecceq`: passed in 5.259 s, peak RSS
    5,604,360,192 bytes;
  - `KerrWaveformTest.test_Kerr_vs_Schwarzchild`: passed in 15.767 s, peak RSS
    6,557,843,456 bytes;
  - `KerrWaveformTest.test_retrograde_orbits`: passed in 15.851 s, peak RSS
    6,544,736,256 bytes.
- The high-memory run exposed a pre-existing upstream fixture defect. Fixture
  index 2 declares `(2,2,0,0)` but its expected value resembles
  `(2,-2,0,5)`; its absolute discrepancy is about `0.408`. The upstream
  assertion is outside the loop and therefore checks only index 4. The new
  validator records but does not enforce index 2, enforces the other four at
  `atol=1e-9`, and still compares all five actual outputs across Mac CPU,
  Ubuntu CPU, and CUDA.

### Ubuntu continuation

1. Pull the synchronized `codex/apple-silicon-dual-host` branch and confirm
   `HEAD` contains this handoff; do not create another branch.
2. Keep the 5.09 GB table out of Git, but make the already verified Ubuntu
   copy available at the exact `src/few/data/` path and confirm its SHA256.
3. Acquire `collaboration/LOCK.md`, then run the two `compare` commands in
   `validation/README.md` for `cpu` and `cuda12x`.
4. Record report hashes, timings, process RSS/CuPy memory, numerical metrics,
   and any failure diagnosis in `collaboration/linux/HANDOFF.md`; do not edit
   files under `collaboration/mac/`.

## 2026-09-01 22:57 CST — native Metal feasibility study

<!-- 2026-09-01 22:57 CST (mac): Hand off the user-directed Apple GPU research,
isolated prototype locations, precision/performance evidence, remaining
scientific gate, validation status, and synchronization boundary. -->

- No production FEW source, backend registry, CMake target, default behavior,
  or validation tolerance changed. Native Objective-C++/Metal prototypes and
  dependency-free drivers are isolated in `collaboration/mac/metal_poc/`;
  compiled dylibs live only in `/tmp`.
- The full analysis is `knowledge/APPLE_GPU_METAL_FEASIBILITY.md`. The Mac has
  only Command Line Tools, but Metal runtime source compilation works without
  PyTorch, MLX, or full Xcode.
- Strict amplitude result: a safe-math double-single kernel splits FP64
  coefficients and CPU-FP64 basis weights into high/low floats. Four accepted
  Kerr points across Regions A/B and all 6993 modes returned normalized max
  `2.40e-15` and relative L2 `2.94e-15`. A six-slice, 128-point boundary/knot
  stress pass had worst normalized max `1.03e-14` and relative L2 `3.99e-15`.
  Warm per-slice synchronized speedup was approximately 3.2x--4.2x.
- Approximate summation result: CPU FP64 evaluates/range-reduces phase splines;
  Metal uses precise FP32 `sincos` and compensated mode accumulation. The
  one-year, 2,103,877-sample, 124-mode run reduced warm end-to-end time from
  `2.103 s` to `0.1870 s` (11.25x), with relative L2 `1.09e-7` and FEW flat
  mismatch `2.66e-15`. Four short parameter cases, including retrograde and
  `a=0`, retained the same error scale.
- Combined one-year result: `2.0964 s` CPU versus `0.18356 s` hybrid Metal
  (11.42x), relative L2 `1.09094e-7`, normalized max `2.18326e-7`, flat
  mismatch `3.66e-15`, peak RSS 6698 MiB, and bitwise repeatability. CPU output
  before/after injection was bitwise identical.
- Decision: proceed later with an opt-in strict Metal amplitude backend. The
  Metal sum is promising research but fails the existing `5e-10` normalized
  waveform gate and needs a LISA PSD/SNR/time-phase-optimized validation grid
  before it can be called scientifically accepted or made default.
- Verification: both `.mm` files compile with Apple clang `-Wall -Wextra`;
  Ruff 0.9.2 check/format and Python compilation pass; the two Apple Accelerate
  unit tests pass. The original accepted CPU/CUDA implementation is untouched.
- No new large artifact exists. The ignored 5.09 GB H5 file is unchanged and
  remains out of Git; the new tracked source/documentation payload is under
  150 KiB.
- These changes are not yet committed or pushed. Before an Ubuntu switch, the
  user must first request/complete the normal Mac commit and GitHub sync; Linux
  should treat this Metal-only directory as read-only research evidence.

## 2026-09-01 23:23 CST — strict Metal mode-sum precision recovery

<!-- 2026-09-01 23:23 CST (mac): Hand off the user-directed follow-up that
isolated the FP32 error sources and demonstrated a full-chain double-single
Metal sum below the existing waveform regression limit. -->

- Added isolated `collaboration/mac/metal_poc/metal_sum_ds.mm`; it is not in
  CMake, the backend registry, or an installed extension. The original FP32
  experiment remains selected by `--precision f32`, while `--precision ds`
  selects the strict kernel through the same temporary ABI and restoration
  checks.
- A short-waveform component diagnostic attributed approximate normalized
  maximum errors of `4.54e-8` to FP32 amplitude-polynomial evaluation,
  `2.36e-8` to FP32 sine/cosine outputs, and `1.01e-8` to FP32 complex Ylms.
  Compensating only the final accumulator therefore could not meet the strict
  gate.
- The strict kernel carries coefficients, local time, phases, amplitudes,
  custom range-reduced sine/cosine, complex products, Ylms, and modal sums as
  high/low FP32 pairs. Metal safe math and explicit FMA residuals are required.
- Summation-only baseline results:
  - 0.001 yr / 2,104 samples: normalized max `4.617e-14`, relative L2
    `2.719e-14`, warm speedup `1.03x`;
  - 0.1 yr / 210,388 samples: normalized max `6.396e-12`, relative L2
    `1.680e-12`, warm speedup `7.42x`;
  - 1.0 yr / 2,103,877 samples: normalized max `5.817e-11`, relative L2
    `1.676e-11`, warm speedup `8.77x`, GPU command approximately `83 ms`.
- Three 0.01-year robustness cases passed: positive-spin retrograde / 148
  modes had normalized max `6.371e-13`; inner `a=0.6, p=8, e=0.3` / 113 modes
  had `7.442e-13`; `a=0` / 136 modes had `4.810e-13`.
- Strict amplitude plus strict sum, one year: CPU `2.11885 s`, Metal
  `0.245868 s`, speedup `8.62x`, normalized max `5.81620e-11`, relative L2
  `1.67612e-11`, numerical-zero flat mismatch, bitwise Metal repeatability,
  and peak RSS `6728 MiB`. This passes the current `5e-10` engineering gate by
  about `8.6x`.
- No production behavior or validation tolerance changed. This is strong Mac
  feasibility evidence, not final acceptance: a broader parameter/mode grid,
  Ubuntu reference comparison, persistent-buffer engineering, and LISA
  PSD-weighted validation remain.
- All additions remain local and uncommitted. Do not switch hosts until the
  user directs a Mac commit/push and the Ubuntu host pulls that exact commit.

## 2026-09-01 23:31 CST — user-directed GitHub handoff

<!-- 2026-09-01 23:31 CST (mac): Record the final synchronization boundary for
the user-directed commit and push of the reviewed native-Metal research set. -->

- This handoff supersedes the earlier local-only synchronization warning: the
  user directed Mac to commit and push the reviewed changes on
  `codex/apple-silicon-dual-host` without creating another branch.
- The synchronization set contains only the collaboration lock/handoff,
  isolated Objective-C++/Metal PoCs and Python drivers, and the two Apple GPU
  research documents. It contains no H5 data, dylib, cache, generated waveform,
  production backend source, CMake change, or validation-tolerance change.
- Before committing, Mac fetched the remote branch and confirmed `HEAD` and
  `origin/codex/apple-silicon-dual-host` were both `f87258e8` with zero
  ahead/behind divergence. The compiled/tested state is recorded in the two
  preceding handoff sections.
- After the push completes, Ubuntu may pull the exact branch, verify the commit
  reported by Git, and treat `collaboration/mac/` as read-only. Production
  integration has not begun; the next dual-host task is broader CPU/CUDA
  reference validation of the strict DS sum before any backend registration.

## 2026-09-01 23:58 CST — strict Metal arrays prepared for Ubuntu

<!-- 2026-09-01 23:58 CST (mac): Hand off the exact five-case unrounded Metal
arrays, structured provenance report, integrity evidence, and Ubuntu comparison
contract requested by Linux commit 0120e06c. -->

- Added the deterministic Mac-only generator
  `collaboration/mac/metal_poc/generate_strict_metal_reference.py`. It records
  the synchronized base commit `0120e06c4050195b06b0791de9fdf09814d16ad2`,
  hashes every participating PoC source, and never modifies a registered
  backend or data file.
- Final Metal-only artifact:
  - path: `collaboration/mac/strict_metal_ds_reference.npz`;
  - size: 33,254,256 bytes;
  - SHA256: `42bda4811e25f94797048b7168ca55e69a26d74ca8c388374052dc42922a1851`;
  - keys: metadata plus baseline short, baseline one year, positive-spin
    retrograde, inner orbit, and zero spin;
  - every waveform is finite, unrounded `complex128`, and its raw array SHA256
    matches both the JSON report and embedded NPZ metadata.
- Structured report:
  - path: `collaboration/mac/strict_metal_ds_report.json`;
  - size: 24,317 bytes;
  - SHA256: `efe9779425c4a64470b3239f92ee950c12efbd309020a7dfd5c0acd0c5da0086`;
  - includes exact inputs, dtype/shape, modes kept, the full amplitude H5,
    trajectory H5, and `LPA.txt` hashes, safe/precise Metal options, Apple
    clang/device metadata, timings, peak RSS, and local CPU error metrics.
- Two complete independent generator processes produced identical raw hashes
  for all five Metal arrays. Within the final process every Metal repeat and
  every CPU before/after comparison was bitwise identical.
- Final local normalized maxima were `4.617e-14` (baseline short),
  `5.81620e-11` (baseline one year), `6.371e-13` (retrograde), `7.340e-13`
  (inner), and `4.827e-13` (zero spin). The one-year warm end-to-end speedup in
  the final artifact run was `8.65x`; all flat mismatches were at or below
  `4.45e-16`.
- `unzip -t`, embedded/report metadata equality, all per-array hash checks,
  finite/dtype/shape checks, Ruff, Python compilation, warning-enabled Apple
  clang compilation, and `git diff --check` pass. The registered 5.09 GB H5,
  temporary dylibs, caches, and CPU arrays remain outside Git.
- Ubuntu should pull the synchronized commit, verify both top-level hashes,
  regenerate all five CPU and CUDA 12.x arrays from the exact report inputs,
  and compute the requested flat and `LPA.txt`-weighted metrics. It must not
  edit `collaboration/mac/`.
- The user explicitly directed handoff, commit, and push. This released-lock
  handoff and both verified artifacts are included in that single Mac commit;
  the pushed Git commit identity is the synchronization boundary for Linux.
