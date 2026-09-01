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
