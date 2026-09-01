# Ubuntu CUDA handoff log

<!-- 2026-09-01 19:31 CST (linux): Record the pulled Mac handoff, validation-input integrity, and initial Ubuntu toolchain state before installing or building FEW. -->

## 2026-09-01 19:31 CST — lock acquired and inputs verified

- Confirmed host: Ubuntu x86_64, Linux 7.0.0-30-generic, Intel Core i5-13400F
  with 16 logical CPUs.
- Confirmed GPU: NVIDIA GeForce RTX 2080 Ti, 22,528 MiB reported memory,
  compute capability 7.5, driver 595.84, and driver-reported CUDA 13.2.
- Confirmed branch: `codex/apple-silicon-dual-host`; pulled fork commit
  `92af2f42`, which contains Apple implementation commit `131c612f`.
- Verified the SHA256 values of the final Mac reference artifact and all five
  transferred FEW data files against `collaboration/mac/HANDOFF.md`; every
  value matches.
- Initial shell PATH has no `python`, `cmake`, or `nvcc` command. The next step
  is to locate an existing isolated package/toolchain manager before changing
  the host environment.
- Acquired `collaboration/LOCK.md` for Ubuntu validation after the user
  directed the host switch and pull.

<!-- 2026-09-01 19:40 CST (linux): Record why the release-wheel comparison was rejected and how the current-source build was isolated from the host system. -->

## 2026-09-01 19:40 CST — reproducible current-source environments

- Created an ignored uv `.venv` with CPython 3.12.13. The first comparison used
  PyPI FEW/core CUDA 2.1.0 and is preserved as
  `cpu_pypi210_mismatch.json`; it is intentionally not an acceptance result.
  The release predates Mac's `47e4fea4` reference baseline and differed in
  ROMAN amplitudes by relative L2 `1.110e-3`.
- Installed micromamba 2.9.0 in the user-local tool directory because the host
  has no passwordless sudo. Created the isolated toolchain prefix
  `/home/ubuntu/.local/share/micromamba/envs/few-ubuntu-20260901`.
- The final toolchain is CPython 3.12, CMake 4.4.3, GCC/G++/GFortran 14.4.0,
  CUDA Toolkit/nvcc 12.9.86, conda-forge LAPACKE/OpenBLAS, and CUDA libraries
  scoped to the micromamba prefix. CUDA Toolkit installation downgraded the
  compiler from 15.3 to the compatible conda-forge 14.4 toolchain.
- Built the current branch once with `FEW_WITH_GPU=OFF` and
  `FEW_USE_APPLE_ACCELERATE=OFF`; this confirmed the Apple CMake option resolves
  off on Linux and the Linux CPU wheel builds successfully.
- A no-version conda CuPy solve proposed CUDA 13.3, which exceeds the
  driver-reported CUDA 13.2. The transaction was interrupted before changes
  were linked; `nvcc --version` remained 12.9.86. Runtime CuPy 14.2.0 remains
  isolated in `.venv`.

<!-- 2026-09-01 19:57 CST (linux): Record CUDA blockers found only by the Ubuntu handoff, their minimal fixes, and the evidence-based AAK tolerance. -->

## 2026-09-01 19:57 CST — CUDA boundary fixes and same-source wheel

- The Mac reference exposed two pre-existing CUDA host/device boundary errors
  in the post-2.1.0 core:
  - `RomanAmplitude` passed CuPy `y/e` arrays implicitly to SciPy's host-only
    spline. `src/few/amplitude/romannet.py` now stages those coordinates on the
    host and returns normalization values to the active CUDA device.
  - `SphericalHarmonicWaveformBase` passed a CuPy batch to the NumPy-owned
    trajectory derivative spline. `src/few/waveform/base.py` now evaluates the
    spline with explicit host times and sends frequencies back to the active
    backend.
- Added `tests/test_roman_cuda_normalization.py` to cover both boundaries when
  CUDA 12.x is available and skip cleanly on non-CUDA hosts. Both tests pass on
  the RTX 2080 Ti.
- The PyPI 2.1.0 CUDA plugin cannot be mixed with the current core because the
  current code's added `k`-mode argument changes the native waveform call.
  Built a same-source CPU+CUDA wheel from the working tree with
  `FEW_WITH_GPU=ON`, `FEW_CUDA_ARCH=75`, and
  `FEW_USE_APPLE_ACCELERATE=OFF`. The wheel contains all four CPU and all four
  `few_backend_cuda12x` native modules.
- The AAK CPU/Mac kernel uses FEW's historical Numerical Recipes Bessel
  approximation while CUDA uses libdevice `jn`. The CUDA result is bitwise
  repeatable and differs by relative L2 `1.034e-9`, while flat-weight mismatch
  is `8.882e-16`. The validator therefore gives only AAK a normalized limit of
  `5e-9`; every other numerical limit and the independent waveform-mismatch
  limit remain unchanged.

<!-- 2026-09-01 20:04 CST (linux): Record final dual-host reports, regression-suite results, and the retained lock state. -->

## 2026-09-01 20:04 CST — Ubuntu validation complete

- Final CPU report: `cpu_comparison.json`, 4,406 bytes, SHA256
  `bbe313c07f327f4d6b1835b3d7f57217649acb1a3572820433ae61cfd8b50e6f`.
  All metadata and six numerical workloads pass. Key results:
  - ROMAN relative L2 `1.143e-15`;
  - Schwarzschild waveform relative L2 `5.817e-15`, mismatch `0`;
  - AAK relative L2 `4.981e-16`, mismatch `0`.
- Final CUDA report: `cuda12x_comparison.json`, 4,429 bytes, SHA256
  `2d0a2dd07906d02e0376292f1ba37e5568675801d14b3c69a8b98a1f436ce1a4`.
  All metadata and six numerical workloads pass. Key results:
  - neural-layer output is bitwise identical to Mac;
  - ROMAN relative L2 `4.315e-15`;
  - Schwarzschild waveform relative L2 `9.450e-15`, mismatch `0`;
  - AAK relative L2 `1.034e-9`, mismatch `8.882e-16`.
- Diagnostic release mismatch report: `cpu_pypi210_mismatch.json`, 4,445
  bytes, SHA256
  `ee273de31d553556d48c70c209de01363711b9c08f9ba33409a565415389f152`.
- Ubuntu CPU fast suite: 44 tests in 101.161 seconds, all passed, 21 skipped.
- Ubuntu CUDA-first fast suite: 46 tests in 131.644 seconds, all passed, 20
  skipped. Logs explicitly selected `cuda12x` for GPU-capable test classes.
- `git diff --check` and JSON parsing of all reports pass. A broad Ruff scan
  still reports unrelated pre-existing modernization and unused-import findings
  in touched legacy files; they were not mixed into this validation change.
- The 5,089,095,248-byte high-memory Kerr amplitude table remains excluded, as
  directed by the Mac handoff.
- Ubuntu retains the edit lock pending user direction. Before a Mac switch,
  commit and synchronize these changes, then have Mac rebuild from source and
  rerun its fast suite; the two Linux-only fixes do not change the Mac CPU
  numerical path or invalidate the existing Mac reference artifact.

<!-- 2026-09-01 20:10 CST (linux): Release Ubuntu editing at the user's request and provide the exact Mac reacquisition checklist. -->

## 2026-09-01 20:10 CST — released for Mac verification

- Committed the complete Ubuntu implementation and validation result as
  `3b030762` (`fix: complete dual-host CUDA validation`).
- Ubuntu editing is released. The user must push the existing branch and pull
  it on Mac before the Mac collaborator acquires `collaboration/LOCK.md`.
- Mac should then:
  1. confirm the branch is `codex/apple-silicon-dual-host` and the worktree is
     clean;
  2. record lock acquisition with an Asia/Shanghai timestamp;
  3. rebuild the current source in its existing Python 3.12 `.venv` with the
     default automatic Apple Accelerate selection;
  4. run `tests/test_apple_accelerate.py` and the fast suite with `slow` and
     `high_memory` disabled;
  5. self-compare `collaboration/mac/apple_silicon_reference.npz` with the CPU
     backend and confirm that the Linux-only CUDA bridges did not alter Mac
     output; and
  6. record all results in `collaboration/mac/HANDOFF.md` before any further
     host switch.
- `tests/test_roman_cuda_normalization.py` is expected to skip cleanly on Mac
  because it specifically guards the CUDA 12.x path.

<!-- 2026-09-01 21:06 CST (linux): Record discovery and integrity verification of the full high-memory Kerr table, plus its user-confirmed out-of-band transfer for Mac validation. -->

## 2026-09-01 21:06 CST — full Kerr table handed to Mac

- A read-only search found the complete high-memory Kerr amplitude table in
  Ubuntu's FEW user-data cache, not in the Git worktree:
  `/home/ubuntu/.local/share/few/v2.1.0.post1.dev93+g92af2f42c.d20260901/download/ZNAmps_l10_m10_n55_DS2Outer.h5`.
- Its size is 5,089,095,248 bytes and its SHA256 is
  `3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`,
  exactly matching `src/few/files/registry.yml`.
- The user copied this file out of band with `scp` to the Mac account at
  `cui@macbook:~/ZNAmps_l10_m10_n55_DS2Outer.h5` and confirmed that the
  transfer completed. Ubuntu cannot directly attest to the destination bytes,
  so Mac must run `shasum -a 256` before using the file.
- The originally supplied Mac project destination
  `/Users/cui/Desktop/FastEMRIWaveforms` returned `No such file or directory`
  over SSH. Mac must first locate its actual checkout, then place the verified
  table under that checkout's `src/few/data/`. That directory ignores data by
  default, so the 5.09 GB table must remain untracked and must not be pushed to
  GitHub.
- Mac should pull this handoff, acquire `collaboration/LOCK.md`, verify the
  table, and run the high-memory `AmpInterpKerrEccEq` and
  `FastKerrEccentricEquatorialFlux` acceptance work. It should record peak
  memory/load behavior, deterministic amplitude and short-waveform reference
  results, and release the lock in `collaboration/mac/HANDOFF.md`.
- After Mac publishes its reference artifact and handoff, Ubuntu should pull
  them and run matching current-source CPU/CUDA consistency checks against the
  same full table. No source code or binary data was changed by this Ubuntu
  documentation-only handoff.

<!-- 2026-09-01 21:34 CST (linux): Record lock acquisition and the ignored symlink used to expose the already verified full Kerr table without duplicating 5.09 GB. -->

## 2026-09-01 21:34 CST — full-table Ubuntu acceptance started

- Fast-forwarded the required branch from `fb61073e` to Mac handoff commit
  `f7028a39`, confirmed a clean worktree, and acquired the shared edit lock.
- Created ignored local symlink
  `src/few/data/ZNAmps_l10_m10_n55_DS2Outer.h5` pointing to Ubuntu's existing
  FEW user-cache copy. This avoids a second 5.09 GB allocation on disk; the
  symlink and target remain outside Git.
- Reverified the file through the project path: SHA256
  `3236d8b5eff618242291e9eeb24638dbbbb82fac464ae575dc3f2ba158c54834`.
  `git check-ignore` confirms `src/few/data/.gitignore` excludes it.
- Next actions under this lock are the Mac-prescribed current-source CPU and
  CUDA 12.x comparisons; their reports and resource metrics will be appended
  below before the lock is released.

<!-- 2026-09-01 21:36 CST (linux): Record successful full-table CPU/CUDA comparison metrics, generated-report identities, and the retained pre-synchronization lock. -->

## 2026-09-01 21:36 CST — full-table Ubuntu acceptance complete

- Both isolated comparisons passed all schema, seed, fixed-input, and data-file
  identity checks against Mac reference SHA256
  `b7728a81e2f566d7db503320804234296b1fb1f8d230908b39482171fbc834b3`.
- Linux CPU comparison passed:
  - all 6993-mode amplitudes: relative L2 `4.872e-17`, normalized maximum
    `4.363e-17`;
  - five targeted modes: relative L2 `1.660e-20`;
  - 2104-sample short waveform: relative L2 `2.750e-11`, normalized maximum
    `6.732e-11`, flat-weight mismatch `0`;
  - model load `10.669 s`, first/warm amplitudes `1.704 s` / `0.1149 s`, and
    first/warm waveform `10.196 s` / `0.2310 s`;
  - peak process RSS `6473.93 MiB`.
- CUDA 12.x comparison on the RTX 2080 Ti passed:
  - all 6993-mode amplitudes and all five targeted modes are exactly equal to
    the Mac reference (`0` maximum and relative-L2 differences);
  - short waveform: relative L2 `2.750e-11`, normalized maximum `6.732e-11`,
    flat-weight mismatch `0`;
  - model load `13.421 s`, first/warm amplitudes `2.075 s` / `0.01188 s`, and
    first/warm waveform `9.039 s` / `0.01839 s`;
  - peak process RSS `6483.68 MiB`; CuPy pool maximum total `5202.91 MiB` and
    final used `4853.78 MiB`.
- Both backends produced finite values and bitwise-repeatable full amplitudes
  and waveforms. The known upstream fixture-index-2 defect recorded by Mac is
  reproduced and remains intentionally excluded only from the upstream-value
  assertion; all five actual values were included in the cross-host comparison.
- Generated reports:
  - `high_memory_kerr_cpu.json`: 14,624 bytes, SHA256
    `e72bc2b5a1f6906242b2b37857dfd3596df1726aea4c5dc0d73370c62116c0d8`;
  - `high_memory_kerr_cuda12x.json`: 14,885 bytes, SHA256
    `1e78a279389c60bea7159e172b6b2ec7f0875f8a6f0be17524bd579b63d77d3a`.
- `jq` acceptance predicates and `git diff --check` pass. The ignored H5
  symlink remains local. Ubuntu retains the edit lock until the user directs a
  commit/push or another host switch.

<!-- 2026-09-01 21:38 CST (linux): Release the Ubuntu edit lock and bind the complete full-table CPU/CUDA reports to the user-directed Git handoff. -->

## 2026-09-01 21:38 CST — released after full Kerr synchronization handoff

- The user directed handoff, commit, and push after both high-memory
  comparisons passed. This handoff commit includes the two small JSON reports,
  this Linux log, and the shared lock transition.
- The 5,089,095,248-byte H5 and its local symlink remain ignored and are not
  staged. A future collaborator may acquire the lock only after pulling this
  branch and confirming a clean worktree.

<!-- 2026-09-01 23:15 CST (linux): Record creation of the shared mixed-precision research plan and knowledge-index update requested by the user. -->

## 2026-09-01 23:15 CST — mixed-precision proposal documented

- Acquired the shared edit lock to add `knowledge/MIXED_PRECISION_PLAN.md` and
  link it from `knowledge/README.md`; no source, build, validation artifact, or
  large data file was changed.
- The proposal keeps the accepted FP64 path as the production default, assigns
  FP64 to dynamics/phases/final accumulation, proposes FP32 for an opt-in Kerr
  coefficient/interpolation experiment, and limits initial FP16 work to ROMAN
  dense multiplication with FP32 accumulation.
- It records the completed full-table timings as the baseline, cites primary
  CUDA/FEW/waveform-accuracy evidence, defines cold/warm and waveform-level
  acceptance gates, and recommends CUDA `mixed32` as the first milestone before
  either FP16 Tensor Core or Apple Metal implementation work.

<!-- 2026-09-01 23:36 CST (linux): Record the safe pull of Mac's strict-Metal research commit and reconciliation of independently unsynchronized lock histories without editing Mac-owned files. -->

## 2026-09-01 23:36 CST — strict-Metal handoff pulled and reviewed

- Preserved the pending Linux mixed-precision documentation in a named stash,
  fast-forwarded from `f87258e8` to Mac commit `5c872d31`, then restored the
  Linux files. The only textual conflict was `collaboration/LOCK.md`; its Mac
  and Linux timestamp histories were both retained, and current ownership was
  reconfirmed for Linux. No file under `collaboration/mac/` was modified.
- Mac's isolated double-single Metal amplitude prototype reaches normalized
  maximum error `2.40e-15` on the accepted four-point/6993-mode workload and
  approximately `3.2x`--`4.2x` warm per-slice speedup.
- Mac's first FP32 summation prototype reaches `11.42x` for a combined one-year
  workload but fails the strict normalized waveform gate. The follow-up
  double-single summation recovers a normalized maximum `5.81620e-11` and
  relative L2 `1.67612e-11` at `8.62x`, passing the current `5e-10` engineering
  limit while retaining bitwise repeatability and numerical-zero flat mismatch.
- The Metal work remains an isolated feasibility prototype: it is absent from
  CMake, backend registration, installed extensions, and production defaults.
  Mac requests broader CPU/CUDA reference validation, persistent-buffer work,
  and LISA PSD-weighted scientific validation before integration.

<!-- 2026-09-01 23:45 CST (linux): Hand back the strict-Metal task because Mac's synchronized commit contains measured summaries but no raw Metal waveform artifact that Linux can compare. -->

## 2026-09-01 23:45 CST — request missing strict-Metal comparison data

- Linux cannot execute the Objective-C++/Metal kernels. Commit `5c872d31`
  contains their source and reviewed scalar metrics, but no raw strict-Metal
  waveform arrays or structured run report. Reconstructing those values on
  CUDA would not validate the actual Metal arithmetic.
- Mac should add a compressed artifact such as
  `collaboration/mac/strict_metal_ds_reference.npz`. Store the unrounded
  `complex128` output produced with both strict double-single amplitude
  interpolation and strict double-single mode summation for this exact common
  configuration: `M=1e6`, `mu=10`, `theta=pi/3`, `phi=pi/4`, `dist=1`,
  `Phi_phi0=0.3`, `Phi_theta0=0`, `Phi_r0=0.7`, and `dt=15 s`.
- Required cases are:
  1. baseline short: `a=0.7, p0=11, e0=0.4, xI=1, T=0.001 yr`;
  2. baseline one year: the same orbit with `T=1.0 yr`;
  3. positive-spin retrograde: the baseline with `xI=-1, T=0.01 yr`;
  4. inner orbit: `a=0.6, p0=8, e0=0.3, xI=1, T=0.01 yr`;
  5. zero spin: `a=0, p0=11, e0=0.4, xI=1, T=0.01 yr`.
- Mac should also add `collaboration/mac/strict_metal_ds_report.json` containing
  schema/seed, exact inputs, dtype/shape, modes kept, H5 and `LPA.txt` hashes,
  safe-math/compiler/device metadata, cold/warm timings, peak RSS, repeatability,
  and local Metal-versus-CPU maximum/relative-L2/flat-mismatch metrics for every
  case. Record both artifact sizes and SHA256 values in the Mac handoff.
- CPU waveform arrays need not be included: Ubuntu will regenerate CPU and
  CUDA 12.x arrays from the exact inputs. The requested Metal-only payload is
  expected to remain below ordinary GitHub's 100 MiB per-file limit; the H5,
  temporary dylibs, and caches must remain out of Git.
- After Mac commits and pushes these missing data on the existing branch,
  Ubuntu will compare every Metal array against both FP64 backends, compute
  flat and `LPA.txt` LISA-weighted error metrics, record CPU/CUDA timings and
  memory, and decide whether the strict DS prototype passes the broader
  cross-host engineering gate.
