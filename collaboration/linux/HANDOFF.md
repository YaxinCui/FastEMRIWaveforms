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
