# Linux handoff: CUDA mixed-precision 5x target

<!-- 2026-09-04 14:27 CST (linux): Start a new Ubuntu handoff rooted directly
at upstream master. This branch deliberately does not inherit the prior Gemini
implementation or Apple production commits; selected conclusions may be
re-established only with explicit provenance and fresh measurements. -->

## Active work

- Branch: `codex/cuda-mixed-precision-5x`
- Base: `origin/master@47e4fea4`
- Performance target: warm end-to-end CUDA waveform median no greater than
  20% of the same-process FP64 reference median on the RTX 2080 Ti.
- Correctness rule: the existing FP64 behavior remains the default and oracle;
  every lower-precision path must be opt-in and pass declared amplitude,
  waveform, finiteness, repeatability, and mode-selection checks.
- Current owner: Ubuntu. Mac must treat production files as read-only until the
  shared lock is released.

## Preserved prior work

- The uncommitted phase-two/phase-three problem-discovery documents from
  `deep-optimization` are preserved in the named Git stash
  `deep-optimization research docs before cuda 5x branch (2026-09-04 CST)`.
- Five pre-existing root-level probe scripts remain untracked and untouched.
- The abandoned temporary branch `cuda-mixed-precision-5x` had no new commits;
  its branch reference was removed after the user required a clean master base
  and a branch name containing `codex`.

## 2026-09-04 implementation and validation status

<!-- 2026-09-04 18:18 CST (linux): Add a resumable checkpoint after the first
5x result and parameter-space accuracy sweep.  Ubuntu still owns the active
lock; this is a live checkpoint, not a release or request to edit. -->

- HEAD is still the clean master base commit `47e4fea4`; all work is uncommitted
  on `codex/cuda-mixed-precision-5x`.  Do not commit or push until the user asks.
- Master CUDA required a compatibility repair in `waveform/base.py`: copy the
  temporary CuPy time array to host for SciPy's DOPR derivative spline, then
  return its result to the selected backend.
- Added explicit (never default) amplitude `mixed32` storage/evaluation and
  several summation ablations.  The winning candidate is
  `mixed32_intrinsic_fast`: FP64 trajectory/phase reconstruction and range
  reduction, FP32 amplitude/trig/harmonics/block accumulation, fused conjugate
  mode pairs, and one FP64 promotion per block.
- Successful wheel/runtime:
  `/tmp/few-cuda5x-symmetry-20260904-1715/`; wheel SHA256
  `3519e5095a2c2c8c275d00f57fe71ed437c0f58b6e572cb1d16d8471addc767e`.
- Unified warm benchmark (`mixed32_kerr_probe.json`): one-year waveform FP64
  `152.4895 ms`, candidate `26.1811 ms`, **5.824x** same-process and `5.872x`
  versus the original master run.  Short cases are only `1.69x`/`1.82x`.
  GPU pool use is about `4950.1 MiB` FP64 versus `2523.9 MiB` candidate.
- Seven-regime sweep (`mixed32_accuracy_sweep.json`): all seven outputs finite,
  all automatic mode selections equal, all flat mismatches below `3.234e-12`.
  Strict pointwise equivalence does not pass: normalized maximum is
  `7.73e-7` to `5.18e-6`, worst for `e0=0.8`.
- Interpretation: the long-waveform 5x performance objective is achieved, but
  detector-noise-weighted mismatch and parameter-bias validation are absent;
  the candidate is not yet approved as a scientific FP64 replacement.
- The five-point amplitude sweep found the `(2,2,0,0)` tabulated reference at
  `(a,p,e)=(0.99090693,1.81708312,0.25313127)` disagrees with both the current
  candidate FP64 path and the untouched master FP64 wheel by about `0.408`.
  The other four tabulated points pass `atol=1e-9`; treat this as a pre-existing
  test/reference issue.
- Focused amplitude/summation tests pass on CUDA and CPU.  Kerr retrograde
  regression passes on CUDA.  Kerr-versus-Schwarzschild still hits the existing
  Schwarzschild ROMAN CuPy-to-SciPy implicit-conversion error and is not caused
  by the mixed Kerr candidate.
- The first accuracy-sweep launch selected a version-specific empty cache and
  left an incomplete 2,323,251,200-byte duplicate.  It was removed after the
  complete 5,089,095,248-byte file at the dev93 cache path was reselected and
  verified by SHA256; the complete file was not modified.

## 2026-09-04 multilayer audit continuation

<!-- 2026-09-04 18:44 CST (linux): Record the knowledge-only continuation so a
future Mac/Linux owner can distinguish restored historical analysis from the
new evidence generated on this branch.  The Ubuntu edit lock remains active. -->

- Restored the three historical problem audits, cross-disciplinary guides,
  Apple reports, literature/textbook indexes and manifests from the named
  `deep-optimization` stash using patch-based file creation.  Existing local
  PDFs were not changed and old production source was not restored.
- Added `knowledge/FEW_PHASE4_EMPIRICAL_EVIDENCE_UPDATE.md`, which maps the
  current master-rooted CUDA evidence back to all twelve layers and adds
  Q143--Q166.
- Newly confirmed issues include workload-dependent speedup, high-eccentricity
  mixed-precision sensitivity, a non-atomic downloader that left a truncated
  final-path file, version-cache duplication, a loop-external five-point test
  assertion, a master/reference amplitude conflict, a broken preprocessing
  import, missing HDF5 model-card metadata, absent real-CUDA public CI, and the
  pre-existing Schwarz CUDA host/device boundary failure.
- The new priority order is semantic/science contract, independent reference,
  LISA TDI/PSD validation, precision sensitivity atlas, and real-GPU CI before
  further arithmetic narrowing.  CPU trajectory/NIT/batching is now the main
  performance research direction because the winning sum kernel is already
  about 6.40 ms while the CPU trajectory is about 15.48 ms.
- `multilayer_failure_probe.py/.json` dynamically reproduces selected static
  findings on the untouched master wheel: NaN masses/equal masses pass Kerr
  preflight, invalid/non-finite mode thresholds pass the policy helper, its
  include-minus warning does not match returned state, an injected integrator
  exception leaves `generating_trajectory=True`, the amplitude preprocessing
  import is absent, and the five-point assertion is outside its loop.  The JSON
  encodes non-finite requested values as strings and is strict-JSON parseable.

## 2026-09-05 GitHub checkpoint

<!-- 2026-09-05 14:24 CST (linux): Record the user-requested add/commit/push
checkpoint before staging. The Ubuntu lock remains active because this is a
remote backup/review point, not yet a transfer of edit ownership to Mac. -->

- The intended commit contains the opt-in CUDA mixed-compute implementation,
  focused regression tests, reproducible benchmark/probe scripts and JSON
  evidence, and the text indexes/audits needed to interpret the result.
- The five pre-existing root-level probe scripts remain deliberately untracked.
  Ignored literature/textbook downloads and the three local reference-repository
  working copies are also excluded; only their manifests, indexes, and
  `knowledge/reference_repos/README.md` are included.
- Release conclusion remains conditional: the one-year workload exceeds the
  5x performance target with small FP64-relative numerical error, while short
  workloads do not reach 5x and detector-weighted/parameter-bias validation is
  still required before scientific-default use.
