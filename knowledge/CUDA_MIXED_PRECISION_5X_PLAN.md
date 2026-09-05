# CUDA mixed-compute plan: 5x end-to-end target

<!-- 2026-09-04 14:27 CST (linux): Migrate only the governing precision and
validation conclusions needed by the new master-rooted branch. Historical
implementations and their source changes are intentionally not copied. -->

## Success contract

The target is a repeatable **5x warm end-to-end speedup**, measured as

```text
median(FP64 reference wall time) / median(opt-in candidate wall time) >= 5.0
```

on the same RTX 2080 Ti, in one process, after explicit warm-up and CUDA stream
synchronization. Kernel-only, cold-start-only, or cross-machine ratios do not
satisfy the target. Cold model construction, warm amplitude evaluation, warm
waveform generation, peak host RSS, peak GPU pool usage, and transfer volume
are reported separately.

## Precision hypothesis

- Keep trajectory ODE integration, separatrix/domain decisions, fundamental
  frequencies, time, and accumulated phases in FP64.
- Test FP32 coefficient storage and FP32 arithmetic for Kerr amplitude
  interpolation; promote results before phase-sensitive use.
- Test FP32 or hardware-supported FP16 inputs with FP32 accumulation only for
  dense ROMAN matrix operations that profiling shows to be material.
- Use FP64, compensated FP32, or double-single arithmetic for cancellation-
  sensitive mode accumulation.
- Permit adaptive FP64 fallback for high spin, high eccentricity, near-
  separatrix, interpolation-boundary, zero-crossing, or non-finite cases.
- Treat storage, arithmetic, and accumulation precision as independent choices.

The RTX 2080 Ti has a nominal FP32/FP64 peak-throughput ratio of 32, but an
FP64-to-FP32 conversion only halves coefficient traffic. Therefore a 5x
end-to-end result requires more than dtype replacement: profiling may select
lazy data access, persistent buffers, fewer allocations/synchronizations,
kernel fusion, batching, layout changes, or a revised CPU/GPU boundary.

## Scientific gates

Every candidate is compared to the unchanged FP64 oracle and must report:

1. all 6993 Kerr amplitudes at representative Region A/B points;
2. high-spin, retrograde, high-eccentricity, near-separatrix, boundary, and
   cancellation-sensitive cases;
3. fixed-mode and automatic-mode-selection agreement;
4. short, medium, and science-duration waveforms;
5. normalized maximum error, relative L2 error, accumulated phase error, flat
   and LISA-noise-weighted mismatch where available;
6. NaN/Inf absence, repeatability, source-data hash, software/hardware
   provenance, and explicit fallback counts.

Initial strict engineering limits inherited from the accepted dual-host
validation are `5e-11` normalized maximum for Kerr amplitudes, `5e-10` for the
short waveform, and `1e-10` for flat mismatch. A relaxed fast mode may use a
different budget only when justified by source SNR, duration, detector PSD,
and parameter-bias evidence; thresholds must not be moved merely to make a
candidate pass.

## Experiment order

1. Rebuild and measure the untouched master FP64 CUDA reference.
2. Produce a synchronized CPU/GPU timeline and Roofline-style classification.
3. Rank interventions by end-to-end upper bound and scientific risk.
4. Implement one opt-in change at a time with tests before composing changes.
5. Retain only combinations with repeatable total-runtime benefit.
6. Stop at the first scientifically accepted 5x result, or document the
   measured bound and next bottleneck if the target is not yet attainable.

## 2026-09-04 measured milestone

<!-- 2026-09-04 18:18 CST (linux): Record the measured outcome separately from
the original plan.  The performance goal and scientific acceptance are kept as
two independent decisions so a fast diagnostic mode is not mistaken for a
drop-in FP64 replacement. -->

The opt-in `mixed32_intrinsic_fast` candidate combines:

- FP64 trajectory integration, phase-spline reconstruction, and phase range
  reduction;
- FP32 amplitude coefficients/evaluation, trigonometric values, spherical
  harmonics, and block-local accumulation;
- an algebraically fused `+m/-m` conjugate-pair sum using four real weights;
- a single FP64 promotion per output block.  The ordinary FP64 path remains
  the default.

On the RTX 2080 Ti, the synchronized same-process median for the representative
one-year waveform (`T=1`, `dt=15 s`, 2,103,877 samples) changed from
`152.4895 ms` to `26.1811 ms`: **5.824x**, so the declared long-waveform
performance target passed.  The candidate does not reach 5x on short inputs:
`1.692x` at `T=0.001` and `1.822x` at `T=0.01`, because fixed trajectory,
selection, and launch costs dominate.  GPU pool use fell from about
`4950.1 MiB` to `2523.9 MiB`.

The seven-case physical sweep covers nominal, Schwarzschild-limit, high-spin,
high-eccentricity, retrograde, weak-field Region B, and near-separatrix inputs.
All seven retained identical automatic mode selections and passed the declared
flat-weight mismatch limit; the largest mismatch was `3.234e-12`.  However,
none passed the inherited `5e-10` strict pointwise gate.  Normalized maximum
error ranged from `7.73e-7` to `5.18e-6`, with the high-eccentricity case worst.
Thus this is a **performance milestone and promising opt-in fast mode, not yet
a scientifically accepted FP64 replacement**.  Detector-noise-weighted
mismatch and parameter-bias studies remain required.

Evidence is machine-readable in
`collaboration/linux/mixed32_kerr_probe.json` and
`collaboration/linux/mixed32_accuracy_sweep.json`; their companion scripts
record synchronization, source/data hashes, hardware, thresholds, and exact
inputs.  A check of all five tabulated amplitude points also exposed a
pre-existing reference inconsistency: the `(2,2,0,0)` point differs by about
`0.408` even on the untouched master FP64 wheel, while the other four are
within `1e-9`.  This must be resolved upstream rather than attributed to mixed
precision.

The original CUDA Kerr retrograde regression passes.  The separate
Kerr-versus-Schwarzschild GPU regression still fails in the pre-existing
Schwarzschild ROMAN amplitude path because SciPy receives a CuPy array via an
implicit conversion; it is outside the candidate Kerr amplitude/summation
path.  The focused amplitude/summation ABI tests pass on both CUDA and CPU.
