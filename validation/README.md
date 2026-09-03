# Dual-host numerical validation

<!-- 2026-09-01 18:59 CST (mac): Add the shared, read-only-on-validation workflow that transfers deterministic Mac references to the Ubuntu CUDA host. -->

`dual_host_consistency.py` generates or compares a deterministic reference
artifact without requiring the 4.74 GiB high-memory Kerr amplitude file.

The covered paths are:

- FP64 real neural-layer GEMM;
- complex-double ROM projection;
- full Schwarzschild ROMAN amplitudes;
- Schwarzschild bicubic amplitudes;
- a short fast Schwarzschild eccentric waveform; and
- a short PN5 AAK waveform.

Generate the Mac reference after building the default Apple Accelerate path:

```sh
.venv/bin/python validation/dual_host_consistency.py generate \
  --backend cpu \
  --output collaboration/mac/apple_silicon_reference.npz
```

The binary artifact cannot contain a collaboration comment, so the Mac owner
must record its size and SHA256 in `collaboration/mac/HANDOFF.md`.

On Ubuntu, compare the same artifact against both backends. The report paths
belong in the Linux-owned directory:

```sh
python validation/dual_host_consistency.py compare \
  --backend cpu \
  --reference collaboration/mac/apple_silicon_reference.npz \
  --report collaboration/linux/cpu_comparison.json

python validation/dual_host_consistency.py compare \
  --backend cuda12x \
  --reference collaboration/mac/apple_silicon_reference.npz \
  --report collaboration/linux/cuda12x_comparison.json
```

<!-- 2026-09-01 19:22 CST (mac): Clarify that cross-host input identity is an enforced validation condition. -->

Comparison fails if the artifact schema, deterministic seed, or ordered
size/SHA256 manifest of the four workload data files differs from the current
host. FEW/Python/platform versions remain report fields because the purpose is
to compare distinct Mac CPU and Ubuntu CPU/CUDA environments.

The reported waveform mismatch is FEW's normalized flat-weight overlap check.
It is a strict cross-backend regression metric, not a substitute for the
source/SNR/PSD-dependent scientific mismatch described in
`knowledge/APPLE_SILICON_RESEARCH.md`.

<!-- 2026-09-01 19:57 CST (linux): Document the CUDA validation evidence behind the AAK-specific normalized tolerance. -->

The AAK CPU kernel uses FEW's historical Numerical Recipes Bessel
approximations, whereas its CUDA kernel uses the CUDA libdevice `jn` function.
This produces a deterministic relative amplitude difference of approximately
`1.1e-9` on the dual-host workload while retaining a flat-weight waveform
mismatch near machine precision. Consequently, only the AAK normalized-array
limit is `5e-9`; all other workload limits and the independent `1e-10`
waveform-mismatch limit remain stricter.

The default workload deliberately excludes `FastKerrEccentricEquatorialFlux`.
Its registered amplitude table `ZNAmps_l10_m10_n55_DS2Outer.h5` is
5,089,095,248 bytes and tagged `high_memory`. Add it only after an explicit
large-data transfer decision.

## Opt-in full-table Kerr validation

<!-- 2026-09-01 21:24 CST (mac): Document the separately transferred,
ignored high-memory table workflow and the Mac-to-Ubuntu acceptance commands. -->

`high_memory_kerr_consistency.py` is the opt-in acceptance runner for the full
Kerr eccentric equatorial path. Before allocating a model it requires the
exact registered size and SHA256 of both `KerrEccEqFluxData.h5` and
`ZNAmps_l10_m10_n55_DS2Outer.h5`. The latter remains ignored and must never be
added to ordinary Git; transfer it out of band to the same
`src/few/data/` path on both hosts.

The runner evaluates all 6993 amplitude modes at four points spanning both
interpolation regions, evaluates FEW's five published Kerr amplitude fixtures,
and generates a deterministic 0.001-year Kerr waveform twice. It records load
and execution timings, process peak RSS, optional CuPy pool usage, finiteness,
and bitwise repeatability. The amplitude and waveform checks intentionally
share the waveform object's single full-table amplitude generator. This avoids
a second 5 GB allocation that macOS's allocator may retain even after Python
releases the first model.

<!-- 2026-09-01 21:25 CST (mac): Record the upstream fixture defect exposed by
running the previously skipped full-table amplitude coverage. -->

Four of the five fixture values are enforced at absolute tolerance `1e-9`.
Fixture index 2 declares mode `(2,2,0,0)`, but its expected value instead
closely resembles `(2,-2,0,5)` and differs from the declared mode by about
`0.408`. The upstream test's assertion is currently outside its loop, so it
silently checks only fixture index 4. The runner records this exception and all
five actual values; importantly, the dual-host comparison still compares all
five Mac/CPU results against Ubuntu CPU and CUDA results.

Generate the small Mac reference:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  validation/high_memory_kerr_consistency.py generate \
  --backend cpu \
  --output collaboration/mac/high_memory_kerr_reference.npz
```

After the artifact and handoff are committed and synchronized, Ubuntu runs
both comparisons while owning the project lock:

```sh
python validation/high_memory_kerr_consistency.py compare \
  --backend cpu \
  --reference collaboration/mac/high_memory_kerr_reference.npz \
  --report collaboration/linux/high_memory_kerr_cpu.json

python validation/high_memory_kerr_consistency.py compare \
  --backend cuda12x \
  --reference collaboration/mac/high_memory_kerr_reference.npz \
  --report collaboration/linux/high_memory_kerr_cuda12x.json
```

Comparison requires identical schema, seed, fixed inputs, and data-file
manifests. Normalized maximum and relative-L2 limits are `5e-11` for Kerr
amplitudes and `5e-10` for the waveform; the independent flat-weight waveform
mismatch limit is `1e-10`.

## Strict-Metal cross-host validation

<!-- 2026-09-02 00:08 CST (linux): Document the integrity-bound five-case
CPU/CUDA reproduction and the deliberately limited interpretation of LPA
noise weighting requested by the Mac strict-Metal handoff. -->

`strict_metal_cross_host.py` verifies the exact Mac NPZ/report identities,
embedded metadata, unrounded `complex128` array hashes, PoC source hashes, and
all three data files before allocating the full Kerr model. It then regenerates
the five report-defined waveforms with one shared full-table model, checks
bitwise repeatability and mode counts, and compares Linux against Metal using
normalized maximum error, relative L2 error, FEW's flat mismatch, and a global
phase-optimized vector mismatch.

The additional `LPA.txt` result is an engineering diagnostic. It uses a
two-sided DFT of FEW's complex strain, log-log interpolation of ASD squared,
and reports both zero-lag phase optimization and circular discrete
time-plus-phase optimization. It deliberately does not claim a complete LISA
TDI response, sky response, parameter-bias calculation, or astrophysical error
budget.

<!-- 2026-09-02 00:15 CST (linux): Separate same-host kernel limits from
cross-host end-to-end acceptance after the Linux CPU reproduction showed the
expected trajectory-sensitive accumulation while all overlap gates passed. -->

The report still evaluates the existing `5e-10` normalized-maximum and
relative-L2 Metal gate, but marks it non-binding across hosts. That limit
isolates Metal only when Metal and CPU consume the same Mac-prepared
trajectory, spline, and mode inputs. Regenerating an end-to-end waveform on
x86_64 also changes those upstream floating-point paths, especially over one
year, so applying the local kernel gate to that result would conflate two
experiments. Cross-host acceptance instead requires exact provenance,
shape/dtype/mode count, finiteness, repeatability, and `1e-10` limits on flat,
phase-optimized, and both LPA-weighted mismatches. The elementwise results
remain recorded so their accumulation is never hidden. A future kernel-only
cross-host gate should transfer the prepared summation inputs as a separate
artifact.

Run each backend in a separate process while Ubuntu owns the edit lock:

```sh
.venv/bin/python validation/strict_metal_cross_host.py \
  --backend cpu \
  --runtime-artifact /tmp/strict_metal_linux_cpu.npz \
  --report collaboration/linux/strict_metal_cpu.json

.venv/bin/python validation/strict_metal_cross_host.py \
  --backend cuda12x \
  --cpu-peer /tmp/strict_metal_linux_cpu.npz \
  --report collaboration/linux/strict_metal_cuda12x.json
```

<!-- 2026-09-02 00:16 CST (linux): Document the ephemeral array bridge used to
make same-host CPU-to-CUDA elementwise validation independent of the Mac
trajectory accumulation. -->

The first command's temporary NPZ is integrity-bound but deliberately
untracked. The second command enforces the elementwise gates for its direct
Linux CPU-to-CUDA comparison and embeds all resulting metrics and the temporary
artifact identity in the persistent CUDA JSON report. The temporary NPZ may be
deleted after the CUDA report is complete.

## Frozen strict-Metal summation validation

<!-- 2026-09-02 11:01 CST (mac): Document the kernel-only experiment added to
separate independent trajectory/spline generation from backend summation. -->

`strict_metal_frozen_sum.py` consumes the exact eight arrays and five ABI
scalars captured immediately before Mac strict Metal mode summation. It does
not load the 5.09 GB amplitude table or regenerate the trajectory, amplitude
spline, phase spline, mode indices, or Ylms. Therefore its `5e-10`
normalized-maximum and relative-L2 gates are binding: any remaining difference
belongs to CPU/CUDA versus strict Metal summation, not to independently
accumulated upstream state.

FEW's summation ABI writes an internal dimensionless waveform. The capture
report records the exact source-frame distance divisor used by FEW, verifies
that dividing the captured raw Metal result reproduces the existing physical
waveform bitwise, and applies that same operation to every CPU/CUDA replay
before comparison.

Mac regeneration and CPU acceptance use:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  collaboration/mac/metal_poc/generate_strict_metal_frozen_sum.py \
  --repetitions 2

VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  validation/strict_metal_frozen_sum.py \
  --backend cpu --repetitions 2
```

After synchronization, Ubuntu should run each backend in a separate process
and write only to its Linux-owned directory:

```sh
.venv/bin/python validation/strict_metal_frozen_sum.py \
  --backend cpu --repetitions 2 \
  --output collaboration/linux/strict_metal_frozen_sum_cpu.json

.venv/bin/python validation/strict_metal_frozen_sum.py \
  --backend cuda12x --repetitions 2 \
  --output collaboration/linux/strict_metal_frozen_sum_cuda12x.json
```

The tracked frozen-input NPZ is approximately 195 KB. Ubuntu still needs the
existing 33 MB strict-Metal waveform reference, but it does not need another
H5 transfer for this kernel-only replay.

<!-- 2026-09-02 13:52 CST (mac): Add the installed opt-in Metal backend to the
same frozen-input acceptance contract used for the isolated PoC and Ubuntu. -->

After building the Apple Silicon Metal extension, validate that the registered
backend reproduces the accepted strict kernel with:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  validation/strict_metal_frozen_sum.py \
  --backend metal --repetitions 2
```

The default output is Mac-owned
`collaboration/mac/strict_metal_production_backend.json`; CPU and CUDA defaults
and the existing evidence files are not overwritten.

<!-- 2026-09-02 14:10 CST (mac): Document the installed-backend, public-API
acceptance that complements the frozen kernel-only replay. -->

The full-table end-to-end acceptance exercises the public
`force_backend="metal"` path over five Kerr waveforms, compares it to CPU
summation inside the same generator, checks repeatability and unchanged CPU
state, and records source/data hashes:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  validation/metal_backend_end_to_end.py --repetitions 2
```

This command requires the ignored 5.09 GB
`ZNAmps_l10_m10_n55_DS2Outer.h5` table and writes only
`collaboration/mac/metal_backend_end_to_end.json`. It enforces `5e-10`
normalized-maximum and relative-L2 limits plus a `1e-10` flat-mismatch limit;
it does not replace the wider cross-host or PSD-weighted scientific campaign.

<!-- 2026-09-02 13:32 CST (linux): Document the mixed host/device pointer
placement required by the CUDA summation ABI after the first frozen replay
exposed an all-device transfer segfault. -->

For CUDA, `phase_times` and `trajectory_times` intentionally remain contiguous
NumPy host arrays: `get_waveform` dereferences those knot arrays in host C++
while determining spline windows and launch scalars. The waveform,
interpolation/phase coefficients, mode indices, and Ylms remain CuPy device
arrays. Treating every frozen array as a device pointer is invalid for this
mixed ABI and causes a native segmentation fault before a numerical report can
be produced.

## Cross-host trajectory reproducibility diagnostic

<!-- 2026-09-02 17:48 CST (mac): Document the production-neutral fast-math A/B
and adaptive-step trace added after independent one-year construction exposed
a larger upstream Mac/Linux pointwise difference. -->

`trajectory_reproducibility.py` runs the established one-year Kerr baseline
twice: once with the production `_p_to_u` Numba `fastmath=True` mapping and once
with an instance-local `fastmath=False` replacement. It does not edit the
production trajectory source. Both variants capture:

- the eight-point sparse trajectory and DOP853 dense-output coefficients;
- each attempted step's time, step size, error estimate, previous-reject state,
  acceptance decision, next step size, and state before/after the attempt;
- fixed direct `_p_to_u` probes that do not depend on a host-computed trajectory;
- exact source/data hashes, runtime versions, and per-array hashes.

The 5.09 GB amplitude table is not used. Only the registered 9.86 MB
`KerrEccEqFluxData.h5` file is required.

Mac regeneration uses:

```sh
.venv/bin/python validation/trajectory_reproducibility.py \
  --output-prefix collaboration/mac/trajectory_reproducibility
```

The accepted Mac artifact is only about 16 KB. On M3 Pro, fast-math and strict
variants are bitwise equal for all 30 captured arrays, all seven attempts are
accepted, and the smallest `abs(err - 1)` is `0.07513375704875513`. Therefore a
one- or two-ULP direct flip of the `err <= 1` branch is not supported for this
Mac baseline. Ubuntu is still required to determine whether the hosts differ
before that controller decision or retain the same step topology with different
states/coefficients.

After synchronization, Ubuntu owns the lock and runs:

```sh
.venv/bin/python validation/trajectory_reproducibility.py \
  --output-prefix collaboration/linux/trajectory_reproducibility \
  --reference-artifact collaboration/mac/trajectory_reproducibility.npz \
  --reference-report collaboration/mac/trajectory_reproducibility.json
```

Ubuntu writes only to `collaboration/linux/`. Its JSON report compares every
array byte-for-byte and numerically, first verifies the reference NPZ against
its JSON, embedded metadata, and per-array hashes, records the first differing
element when shapes agree, and compares the common first-axis prefix if the
number of DOP853 attempts differs. A fast-math effect is established only if
the Ubuntu strict variant moves materially closer to the Mac strict artifact;
same-host equality alone is not sufficient.
