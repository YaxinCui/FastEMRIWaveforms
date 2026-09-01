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
