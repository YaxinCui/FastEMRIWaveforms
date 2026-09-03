# macOS-owned workspace

<!-- 2026-09-01 18:24 CST (mac): Declare the macOS-only write area. -->

Only the macOS collaborator writes files in this directory. The Ubuntu
collaborator may read these records when validating or resuming work.

<!-- 2026-09-02 11:01 CST (mac): Identify the bounded frozen-summation payload
that Ubuntu may read but must validate into its own write area. -->

`strict_metal_frozen_sum_inputs.npz` and its JSON report contain the exact
prepared strict-Metal summation inputs for five cases. The Mac CPU result is
`strict_metal_frozen_sum_cpu.json`; Ubuntu must place CPU/CUDA replay reports
under `collaboration/linux/` and must not rewrite these Mac-owned files.

<!-- 2026-09-02 14:10 CST (mac): Identify the production-backend acceptance
records added after Ubuntu closed the frozen CPU/CUDA replay. -->

`strict_metal_production_backend.json` proves that the installed backend
reproduces every frozen strict-Metal output exactly.
`metal_backend_end_to_end.json` records the five-case public-API CPU/Metal
comparison, source/data identities, repeatability, memory, and timings. These
two files are Mac-owned; Linux may read them but writes follow-up evidence only
under `collaboration/linux/`.

<!-- 2026-09-02 17:48 CST (mac): Identify the compact trajectory/step-decision
artifact prepared for the first P0 cross-host reproducibility experiment. -->

`trajectory_reproducibility.npz` and `trajectory_reproducibility.json` contain
the one-year Kerr fast-math/strict A/B, all DOP853 attempt decisions, sparse
trajectory and dense-output coefficients, exact source/data identities, and
Mac-local comparisons. The accepted identities are:

- NPZ: 15,590 bytes, SHA256
  `f3e60569274d6dbe084ac4d80e66c5180468601d91a10f18d5cc1804fc0e9d4f`;
- JSON: 13,052 bytes, SHA256
  `bfce21ec9ac0cbb2cd7bccd8ad992bb18a5f22355f51c76a4edd1df232e63d34`.

Linux may read the Mac files as a reference but must write its replay NPZ/JSON
under `collaboration/linux/`.
