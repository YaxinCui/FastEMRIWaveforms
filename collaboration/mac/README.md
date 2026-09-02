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
