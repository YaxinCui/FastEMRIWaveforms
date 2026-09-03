# Ubuntu-owned workspace

<!-- 2026-09-01 18:24 CST (mac bootstrap): Create the Ubuntu write area; future changes here belong only to the Ubuntu collaborator. -->

Only the Ubuntu CUDA collaborator writes files in this directory after this
bootstrap file. The macOS collaborator may read the records but must not modify
them. Ubuntu should record CUDA environment details, reference outputs,
tolerances, and validation results here.

<!-- 2026-09-03 22:33 CST (linux): Index the first experiment on the dedicated
CUDA mixed-compute branch. The probe and JSON are Linux-owned, while the Mac
collaborator remains read-only for this directory. -->

The first outcome-driven CUDA experiment is
[`cuda_mixed_compute_probe.py`](cuda_mixed_compute_probe.py), with its pinned
RTX 2080 Ti result in
[`cuda_mixed_compute_probe.json`](cuda_mixed_compute_probe.json). It compares
the accepted FP64 ROMAN wrapper with the opt-in, still-FP64 CuPy matrix path at
operator and end-to-end waveform levels; it is not evidence that reduced
precision is already accepted.
