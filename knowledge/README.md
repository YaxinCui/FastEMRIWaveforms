# Apple Silicon and FEW knowledge index

<!-- 2026-09-01 18:38 CST (mac): Expand the source-backed index after the pre-implementation Apple Silicon research pass. -->

<!-- 2026-09-01 23:15 CST (linux): Index the user-requested mixed-precision research plan while preserving FP64 as the accepted production baseline. -->

<!-- 2026-09-01 23:37 CST (linux): Cross-link the pulled Mac strict double-single Metal feasibility evidence with the mixed-precision plan. -->

<!-- 2026-09-02 15:17 CST (mac): Link the reproducible local literature archive
and whole-pipeline review, and update the Metal status after explicit production
integration without changing CPU/CUDA defaults. -->

<!-- 2026-09-03 17:18 CST (linux): Index the user-requested, source-cited
undergraduate-level Apple Silicon CPU adaptation report. -->

<!-- 2026-09-03 18:00 CST (linux): Index the evidence-based Apple Silicon CPU
acceleration test report and scientific-acceptance plan. -->

This index favors primary documentation and papers. It records why a technique
is relevant before code or large reference artifacts are added to the project.

The detailed evidence matrix and implementation gates are in
[APPLE_SILICON_RESEARCH.md](APPLE_SILICON_RESEARCH.md).

The concise explanation of the Apple Silicon FP64 CPU adaptation—principles,
Accelerate/GCD implementation, measured results, and validation boundaries—is
in [APPLE_SILICON_CPU_ADAPTATION_REPORT.md](APPLE_SILICON_CPU_ADAPTATION_REPORT.md).

The completed evidence, reproducible performance protocol, layered test matrix,
numerical error definitions, current regression gates, and the boundary between
engineering and LISA/SNR-dependent scientific acceptance are in
[APPLE_SILICON_CPU_ACCELERATION_TEST_REPORT.md](APPLE_SILICON_CPU_ACCELERATION_TEST_REPORT.md).

The opt-in FP64/FP32/FP16 workload partition, staged prototype order, and
waveform-level acceptance gates are in
[MIXED_PRECISION_PLAN.md](MIXED_PRECISION_PLAN.md).

The measured native-Metal float/double-single feasibility study is in
[APPLE_GPU_METAL_FEASIBILITY.md](APPLE_GPU_METAL_FEASIBILITY.md). It records the
historical proof of concept that led to the now-integrated, explicitly opt-in
strict time-domain Metal mode-sum backend. CPU and CUDA defaults remain
unchanged.

The source-verified, local-only literature collection and its problem-to-source
map are in [library/INDEX.md](library/INDEX.md). Exact URLs, sizes, page counts,
and SHA-256 digests are tracked in
[library/MANIFEST.tsv](library/MANIFEST.tsv); the 74 MB of PDF payloads are
ignored by Git and independently reconstructible on either host.

The end-to-end pipeline, layered error budget, 5 GB HDF5 access audit, backend
architecture risks, and prioritized experiment sequence are in
[FEW_ARCHITECTURE_AND_APPLE_ADAPTATION.md](FEW_ARCHITECTURE_AND_APPLE_ADAPTATION.md).

## Apple numerical acceleration

- [Apple Accelerate overview](https://developer.apple.com/documentation/accelerate)
  — Apple describes CPU vector processing, BLAS, LAPACK, vDSP, vForce, sparse
  solvers, and BNNS as high-performance, energy-efficient primitives selected
  for the processor at runtime.
- [Apple BLAS documentation](https://developer.apple.com/documentation/accelerate/blas-library)
  — authoritative column-major CBLAS interface and threading controls. macOS 26
  aligns the new interface with LAPACK 3.12 and enables it through
  `ACCELERATE_NEW_LAPACK`.
- [Apple BLAS threading model](https://developer.apple.com/documentation/accelerate/blas_threading)
  — BLAS/LAPACK may use framework-managed threads, while applications that
  parallelize the enclosing work can explicitly request single-threaded calls.
- [Apple vForce](https://developer.apple.com/documentation/accelerate/vforce-library)
  — vectorized double- and single-precision transcendental functions, including
  paired sine/cosine evaluation, selected for the current architecture.
- [Apple vDSP](https://developer.apple.com/documentation/accelerate/vdsp)
  — double-precision vector arithmetic and FFT/DFT primitives.
- [Apple `dispatch_apply_f`](https://developer.apple.com/documentation/dispatch/dispatch_apply_f)
  — the system concurrent-queue primitive for a synchronous parallel loop;
  iterations must be reentrant and independent.
- [Apple Metal resources and specifications](https://developer.apple.com/metal/resources/)
  — authoritative entry point for Metal language and GPU feature-set tables.
- [Apple JAX Metal plug-in](https://developer.apple.com/metal/jax/) — Apple lists
  `float64`, `complex64`, and `complex128` among the unsupported data types.
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) —
  maps PyTorch graphs and kernels to Metal Performance Shaders on the GPU.
- [PyTorch's current MPS tensor allocation source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/EmptyTensor.cpp)
  — rejects double tensors because the MPS framework does not support them.
- [MLX NumPy interoperability](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/numpy.rst)
  — the Apple MLX project documents that NumPy `float64` arrays convert to MLX
  `float32` by default. FEW's native kernels use `double` and `complex128`, so
  an MLX/Metal port needs a waveform-error study before it can replace the FP64
  reference path.
- [AdaptiveCpp experimental Metal backend](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-metal.md)
  — records no hardware `double`, no 64-bit atomics, and no portable-CUDA
  support on its Metal backend.
- [`metal-float64`](https://github.com/philipturner/metal-float64) — an archived,
  explicitly unfinished software-FP64 experiment; useful evidence, but not a
  production dependency.

## FEW and EMRI methods

- [FastEMRIWaveforms: New tools for millihertz gravitational-wave data analysis](https://arxiv.org/abs/2104.04582)
  — the primary FEW framework paper, including modular waveform construction,
  mode reduction, GPU acceleration, and accuracy considerations.
- [The Fast and the Frame-Dragging](https://arxiv.org/abs/2506.09470) — recent
  FEW work on efficient eccentric equatorial inspirals into rapidly spinning
  black holes. It reports approximately `1e-5` LISA-weighted model mismatch over
  most of the domain and identifies trajectory and summation as major costs.
- [Model Waveform Accuracy Standards](https://arxiv.org/abs/0809.3844) — derives
  detector-noise-weighted error requirements for detection and parameter
  measurement. It is the basis for validating any reduced-precision experiment
  with waveform mismatch rather than elementwise error alone.
- [Use and Abuse of the Model Waveform Accuracy Standards](https://arxiv.org/abs/0907.0457)
  — warns against treating one generic tolerance as valid for every detector,
  source SNR, and data-analysis use.
- [Upstream FEW repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)
  and [documentation](https://fastemriwaveforms.readthedocs.io/en/stable/) —
  implementation and user-facing backend behavior.

## Current engineering decision

The first macOS path uses Accelerate's FP64 `dgemm`, `zgemm`, and `dgtsv`.
This removes an accidental Homebrew/gfortran runtime dependency and accelerates
the ROMAN dense layers without changing FEW's double/complex-double storage.
The strict time-domain mode sum also has an explicit Apple-Metal implementation
using double-single arithmetic with NumPy host arrays. It remains opt-in and is
gated by frozen-input, one-year waveform, packaging, and cross-host validation;
broader Metal coverage must still be justified by end-to-end measurements rather
than kernel throughput alone.

<!-- 2026-09-03 CST (linux): Expand knowledge base with comprehensive domain guides, verified local paper archive, and reference engineering repositories. -->

## Accelerated Computing Knowledge Base

To guide subsequent algorithm optimization and GPU mixed-precision acceleration, three deep domain guides, an authenticated 36-paper local PDF archive, and curated open-source reference repositories have been integrated into the local workspace:

1. **[GW Physics & EMRI Theory Guide](guides/GW_PHYSICS_AND_EMRI_THEORY.md)**:
   - Relativistic black hole perturbation theory, Teukolsky equations, Kerr bound geodesics ($p, e, x_I$), and separatrices.
   - Adiabatic radiation reaction, multipolar mode decompositions, and mode-selection energy thresholds.
   - Rigorous scientific error budgets: Lindblom accuracy criteria, noise-weighted inner product, Overlap, and detector-weighted mismatch ($\mathcal{M}$).
2. **[Signal Processing & Likelihood Acceleration Guide](guides/SIGNAL_PROCESSING_AND_LIKELIHOOD_ACCELERATION.md)**:
   - Frequency-domain waveform generation (*Fast and Fourier*), Stationary Phase Approximation (SPA), and multivoice decomposition.
   - Non-Uniform Fast Fourier Transforms (NUFFT / cuFINUFFT) on GPUs.
   - Likelihood evaluation acceleration for MCMC: Heterodyned Likelihood / Relative Binning, and Reduced Order Quadrature (ROQ).
   - LISA Time Delay Interferometry (TDI) instrument response modeling on GPUs.
3. **[Parallel Computing & Mixed-Precision Practice Guide](guides/PARALLEL_COMPUTING_AND_MIXED_PRECISION.md)**:
   - Modern GPU architecture breakdown (Turing SM 7.5, FP64 vs FP32 vs Tensor Core throughput ratios).
   - Tiered precision design: rigid FP64 trajectory/phase vs FP32 amplitude tables vs Double-Single compensated mode accumulation.
   - CUDA engineering patterns: cuBLAS handle lifecycle reuse, stream concurrency, and memory coalescing.
   - 5GB Kerr table optimization: replacing 13-second eager loading with lazy HDF5 hyperslab slice reads and LRU caching.
4. **[Curated Engineering Reference Repositories](reference_repos/README.md)**:
   - `fastlisaresponse` (`lisa-on-gpu`): in-situ CuPy/CUDA instrument response and polynomial interpolation.
   - `QD`: industrial-strength Double-Double and Quad-Double compensated arithmetic templates.
   - `sleef`: ultra-fast SIMD/vectorized elementary functions for transcendental kernels.
5. **[Authenticated Literature Archive (36 Local PDFs)](library/INDEX.md)**:
   - All 36 primary literature papers (74.05 MB) downloaded and verified with exact SHA-256 digests in [`library/MANIFEST.tsv`](library/MANIFEST.tsv) under `library/downloads/`.

## Classical Textbooks, Comprehensive Monographs, and Official Manuals

To provide foundational theoretical depth beyond research papers, a dedicated archive of **18 classical graduate-level textbooks, comprehensive monographs, and official hardware/library manuals** has been downloaded, verified, and integrated locally:

- **[Textbook Library Index & Reading Map](textbooks/INDEX.md)**: Curated guide mapping standard textbook chapters (GR, self-force, DSP, CUDA) directly to FEW modules and optimization tasks.
- **[Textbook Manifest](textbooks/MANIFEST.tsv)**: Provenance tracking of all 18 local volumes (35.8 MB), exact URLs, pages, and SHA-256 digests in `textbooks/downloads/`.
- **[Textbook Deep Synthesis & Derivations](guides/TEXTBOOK_DEEP_SYNTHESIS.md)**: Concise, textbook-level synthesis of key theorems, derivations, and mathematical formalisms:
  - *Kerr Metric, First Integrals & Mino Time Decoupling* (Carter 1968, Carroll GR).
  - *Two-Timescale Adiabatic Radiation Reaction & Teukolsky Master Equation* (Poisson 2004, Barack & Pound 2018).
  - *Matched Filter Theorem, Stationary Gaussian Noise & Fisher Information Matrix* (Jaranowski & Krolak 2012, Cutler & Flanagan 1994).
  - *IEEE 754 Floating-Point Error-Free Transformations (TwoSum) & GPU Latency Hiding* (Kirk & Hwu PMPP, NVIDIA CUDA Guides).
