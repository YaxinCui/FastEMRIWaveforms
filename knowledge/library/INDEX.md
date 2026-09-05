# FEW research library

<!-- 2026-09-02 15:17 CST (mac): Add a problem-driven index for the local,
source-verified FEW/EMRI, numerical-analysis, signal-processing, and GPU
literature archive shared by the two hosts. -->

<!-- 2026-09-04 12:05 CST (linux): Extend the local archive with 19
source-verified phase-two papers/reports and update the problem map for
post-adiabatic dynamics, resonance/plunge switching, fast-model systematics,
global LISA analysis, numerical diagnosis, and scientific provenance. -->

<!-- 2026-09-04 13:37 CST (linux): Add 10 phase-three sources and the
adversarial problem map for convention errors, circular validation, population
bias, event-sensitive differentiation, concurrency, and runtime reliability. -->

This directory separates reproducible metadata from bulky source material:

- [`MANIFEST.tsv`](MANIFEST.tsv) tracks the original URL, exact filename, byte
  count, PDF page count, and SHA-256 digest for every source.
- `downloads/` contains 65 locally downloaded PDFs totaling 171,399,038 bytes.
  It is intentionally ignored by Git. Each host may reconstruct it from the
  manifest instead of moving binary literature through the project branch.
- Binary PDFs cannot carry the Mac/Linux collaboration comments used by this
  project. The dated host annotation in the manifest records their provenance.

The archive is intentionally problem-driven. A paper is useful here only if it
can change an implementation decision, explain an observed error, or define an
acceptance test.

## Why several disciplines matter

FEW is a chain, and a small weakness at one stage can dominate everything after
it:

1. **Relativistic physics** determines the orbital frequencies, radiation
   reaction, harmonic content, and what physical approximation is being made.
2. **Numerical analysis** controls adaptive ODE decisions, interpolation,
   transcendental functions, cancellation, and the reproducibility of millions
   of accumulated radians of phase.
3. **Signal processing** defines the meaningful comparison: detector-weighted
   inner products, maximization over time and phase, FFT/NUFFT methods, and fast
   likelihood approximations.
4. **Parallel computing** decides where the work belongs: CPU vector units,
   Metal, CUDA, or a hybrid pipeline, and how data movement and synchronization
   affect the actual end-to-end time.
5. **Scientific-data engineering** determines whether multi-gigabyte amplitude
   tables are loaded, sliced, cached, repacked, and validated efficiently.

Pointwise waveform error, mismatch, physical model error, and runtime are thus
four different quantities. Optimizing one does not establish the others.

## Phase-two additions

The 19 sources added on Ubuntu close six gaps in the original archive:

| New question | New evidence | Decision it changes |
| --- | --- | --- |
| How should 0PA evolve toward 1PA? | second-order self-force waveforms; invariant pseudo-Hamiltonian and spinning-secondary 1PA work | A future 1PA API needs an explicit phase-space, gauge/invariant-observable, domain, and data contract; it is not merely another forcing array |
| Can trajectory acceleration change the mathematics first? | original NIT inspirals and resonance-aware partial/full averaging | Benchmark NIT overlap before investing heavily in parallel-in-time or low-precision ODE kernels |
| What happens at resonance, plunge, and ringdown? | resonance switching, self-force transition-to-plunge, eccentric Kerr ringdown | Treat these as event- and asymptotic-matching interfaces with continuity and mode-mixing tests, not one cutoff parameter |
| Are mode and interpolation errors already evidenced? | 2026 fast relativistic EMRI systematics study | Reproduce the circular-Kerr-flux result, but maintain separate flux and strain convergence ledgers before judging FEW's eccentric amplitude table |
| What workload should performance serve? | LISA work packages, waveform review, global analysis, differentiable and importance-sampled inference | Add search, incremental global-fit, nearby-parameter, gradient, and failure-visible workloads instead of timing one isolated waveform |
| How should numerical and release risk be diagnosed? | mixed-precision survey, Herbgrind, Verificarlo, surrogate review, FAIR4RS | Select precision by conditioning and correction mechanisms; attach local uncertainty, oracle independence, model/data identity, and provenance |

The full synthesis and falsifiable hypothesis backlog are in
[`../FEW_PHASE2_CROSS_DISCIPLINARY_HYPOTHESES.md`](../FEW_PHASE2_CROSS_DISCIPLINARY_HYPOTHESES.md).
Living official pages and code repositories that should be bookmarked rather
than archived as PDFs are indexed in [`PHASE2_WEB_INDEX.md`](PHASE2_WEB_INDEX.md).

## Phase-three adversarial additions

The next 10 sources were selected by a different question: how can a result
remain finite, smooth, reproducible across backends, and still support a wrong
scientific conclusion?

| Hidden failure | New evidence | FEW implication |
| --- | --- | --- |
| Source/redshifted mass and observable conventions | LISA EMRI science; frequency-domain TDI adaptation | Mass frame, epoch, waveform frame, observable, units, polarization, and complex convention belong in the result identity |
| Self-consistent validation can miss model error | simulation-based calibration; GP model-error marginalization | Separate computational self-SBC from cross-model/independent-physics validation |
| Small event errors can accumulate | population-bias and overlapping-signal/GR-test studies | Validate selection functions and catalogs, not only one-source posterior or mismatch |
| Surrogate confidence is task- and domain-dependent | GPR waveform surrogate; near-extremal-spin extrapolation | Use blocked holdouts, calibrated coverage, and semantic OOD rather than rectangular domain membership |
| Switching systems require switching derivatives | state-dependent-event differentiation | Gradients need event-time, guard/reset, and fixed-event-order checks |
| File integrity is not runtime integrity | GPU permanent-fault study; official HDF5 concurrency and GPU diagnostic guidance | Add call-history, exception-recovery, concurrency, and long-run integrity experiments |

The resulting 68 additional problems and 12 experiments are in
[`../FEW_PHASE3_ADVERSARIAL_FAILURE_AUDIT.md`](../FEW_PHASE3_ADVERSARIAL_FAILURE_AUDIT.md).

## Problem-to-source map

| Observed problem or decision | Most useful local sources | What they can answer |
| --- | --- | --- |
| Independent Mac/Linux trajectories drift by about `1.77e-5` pointwise after a year | FEW 2021/2025; Kerr geodesics; self-force review; CUDA IEEE-754; SLEEF; correctly rounded vector functions; waveform-accuracy papers | Whether the first divergence is an ODE step decision, dense-output interpolation, `sin`/`cos`, or later mode summation; how much of it is phase rather than amplitude; which comparison is scientifically meaningful |
| Apple GPUs lack native FP64 in the Metal kernel language | Apple Metal specifications; float-float GPU operators; quad-double arithmetic; Metal-Sci; Apple HPC studies | Which kernels are safe in FP32, where double-single is justified, and how to measure accuracy/performance instead of assuming a framework backend solves the problem |
| Thousands of oscillatory modes can cancel | Accurate sum/dot product; exact superaccumulators; reproducible summation; FEW framework papers | When pairwise, compensated, binned, or reproducible reductions are worthwhile; how reduction order affects cross-device results |
| Frequency-domain generation may avoid dense time sampling | *Fast and Fourier*; multivoice decomposition; FINUFFT; cuFINUFFT; VkFFT source code | When stationary-phase or nonuniform FFT algorithms reduce work, and which parts could be made portable across Metal and CUDA |
| A backend needs a scientific acceptance criterion | Three waveform-accuracy papers; FINDCHIRP; LISA sensitivity curves; fast LISA response | How to use noise-weighted inner products, time/phase alignment, SNR dependence, and detector response rather than one universal elementwise tolerance |
| A fast kernel does not make a fast application | fast LISA response; Metal-Sci; Metal occupancy/counter documentation; CUDA best practices | How to separate setup, transfers, CPU preparation, dispatch, synchronization, and steady-state throughput; how to test held-out sizes |
| The Kerr amplitude file is about 5 GB and is eagerly materialized | FEW 2025 plus the current `AmpInterpKerrEccEq` source; HDF5 chunking and compressed-I/O guidance | How lazy spin-slice loading, an LRU cache, selected-mode reads, or a derived chunked layout may reduce cold-start latency and memory pressure |
| Parameter estimation needs many nearby waveform evaluations | Reduced-order quadrature; relative binning; heterodyned likelihood | How to reduce the number of expensive waveform or likelihood operations without weakening the reference waveform generator |

## Curated engineering references

These repositories are more useful than another general GPU tutorial because
they expose algorithms, tests, or portability choices relevant to FEW:

- [FEW upstream](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)
  — reference architecture, tests, and CUDA implementation.
- [lisa-on-gpu / fastlisaresponse](https://github.com/mikekatz04/lisa-on-gpu)
  — CuPy/CUDA LISA response construction and interpolation design.
- [FINUFFT](https://github.com/flatironinstitute/finufft) and
  [cuFINUFFT](https://github.com/flatironinstitute/cufinufft) — CPU and CUDA
  nonuniform FFT implementations with accuracy controls.
- [VkFFT](https://github.com/DTolm/VkFFT) — portable FFT code supporting Metal
  and CUDA; especially relevant if FEW's frequency-domain path becomes a target.
- [SLEEF](https://github.com/shibatch/sleef) — portable vector elementary
  functions, useful for studying cross-library `sin`/`cos` differences.
- [ExBLAS](https://github.com/riakymch/exblas) and
  [ReproBLAS](https://github.com/willow-ahrens/ReproBLAS) — reproducible and
  accurate reduction strategies.
- [QD](https://github.com/BL-highprecision/QD) — double-double and quad-double
  reference algorithms.
- [Metal-Sci kernels](https://github.com/vicgalle/metal-sci-kernels) — Apple GPU
  scientific-kernel benchmark methodology and code.
- [MLX](https://github.com/ml-explore/mlx) — useful for Apple unified-memory and
  execution-model ideas, but not a drop-in FP64 FEW backend.

Official implementation references that should remain bookmarked rather than
copied into the repository:

- [Metal Shading Language and feature tables](https://developer.apple.com/metal/resources/)
- [Finding Metal GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy)
- [Apple GPU counter statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
- [Capturing a Metal workload](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode)
- [Porting Metal code to Apple silicon](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
- [HDF5 chunking](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/chunking_in_hdf5.html)
  and [compressed-I/O performance](https://support.hdfgroup.org/documentation/hdf5/latest/improve_compressed_perf.html)

## Reading order for the next investigations

For the cross-host long-trajectory discrepancy:

1. FEW 2021 and 2025 for the construction and known error sources.
2. The self-force review and Kerr-geodesic paper for the physical/mathematical
   variables that must be checkpointed.
3. CUDA IEEE-754, SLEEF, and correctly rounded vector-function work for backend
   variation.
4. The waveform-accuracy and LISA sensitivity papers for the final acceptance
   metric.

For the next Apple acceleration increment:

1. Metal feature tables and language specification for hard device constraints.
2. Metal-Sci and Apple counter documentation for measurement design.
3. Accurate-reduction and float-float papers for numerically sensitive kernels.
4. HDF5 guidance and the FEW amplitude-table access pattern before porting more
   arithmetic to the GPU.

## Deliberately deferred material

- MPSGraph, PyTorch MPS, and generic neural-network optimization papers do not
  solve FEW's FP64 phase requirement and are not core dependencies.
- The Apple Neural Engine is optimized for a different workload and precision
  regime; no FEW production plan should depend on it without a new measured
  numerical study.
- Large textbooks, duplicate preprints, and unsourced blog posts are not kept
  just to make the archive larger.
- Repacking the registered HDF5 science data is an engineering experiment, not
  a literature task. Any repacked file must be a new ignored derived artifact
  with its own checksum and numerical comparison; the registered source file
  must never be overwritten.

## Integrity

The original 36-file archive was checked on Mac with Python 3.12 and `pypdf`.
After the Ubuntu phase-two and phase-three extensions, all 65 manifest rows are
independently rechecked with `stat`, `sha256sum`, `pdfinfo`, and the `%PDF-`
signature: every file must match its recorded byte count, page count, and
SHA-256 digest; the current total is exactly 171,399,038 bytes. These inspection
tools are not FEW runtime or build dependencies.
