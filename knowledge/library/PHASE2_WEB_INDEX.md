# FEW phase-two living web and code index

<!-- 2026-09-04 12:05 CST (linux): Create a source-first index for official
documentation and actively maintained research code used in the second
cross-disciplinary problem-discovery pass. These links supplement, rather than
replace, the immutable PDF identities in MANIFEST.tsv. -->

<!-- 2026-09-04 13:37 CST (linux): Extend the living index with concurrency
and runtime-integrity sources used by the phase-three adversarial audit. The
historical filename is retained to avoid breaking existing links. -->

<!-- 2026-09-04 18:39 CST (linux): Restore this text-only living index from
the deep-optimization stash for cross-host reading.  No PDFs or old production
source were restored; current additions belong in a timestamped later pass. -->

This file records pages whose current documentation, repository history, or
machine-readable interface matters more than a frozen PDF. A link is not an
endorsement or proof that its technique works in FEW; each entry states the
question it can help answer.

## Independent physics and algorithm oracles

- [pybhpt](https://github.com/znasipak/pybhpt) — Python bindings to black-hole
  perturbation calculations; candidate independent values for limited
  geodesic/Teukolsky checks.
- [Fast Self-Forced Inspirals](https://github.com/BlackHolePerturbationToolkit/Fast_Self-Forced_Inspirals)
  — reference implementation accompanying the NIT trajectory work; useful for
  defining an overlap experiment, not a drop-in replacement.
- [Self-force transition-to-plunge ancillary code](https://github.com/gcompere/SelfForceFrameworkForTransitionToPlungeWaveforms)
  — implementation evidence for matched inspiral/transition constructions.
- [Black Hole Perturbation Toolkit](https://bhptoolkit.org/) — registry of
  independent geodesic, Teukolsky, self-force, and waveform resources from
  which a reference ladder can be assembled.

## LISA observable and data-analysis contracts

- [LISA Data Challenges](https://lisa.pages.in2p3.fr/LDC/) — challenge data,
  conventions, and analysis context for testing more than isolated source
  strain.
- [Typed LISA Toolkit API](https://lisa-apc.pages.in2p3.fr/typed-lisa-toolkit/api/toplevel.html)
  — explicit distinction between source and projected waveforms, useful when
  defining result types and detector-response fixtures.
- [fastlisaresponse](https://github.com/mikekatz04/lisa-on-gpu) — independent
  CPU/GPU LISA response path used by FEW analyses; candidate cross-check for
  channel, delay, time, and coordinate conventions.
- [LISA sensitivity tools](https://github.com/eXtremeGravityInstitute/LISA_Sensitivity)
  — executable sensitivity/noise conventions for detector-weighted tests;
  exact version and configuration must be stored with each result.

## Data layout and heterogeneous execution

- [HDF5 chunking guide](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/chunking_in_hdf5.html)
  — explains why chunk shape and cache policy must follow the actual query
  geometry.
- [HDF5 compressed-I/O guidance](https://support.hdfgroup.org/documentation/hdf5/latest/improve_compressed_perf.html)
  — helps distinguish capacity savings from decompression and cache costs.
- [CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  — lifecycle, update, synchronization, and thread-safety constraints for
  replaying a stable execution graph.
- [Apple Metal resources](https://developer.apple.com/metal/resources/) —
  authoritative language and feature tables for capabilities such as native
  data types; framework marketing must not substitute for these constraints.
- [Apple Metal performance tools](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
  — counter-based evidence for residency, occupancy, memory pressure, and
  sustained mobile workloads.

## Reproducible numerical and scientific software

- [Verificarlo](https://github.com/verificarlo/verificarlo) — Monte Carlo
  arithmetic instrumentation for locating unstable operations in a reduced
  CPU reproduction before attempting cross-language/backend coverage.
- [Herbgrind](https://github.com/uwplse/herbgrind) — dynamic root-cause
  analysis for floating-point error; useful only if its instrumentation sees
  the relevant compiled path.
- [ExBLAS](https://github.com/riakymch/exblas) and
  [ReproBLAS](https://github.com/willow-ahrens/ReproBLAS) — reference designs
  for accurate/reproducible reductions when a measured cancellation problem
  justifies their cost.
- [FAIR4RS principles](https://fair-software.eu/) — living implementation
  guidance for persistent software identity, metadata, licensing, and reuse;
  FEW should extend the same discipline to model weights and numerical data.

## Phase-three concurrency and runtime integrity

- [h5py parallel HDF5 guidance](https://docs.h5py.org/en/stable/mpi.html) —
  advises each reader process to open a file independently and warns against
  opening HDF5 state before forking.
- [HDF5 thread-safety design](https://portal.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html)
  — documents the library-wide serialization used by the thread-safe build;
  safety and parallel speedup are separate questions.
- [NVIDIA GPU memory diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/memory.html)
  — official memory-integrity checks and explicit skip/failure states; file
  hashes cannot replace device-health evidence.
- [NVIDIA user-visible GPU health statistics](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
  — ECC, retired-page, and remapping counters that should accompany long-run
  accelerator evidence when supported by the device.

## Rules for use

1. Pin a commit, release, dataset, configuration, and license before treating
   code as an oracle or dependency.
2. Record which values are independent. Two backends sharing FEW formulas and
   tables provide strong implementation regression evidence but not an
   independent physics validation.
3. Archive a public paper/report in `downloads/` only when stable offline
   reading adds value; otherwise retain the official living URL here.
4. Any adopted technique must enter a falsifiable FEW experiment with a
   scientific error metric, end-to-end workload, resource measurement, and a
   failure/rejection rule.
