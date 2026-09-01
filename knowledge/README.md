# Apple Silicon and FEW knowledge index

<!-- 2026-09-01 18:38 CST (mac): Expand the source-backed index after the pre-implementation Apple Silicon research pass. -->

This index favors primary documentation and papers. It records why a technique
is relevant before code or large reference artifacts are added to the project.

The detailed evidence matrix and implementation gates are in
[APPLE_SILICON_RESEARCH.md](APPLE_SILICON_RESEARCH.md).

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
Metal/MPS remains a later experimental path gated by measured end-to-end
waveform mismatch, not by kernel throughput alone.
