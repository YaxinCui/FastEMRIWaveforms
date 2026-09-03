"""CUDA regression coverage for host-only numerical boundaries.

2026-09-01 19:48 CST (linux): Exercise explicit SciPy and trajectory-spline host
bridges found necessary during Apple-Silicon/CUDA consistency validation.
2026-09-01 20:37 CST (mac): Apply the repository's pinned Ruff formatting after
the Ubuntu-to-Mac handoff; test behavior is unchanged.
2026-09-03 22:00 CST (linux): Cover the opt-in same-FP64 CuPy ROMAN experiment,
including explicit policy validation and the existing normalized CUDA boundary.
"""

import unittest

import numpy as np

from few import get_backend
from few.amplitude.romannet import RomanAmplitude
from few.waveform import FastSchwarzschildEccentricFlux


class RomanCudaNormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.backend = get_backend("cuda12x")
        except Exception as error:
            raise unittest.SkipTest("CUDA 12.x backend is unavailable") from error

    def test_renormalized_amplitudes_stay_on_cuda(self):
        p = np.asarray([10.0, 11.0, 12.0])
        e = np.asarray([0.1, 0.2, 0.3])
        amplitude = RomanAmplitude(buffer_length=8, force_backend="cuda12x")

        output = amplitude(0.0, p, e, np.ones_like(p))

        self.assertIsInstance(output, self.backend.xp.ndarray)
        self.assertTrue(
            bool(self.backend.xp.all(self.backend.xp.isfinite(output)).get())
        )

    def test_cupy_fp64_roman_matches_legacy_cuda(self):
        p_grid, e_grid = np.meshgrid(
            np.linspace(10.0, 14.0, 16), np.linspace(0.1, 0.6, 8)
        )
        p = p_grid.ravel()
        e = e_grid.ravel()
        x = np.ones_like(p)
        legacy = RomanAmplitude(buffer_length=256, force_backend="cuda12x")
        candidate = RomanAmplitude(
            buffer_length=256,
            force_backend="cuda12x",
            cuda_roman_mode="cupy_fp64",
        )

        expected = legacy(0.0, p, e, x)
        actual = candidate(0.0, p, e, x)
        scale = self.backend.xp.max(self.backend.xp.abs(expected))
        normalized_max = (
            self.backend.xp.max(self.backend.xp.abs(actual - expected)) / scale
        )
        relative_l2 = self.backend.xp.linalg.norm(
            (actual - expected).ravel()
        ) / self.backend.xp.linalg.norm(expected.ravel())

        self.assertEqual(actual.dtype, self.backend.xp.complex128)
        self.assertLessEqual(float(normalized_max.get()), 5.0e-12)
        self.assertLessEqual(float(relative_l2.get()), 5.0e-12)

    def test_cupy_fp64_roman_rejects_cpu_backend(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA/CuPy backend"):
            RomanAmplitude(force_backend="cpu", cuda_roman_mode="cupy_fp64")

    def test_cupy_fp64_schwarzschild_waveform_matches_legacy_cuda(self):
        args = (1e6, 1e1, 8.0, 0.2, np.pi / 3, np.pi / 4)
        kwargs = {"dist": 1.0, "T": 0.001, "dt": 15.0}
        legacy = FastSchwarzschildEccentricFlux(force_backend="cuda12x")
        candidate = FastSchwarzschildEccentricFlux(
            amplitude_kwargs={"cuda_roman_mode": "cupy_fp64"},
            force_backend="cuda12x",
        )

        expected = legacy(*args, **kwargs)
        actual = candidate(*args, **kwargs)
        scale = self.backend.xp.max(self.backend.xp.abs(expected))
        normalized_max = (
            self.backend.xp.max(self.backend.xp.abs(actual - expected)) / scale
        )
        relative_l2 = self.backend.xp.linalg.norm(
            (actual - expected).ravel()
        ) / self.backend.xp.linalg.norm(expected.ravel())

        self.assertEqual(actual.dtype, self.backend.xp.complex128)
        self.assertLessEqual(float(normalized_max.get()), 5.0e-11)
        self.assertLessEqual(float(relative_l2.get()), 5.0e-11)

    def test_unknown_cuda_roman_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cuda_roman_mode must be one of"):
            RomanAmplitude(force_backend="cuda12x", cuda_roman_mode="unknown")

    def test_waveform_frequency_spline_accepts_cuda_batches(self):
        waveform = FastSchwarzschildEccentricFlux(force_backend="cuda12x")

        output = waveform(
            1e6,
            1e1,
            8.0,
            0.2,
            np.pi / 3,
            np.pi / 4,
            dist=1.0,
            T=0.001,
            dt=15.0,
        )

        self.assertIsInstance(output, self.backend.xp.ndarray)
        self.assertTrue(
            bool(self.backend.xp.all(self.backend.xp.isfinite(output)).get())
        )


if __name__ == "__main__":
    unittest.main()
