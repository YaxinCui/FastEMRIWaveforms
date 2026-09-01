"""CUDA regression coverage for host-only numerical boundaries.

2026-09-01 19:48 CST (linux): Exercise explicit SciPy and trajectory-spline host
bridges found necessary during Apple-Silicon/CUDA consistency validation.
2026-09-01 20:37 CST (mac): Apply the repository's pinned Ruff formatting after
the Ubuntu-to-Mac handoff; test behavior is unchanged.
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
