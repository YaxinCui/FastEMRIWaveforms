"""Apple Accelerate numerical checks.

2026-09-01 18:24 CST (mac): Added deterministic FP64 regression coverage for
the matrix kernels changed by the Apple Silicon adaptation.
"""

import sys
import unittest

import numpy as np

from few import get_backend


@unittest.skipUnless(sys.platform == "darwin", "Apple Accelerate is macOS-only")
class AppleAccelerateTest(unittest.TestCase):
    def setUp(self):
        self.backend = get_backend("cpu")
        self.rng = np.random.default_rng(20260901)

    def test_neural_layer_matches_numpy(self):
        m, k, n = 37, 11, 19
        matrix = self.rng.normal(size=(m, k))
        weights = self.rng.normal(size=(k, n))
        bias = self.rng.normal(size=n)
        output = np.empty(m * n, dtype=np.float64)

        self.backend.neural_layer_wrap(
            output,
            np.asfortranarray(matrix).ravel(order="F"),
            np.asfortranarray(weights).ravel(order="F"),
            bias,
            m,
            k,
            n,
            1,
        )

        expected = matrix @ weights + bias
        expected = np.where(expected >= 0.0, expected, 0.2 * expected)
        np.testing.assert_allclose(
            output.reshape((m, n), order="F"), expected, rtol=2e-14, atol=2e-14
        )

    def test_complex_projection_matches_numpy(self):
        m, k, n = 31, 7, 23
        network_output = self.rng.normal(size=(m, 2 * k))
        transform = self.rng.normal(size=(k, n)) + 1j * self.rng.normal(size=(k, n))
        network_complex = np.empty(m * k, dtype=np.complex128)
        output = np.empty(m * n, dtype=np.complex128)
        factor = 0.001

        self.backend.transform_output_wrap(
            output,
            np.asfortranarray(transform).ravel(order="F"),
            network_complex,
            np.asfortranarray(network_output).ravel(order="F"),
            m,
            k,
            factor,
            n,
        )

        expected_network = (network_output[:, :k] + 1j * network_output[:, k:]) * factor
        np.testing.assert_allclose(
            output.reshape((m, n), order="F"),
            expected_network @ transform,
            rtol=2e-14,
            atol=2e-14,
        )


if __name__ == "__main__":
    unittest.main()
