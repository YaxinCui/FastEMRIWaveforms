"""Apple Metal backend selection and lifecycle checks.

2026-09-02 13:50 CST (mac): Cover the explicit-only hybrid backend without
placing Metal in FEW's default CUDA/CPU selection path.
"""

import platform
import sys
import unittest

import numpy as np

from few import get_backend
from few.summation.interpolatedmodesum import InterpolatedModeSum


@unittest.skipUnless(
    sys.platform == "darwin" and platform.machine() == "arm64",
    "The Metal backend is supported only on Apple Silicon",
)
class MetalBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = get_backend("metal")

    def test_explicit_backend_features(self):
        self.assertEqual(self.backend.name, "metal")
        self.assertTrue(self.backend.uses_metal)
        self.assertTrue(self.backend.uses_gpu)
        self.assertTrue(self.backend.uses_numpy)
        self.assertFalse(self.backend.uses_cupy)
        self.assertFalse(self.backend.uses_cuda)
        self.assertIs(self.backend.xp, np)

    def test_default_selection_remains_non_metal(self):
        self.assertNotEqual(InterpolatedModeSum().backend_name, "metal")

    def test_backend_is_a_singleton(self):
        self.assertIs(get_backend("metal"), self.backend)
        self.assertFalse(self.backend._metal_summation.closed)


if __name__ == "__main__":
    unittest.main()
