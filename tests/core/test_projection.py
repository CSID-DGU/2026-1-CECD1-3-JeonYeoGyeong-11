import unittest

import numpy as np

from spectral_fl.projection import flatten_weights, unflatten_like


class ProjectionTest(unittest.TestCase):
    def test_unflatten_like_restores_shapes_and_dtypes(self):
        template = [
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((1,), dtype=np.float64),
        ]
        flat = np.arange(7, dtype=np.float64)

        restored = unflatten_like(flat, template)

        self.assertEqual(restored[0].shape, (2, 3))
        self.assertEqual(restored[1].shape, (1,))
        self.assertEqual(restored[0].dtype, np.float32)
        self.assertEqual(restored[1].dtype, np.float64)
        self.assertTrue(np.allclose(flatten_weights(restored), flat))


if __name__ == "__main__":
    unittest.main()
