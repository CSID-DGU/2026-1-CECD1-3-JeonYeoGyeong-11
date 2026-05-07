import unittest

import numpy as np

from spectral_fl.strategies.spectral.momentum import apply_server_optimizer


class SpectralMomentumModuleTest(unittest.TestCase):
    def test_no_optimizer_returns_candidate_unchanged(self):
        current = [np.array([1.0, 2.0], dtype=np.float64)]
        candidate = [np.array([2.0, 4.0], dtype=np.float64)]

        new_global, momentum, diagnostics = apply_server_optimizer(
            current_global=current,
            candidate_global=candidate,
            server_learning_rate=1.0,
            server_momentum=0.0,
            server_momentum_vector=None,
        )

        self.assertIs(new_global, candidate)
        self.assertIsNone(momentum)
        self.assertEqual(diagnostics["server_optimizer"], "none")

    def test_server_sgd_uses_candidate_delta_direction(self):
        current = [np.array([1.0, 2.0], dtype=np.float64)]
        candidate = [np.array([3.0, 6.0], dtype=np.float64)]

        new_global, momentum, diagnostics = apply_server_optimizer(
            current_global=current,
            candidate_global=candidate,
            server_learning_rate=0.5,
            server_momentum=0.0,
            server_momentum_vector=None,
        )

        self.assertIsNone(momentum)
        self.assertEqual(diagnostics["server_optimizer"], "server_sgd")
        self.assertTrue(np.allclose(new_global[0], np.array([2.0, 4.0])))

    def test_server_momentum_carries_previous_step(self):
        current = [np.array([0.0, 0.0], dtype=np.float64)]
        first = [np.array([1.0, 0.0], dtype=np.float64)]

        _, momentum, diagnostics = apply_server_optimizer(
            current_global=current,
            candidate_global=first,
            server_learning_rate=1.0,
            server_momentum=0.9,
            server_momentum_vector=None,
        )

        self.assertEqual(diagnostics["server_optimizer"], "fedavgm_style_momentum")
        self.assertTrue(np.allclose(momentum[0], np.array([-1.0, 0.0])))

        second = [np.array([0.0, 2.0], dtype=np.float64)]
        new_global, momentum, _ = apply_server_optimizer(
            current_global=current,
            candidate_global=second,
            server_learning_rate=1.0,
            server_momentum=0.9,
            server_momentum_vector=momentum,
        )

        self.assertTrue(np.allclose(momentum[0], np.array([-0.9, -2.0])))
        self.assertTrue(np.allclose(new_global[0], np.array([0.9, 2.0])))


if __name__ == "__main__":
    unittest.main()
