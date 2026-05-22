import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.ema import update_client_update_ema


class GraphFLEmaTest(unittest.TestCase):
    def test_update_client_update_ema_initializes_and_copies_updates(self):
        local_updates = [[np.array([1.0, 2.0])]]

        ema_updates, source, stored_updates, stored_cids = update_client_update_ema(
            local_updates=local_updates,
            cids=["c1"],
            previous_updates=None,
            previous_cids=None,
            alpha=0.8,
        )

        self.assertEqual(source, "initialized_current_update")
        self.assertEqual(stored_cids, ["c1"])
        np.testing.assert_allclose(ema_updates[0][0], [1.0, 2.0])
        local_updates[0][0][0] = 99.0
        self.assertEqual(stored_updates[0][0][0], 1.0)

    def test_update_client_update_ema_blends_existing_updates(self):
        ema_updates, source, stored_updates, _ = update_client_update_ema(
            local_updates=[[np.array([3.0, 5.0])]],
            cids=["c1"],
            previous_updates=[[np.array([1.0, 1.0])]],
            previous_cids=["c1"],
            alpha=0.25,
        )

        self.assertEqual(source, "ema_update")
        np.testing.assert_allclose(ema_updates[0][0], [2.5, 4.0])
        np.testing.assert_allclose(stored_updates[0][0], [2.5, 4.0])

    def test_update_client_update_ema_reinitializes_when_cids_change(self):
        ema_updates, source, _, stored_cids = update_client_update_ema(
            local_updates=[[np.array([2.0])]],
            cids=["new"],
            previous_updates=[[np.array([10.0])]],
            previous_cids=["old"],
            alpha=1.0,
        )

        self.assertEqual(source, "initialized_current_update")
        self.assertEqual(stored_cids, ["new"])
        np.testing.assert_allclose(ema_updates[0][0], [2.0])


if __name__ == "__main__":
    unittest.main()
