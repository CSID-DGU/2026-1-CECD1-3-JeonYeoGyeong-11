import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.graph_state import select_round_graph


class GraphFLGraphStateTest(unittest.TestCase):
    def test_select_round_graph_uses_raw_current_when_ema_disabled(self):
        current = np.array([[0.0, 1.0], [1.0, 0.0]])
        previous = np.ones((2, 2), dtype=np.float64)

        graph, source = select_round_graph(
            current_graph=current,
            previous_graph_ema=previous,
            use_ema_graph=False,
            in_warmup=False,
            ema_alpha=0.8,
        )

        self.assertEqual(source, "raw_current_graph")
        self.assertIs(graph, current)

    def test_select_round_graph_uses_current_during_warmup(self):
        current = np.array([[0.0, 2.0], [2.0, 0.0]])

        graph, source = select_round_graph(
            current_graph=current,
            previous_graph_ema=np.ones((2, 2), dtype=np.float64),
            use_ema_graph=True,
            in_warmup=True,
            ema_alpha=0.5,
        )

        self.assertEqual(source, "warmup_current_graph")
        self.assertIs(graph, current)

    def test_select_round_graph_preserves_ema_label_when_previous_missing(self):
        current = np.array([[0.0, 3.0], [3.0, 0.0]])

        graph, source = select_round_graph(
            current_graph=current,
            previous_graph_ema=None,
            use_ema_graph=True,
            in_warmup=False,
            ema_alpha=0.5,
        )

        self.assertEqual(source, "ema_graph")
        self.assertIs(graph, current)

    def test_select_round_graph_blends_previous_and_current_graph(self):
        current = np.array([[0.0, 4.0], [4.0, 0.0]])
        previous = np.array([[0.0, 2.0], [2.0, 0.0]])

        graph, source = select_round_graph(
            current_graph=current,
            previous_graph_ema=previous,
            use_ema_graph=True,
            in_warmup=False,
            ema_alpha=0.25,
        )

        self.assertEqual(source, "ema_graph")
        np.testing.assert_allclose(graph, [[0.0, 3.5], [3.5, 0.0]])


if __name__ == "__main__":
    unittest.main()
