import unittest
from types import SimpleNamespace

import numpy as np
from flwr.common import ndarrays_to_parameters

from graphfl_lab.strategies.graphfl.fit_results import collect_client_fit_batch


class GraphFLFitResultsTest(unittest.TestCase):
    def test_collect_client_fit_batch_sorts_by_numeric_cid_and_converts_arrays(self):
        results = [
            (
                SimpleNamespace(cid="10"),
                SimpleNamespace(
                    parameters=ndarrays_to_parameters([np.array([10.0])]),
                    num_examples=5,
                    metrics={"cid": "10", "accuracy": 0.1},
                ),
            ),
            (
                SimpleNamespace(cid="2"),
                SimpleNamespace(
                    parameters=ndarrays_to_parameters([np.array([2.0])]),
                    num_examples=7,
                    metrics={"cid": "2", "accuracy": 0.2},
                ),
            ),
        ]

        batch = collect_client_fit_batch(results)

        self.assertEqual(batch.cids, ["2", "10"])
        self.assertEqual(batch.n_examples, [7, 5])
        np.testing.assert_allclose(batch.n_examples_arr, [7.0, 5.0])
        np.testing.assert_allclose(batch.local_weights[0][0], [2.0])
        self.assertEqual(batch.client_metrics[0]["accuracy"], 0.2)

    def test_collect_client_fit_batch_uses_proxy_cid_when_metric_missing(self):
        results = [
            (
                SimpleNamespace(cid="client-b"),
                SimpleNamespace(
                    parameters=ndarrays_to_parameters([np.array([1.0])]),
                    num_examples=1,
                    metrics={},
                ),
            )
        ]

        batch = collect_client_fit_batch(results)

        self.assertEqual(batch.cids, ["client-b"])
        self.assertEqual(batch.client_metrics, [{}])


if __name__ == "__main__":
    unittest.main()
