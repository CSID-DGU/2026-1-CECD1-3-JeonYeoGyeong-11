import unittest

import numpy as np

from graphfl_lab.strategies.graphfl.targets import (
    AggregationTargetConfig,
    AggregationTargetContext,
    AggregationTargetResult,
    evaluate_aggregation_target,
    register_aggregation_target,
    unregister_aggregation_target,
)


class AggregationTargetRegistryTest(unittest.TestCase):
    def test_custom_target_is_evaluated_once_and_shared(self):
        name = "test_shared_post_updates"
        calls = {"count": 0}

        @register_aggregation_target(name, override=True)
        def target(context: AggregationTargetContext) -> AggregationTargetResult:
            calls["count"] += 1
            beta = float(context.config.parameters["beta"])
            post = [
                [np.asarray(layer) * beta for layer in client]
                for client in context.local_updates
            ]
            return AggregationTargetResult(
                post_local_updates=post,
                target_used=name,
                metadata={"beta": beta},
            )

        try:
            current = [np.asarray([10.0, 20.0])]
            updates = [
                [np.asarray([2.0, 4.0])],
                [np.asarray([6.0, 8.0])],
            ]
            weights = [
                [current[0] + client[0]]
                for client in updates
            ]
            evaluation = evaluate_aggregation_target(
                current_global=current,
                local_weights=weights,
                local_updates=updates,
                alpha_norm=np.asarray([0.25, 0.75]),
                config=AggregationTargetConfig(
                    target=name,
                    parameters={"beta": 0.5},
                ),
            )

            self.assertEqual(calls["count"], 1)
            np.testing.assert_allclose(
                evaluation.post_flat_updates,
                np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            )
            np.testing.assert_allclose(
                evaluation.candidate_global[0],
                np.asarray([12.5, 23.5]),
            )
            self.assertEqual(evaluation.metadata["parameters"], {"beta": 0.5})
            self.assertEqual(evaluation.metadata["target_used"], name)
        finally:
            unregister_aggregation_target(name)


if __name__ == "__main__":
    unittest.main()
