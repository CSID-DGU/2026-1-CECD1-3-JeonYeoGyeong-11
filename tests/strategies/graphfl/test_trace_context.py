import unittest

from graphfl_lab.strategies.graphfl.trace_context import with_run_context


class GraphFLTraceContextTest(unittest.TestCase):
    def test_with_run_context_adds_missing_round_and_values(self):
        enriched = with_run_context(
            {"phase": "relation"},
            round_number=3,
            run_id="run-a",
            variant="ours",
            seed=7,
        )

        self.assertEqual(enriched["round"], 3)
        self.assertEqual(enriched["values"]["run_id"], "run-a")
        self.assertEqual(enriched["values"]["variant"], "ours")
        self.assertEqual(enriched["values"]["seed"], 7)

    def test_with_run_context_preserves_existing_round_and_values(self):
        enriched = with_run_context(
            {
                "round": 9,
                "values": {
                    "run_id": "existing",
                    "variant": "baseline",
                    "seed": 1,
                    "status": "ok",
                },
            },
            round_number=3,
            run_id="run-a",
            variant="ours",
            seed=7,
        )

        self.assertEqual(enriched["round"], 9)
        self.assertEqual(enriched["values"]["run_id"], "existing")
        self.assertEqual(enriched["values"]["variant"], "baseline")
        self.assertEqual(enriched["values"]["seed"], 1)
        self.assertEqual(enriched["values"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
