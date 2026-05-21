import unittest

from graphfl_lab.lifecycle.context import (
    AggregationContext,
    RelationContext,
    RoundContext,
    StateExtractionContext,
    TopologyContext,
    make_state_extraction_context,
)
from graphfl_lab.lifecycle.modules import (
    AggregationOperator,
    ClientStateExtractor,
    ModuleResult,
    RelationEstimator,
    TopologyOperator,
)
from graphfl_lab.lifecycle.traces import RoundTraceBundle, TraceRecord


class LifecycleContextTest(unittest.TestCase):
    def test_round_context_normalizes_strategy_values(self):
        state_store = {}
        context = RoundContext(
            server_round="3",
            cids=["1", "2"],
            rng="rng",
            config={"graph_mode": "knn"},
            state_store=state_store,
        )

        self.assertEqual(context.server_round, 3)
        self.assertEqual(context.cids, ("1", "2"))
        self.assertEqual(context.config, {"graph_mode": "knn"})
        self.assertIs(context.state_store, state_store)

    def test_round_context_rejects_invalid_round(self):
        with self.assertRaisesRegex(ValueError, "server_round"):
            RoundContext(server_round=-1, cids=["0"])

    def test_state_extraction_helper_builds_phase_context(self):
        context = make_state_extraction_context(
            server_round=1,
            cids=["a", "b"],
            global_weights=["global"],
            local_weights=[["wa"], ["wb"]],
            local_updates=[["ua"], ["ub"]],
            num_examples=[5, 7],
            client_metrics=[{"loss": 0.1}, {"loss": 0.2}],
            config={"graph_source": "classifier_head_update"},
        )

        self.assertIsInstance(context, StateExtractionContext)
        self.assertEqual(context.round_context.cids, ("a", "b"))
        self.assertEqual(context.local_weights, (["wa"], ["wb"]))
        self.assertEqual(context.local_updates, (["ua"], ["ub"]))
        self.assertEqual(context.num_examples, (5, 7))
        self.assertEqual(context.client_metrics[0]["loss"], 0.1)

    def test_phase_contexts_hold_previous_phase_outputs_only(self):
        round_context = RoundContext(server_round=1, cids=["a"])
        relation = RelationContext(round_context=round_context, client_state_output={"state": "x"})
        topology = TopologyContext(round_context=round_context, relation_output={"relation": "x"})
        aggregation = AggregationContext(
            round_context=round_context,
            topology_output={"adjacency": "x"},
            local_updates=[["u"]],
            num_examples=[3],
        )

        self.assertEqual(relation.client_state_output["state"], "x")
        self.assertEqual(topology.relation_output["relation"], "x")
        self.assertEqual(aggregation.local_updates, (["u"],))
        self.assertEqual(aggregation.num_examples, (3,))


class ModuleResultTest(unittest.TestCase):
    def test_ok_result_keeps_output_trace_and_metadata(self):
        record = TraceRecord(
            phase="relation",
            module="cosine",
            name="scores",
            values={"status": "ok", "support_level": "core-supported"},
        )

        result = ModuleResult.ok(
            output={"matrix": "relation"},
            trace_records=[record],
            metadata={"metric": "cosine"},
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.support_level, "core-supported")
        self.assertEqual(result.output, {"matrix": "relation"})
        self.assertEqual(result.trace_records, (record,))
        self.assertEqual(result.metadata["metric"], "cosine")
        self.assertEqual(result.to_trace_bundle().records, [record])

    def test_trace_bundle_is_accepted_as_result_records(self):
        bundle = RoundTraceBundle(
            records=[
                TraceRecord(
                    phase="topology",
                    module="knn",
                    name="adjacency",
                    values={"status": "ok"},
                )
            ]
        )

        result = ModuleResult.ok(trace_records=bundle)

        self.assertEqual(result.trace_records, tuple(bundle.records))

    def test_unsupported_result_is_explicit(self):
        record = TraceRecord(
            phase="relation",
            module="learned_attention",
            name="scores",
            values={
                "status": "unsupported",
                "support_level": "interface-target",
                "component_kind": "RelationEstimator",
            },
        )

        result = ModuleResult.unsupported(
            support_level="interface-target",
            message="learned attention is an interface target",
            trace_records=record,
        )

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.support_level, "interface-target")
        self.assertEqual(result.error_type, "UnsupportedModule")
        self.assertIn("learned attention", result.error_message)
        self.assertEqual(result.trace_records[0].values["status"], "unsupported")

    def test_error_result_captures_exception_type(self):
        error = RuntimeError("failed relation solve")

        result = ModuleResult.error(error, support_level="proxy-supported")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.support_level, "proxy-supported")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertEqual(result.error_message, "failed relation solve")

    def test_result_rejects_invalid_status_support_and_trace_items(self):
        with self.assertRaisesRegex(ValueError, "status"):
            ModuleResult(status="done", support_level="core-supported")
        with self.assertRaisesRegex(ValueError, "support_level"):
            ModuleResult(status="ok", support_level="exact")
        with self.assertRaisesRegex(TypeError, "TraceRecord"):
            ModuleResult.ok(trace_records=[object()])


class ModuleProtocolTest(unittest.TestCase):
    def test_runtime_protocols_require_run_method(self):
        class StateModule:
            def run(self, context):
                return ModuleResult.ok(output=context)

        module = StateModule()

        self.assertIsInstance(module, ClientStateExtractor)
        self.assertIsInstance(module, RelationEstimator)
        self.assertIsInstance(module, TopologyOperator)
        self.assertIsInstance(module, AggregationOperator)


if __name__ == "__main__":
    unittest.main()
