import unittest

from graphfl_lab.lifecycle.delivery import (
    DeliveryContext,
    GlobalDeliveryPolicy,
    InterfaceTargetDeliveryPolicy,
    MissingPersonalizedStateError,
    PreviousPersonalizedDeliveryPolicy,
)
from graphfl_lab.lifecycle.state_store import StateStore


class DeliveryPolicyTest(unittest.TestCase):
    def test_global_delivery_sends_same_model(self):
        context = DeliveryContext(
            server_round=1,
            cids=["a", "b"],
            global_model="global",
            state_store=StateStore(),
        )

        result = GlobalDeliveryPolicy().run(context)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, {"a": "global", "b": "global"})
        self.assertEqual(result.trace_records[0].values["delivery_policy"], "global")

    def test_previous_personalized_uses_global_cold_start_with_trace(self):
        store = StateStore()
        store.set_personalized_model("a", "personalized-a")
        context = DeliveryContext(
            server_round=2,
            cids=["a", "b"],
            global_model="global",
            state_store=store,
        )

        result = PreviousPersonalizedDeliveryPolicy().run(context)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.support_level, "proxy-supported")
        self.assertEqual(result.output["a"], "personalized-a")
        self.assertEqual(result.output["b"], "global")
        self.assertEqual(result.trace_records[0].values["delivery_cold_start"], "global_with_trace")

    def test_previous_personalized_strict_mode_raises(self):
        context = DeliveryContext(
            server_round=2,
            cids=["missing"],
            global_model="global",
            state_store=StateStore(),
        )

        with self.assertRaises(MissingPersonalizedStateError):
            PreviousPersonalizedDeliveryPolicy(strict=True).run(context)

    def test_interface_target_delivery_is_unsupported(self):
        context = DeliveryContext(
            server_round=1,
            cids=["a"],
            global_model="global",
            state_store=StateStore(),
        )

        result = InterfaceTargetDeliveryPolicy("cloud_model").run(context)

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.support_level, "interface-target")


if __name__ == "__main__":
    unittest.main()
