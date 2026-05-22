import unittest

from graphfl_lab.lifecycle.local_hooks import (
    InterfaceTargetLocalObjectiveHook,
    LocalHookContext,
    NoneLocalObjectiveHook,
    ProximalToDeliveredModelHook,
)


class LocalHookTest(unittest.TestCase):
    def test_none_hook_emits_noop_fit_config(self):
        result = NoneLocalObjectiveHook().run(LocalHookContext(server_round=1))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output["fit_config"]["local_objective_hook"], "none")
        self.assertEqual(result.trace_records[0].values["support_level"], "core-supported")

    def test_proximal_hook_emits_client_fit_config(self):
        result = ProximalToDeliveredModelHook(mu=0.2).run(LocalHookContext(server_round=3))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.support_level, "proxy-supported")
        self.assertEqual(result.output["fit_config"]["local_objective_hook"], "proximal_to_delivered_model")
        self.assertEqual(result.output["fit_config"]["proximal_mu"], 0.2)
        self.assertTrue(result.trace_records[0].values["requires_client_support"])

    def test_interface_target_hook_is_unsupported(self):
        result = InterfaceTargetLocalObjectiveHook("mask_previous_personalized").run(
            LocalHookContext(server_round=1)
        )

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.support_level, "interface-target")


if __name__ == "__main__":
    unittest.main()
