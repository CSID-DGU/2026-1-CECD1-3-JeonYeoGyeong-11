import unittest

import numpy as np

from spectral_fl.lifecycle.state_store import StateStore, state_store_from_mapping


class StateStoreTest(unittest.TestCase):
    def test_ema_updates_graph_and_relation_are_stored(self):
        store = StateStore()

        first = store.update_ema_updates([np.array([1.0, 0.0])], alpha=0.5)
        second = store.update_ema_updates([np.array([0.0, 1.0])], alpha=0.5)
        store.set_ema_graph(np.eye(2))
        store.set_previous_relation(np.ones((2, 2)))

        self.assertTrue(np.allclose(first[0], np.array([1.0, 0.0])))
        self.assertTrue(np.allclose(second[0], np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(store.ema_graph, np.eye(2)))
        self.assertTrue(np.allclose(store.previous_relation, np.ones((2, 2))))

    def test_personalized_models_basic_get_set_is_proxy_storage(self):
        store = StateStore()
        store.set_personalized_model("c1", {"weights": [1, 2]})

        self.assertEqual(store.get_personalized_model("c1"), {"weights": [1, 2]})
        self.assertIsNone(store.get_personalized_model("missing"))

    def test_trace_records_available_state(self):
        store = StateStore()
        store.update_ema_updates([np.array([3.0, 4.0])])
        store.set_personalized_model("c1", "model")

        trace = store.trace_record(round_number=2)

        self.assertEqual(trace.phase, "state_store")
        self.assertTrue(trace.values["ema_updates_available"])
        self.assertEqual(trace.values["personalized_model_count"], 1)

    def test_state_store_from_mapping(self):
        store = state_store_from_mapping({"personalized_models": {"c": "m"}})

        self.assertEqual(store.get_personalized_model("c"), "m")


if __name__ == "__main__":
    unittest.main()
