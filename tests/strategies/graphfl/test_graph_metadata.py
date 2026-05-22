import unittest

from graphfl_lab.strategies.graphfl.graph_metadata import client_cluster_ids_from_meta


class GraphFLGraphMetadataTest(unittest.TestCase):
    def test_client_cluster_ids_from_meta_converts_matching_list(self):
        values = client_cluster_ids_from_meta({"cluster_ids": ["1", 2]}, ["a", "b"])

        self.assertEqual(values, [1, 2])

    def test_client_cluster_ids_from_meta_uses_fallback_when_missing(self):
        values = client_cluster_ids_from_meta({}, ["a", "b", "c"])

        self.assertEqual(values, [-1, -1, -1])

    def test_client_cluster_ids_from_meta_uses_fallback_when_length_mismatch(self):
        values = client_cluster_ids_from_meta({"cluster_ids": [1]}, ["a", "b"])

        self.assertEqual(values, [-1, -1])


if __name__ == "__main__":
    unittest.main()
