"""The g2/g3 scenario input B hands to C: invented sellers that load into the real runtime."""
import hashlib
import tempfile
import unittest

from commerce.packages.recommender import scenario as sc
from commerce.packages.recommender.serving import canonical_npz


def install_random(runtime, variant, version, seed=0):
    manifest, tensors = sc.random_init_release(runtime, variant, seed=seed)
    data = canonical_npz(tensors, [t["name"] for t in manifest["tensors"]])
    release = {"schema_version": "model_release.v1", "model_version": version,
               "manifest_hash": manifest["manifest_hash"], "weights_sha256": hashlib.sha256(data).hexdigest(),
               "weights_size_bytes": len(data)}
    runtime.install_release(release, manifest, tensors, model_variant=variant)


def request(seller, customer, top_n=50):
    return {"schema_version": "recommendation_request.v1", "seller_id": seller, "customer_id_local": customer,
            "as_of": sc.AS_OF, "candidate_item_ids": None, "top_n": top_n}


class ScenarioTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = sc.scenario()

    def test_the_new_seller_shares_no_product_with_the_cohort(self):
        self.assertEqual(set(self.data.sellers), set(sc.COHORT) | {sc.NEW_SELLER})
        cohort_titles = {i["title_text"] for s in sc.COHORT for i in self.data.sellers[s].catalog}
        cohort_ids = {i["item_id_local"] for s in sc.COHORT for i in self.data.sellers[s].catalog}
        new = self.data.sellers[sc.NEW_SELLER].catalog
        self.assertFalse({i["title_text"] for i in new} & cohort_titles)
        self.assertFalse({i["item_id_local"] for i in new} & cohort_ids)

    def test_every_seller_loads_and_its_customers_get_model_scores(self):
        for seller in self.data.sellers.values():
            runtime = sc.open_test_runtime(seller.seller_id, self.tmp.name)
            sc.load_seller(runtime, seller)
            install_random(runtime, "text_relation", "base-0")
            served = runtime.predict_local(request(seller.seller_id, seller.customers[0]))
            self.assertIsNone(served["fallback_reason"], seller.seller_id)

    def test_a_purchase_gives_the_new_item_a_relation_and_moves_its_score(self):
        seller = self.data.sellers["g3-seller-a"]
        runtime = sc.open_test_runtime(seller.seller_id, self.tmp.name)
        sc.load_seller(runtime, seller)
        install_random(runtime, "text_relation", "base-0")
        customer = seller.customers[1]
        before = {i["item_id_local"]: i["score"] for i in runtime.predict_local(request(seller.seller_id, customer))["items"]}
        epoch = runtime.store.feature_epoch
        runtime.ingest_purchase_event(sc.new_item_purchase(seller.seller_id, seller.customers[2],
                                                           seller.catalog[0]["item_id_local"]))
        self.assertEqual(runtime.store.feature_epoch, epoch + 1)
        after = {i["item_id_local"]: i["score"] for i in runtime.predict_local(request(seller.seller_id, customer))["items"]}
        self.assertIn(sc.NEW_ITEM, before)
        self.assertNotAlmostEqual(before[sc.NEW_ITEM], after[sc.NEW_ITEM], places=6)


if __name__ == "__main__":
    unittest.main()
