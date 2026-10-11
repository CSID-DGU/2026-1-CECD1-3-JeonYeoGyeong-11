"""OQ03: a product has the same text and z in the lab and through the seller runtime.

The lab (harex_compare, e_g0) builds texts with catalog_item_text from the adapters'
catalog_item.v1 and reads z from a ZCache; the service receives catalog_item.v1 through
upsert_catalog_item, stores it in the feature ledger and builds z the same way. A tiny
random BERT stands in for the encoder. z may differ around 1e-8 with the batch padding
(nlp-encoder.md §4), hence the 1e-6 tolerance.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from commerce.packages.data_adapters.dunnhumby import Product, catalog_item as dunnhumby_item
from commerce.packages.data_adapters.instacart import load_assignment, load_instacart
from commerce.packages.data_adapters.text import catalog_item_text
from commerce.packages.recommender.seller_runtime import SellerRuntime
from commerce.packages.recommender.serving import SERVICE_ENCODER, FrozenText
from commerce.packages.recommender.tests import write_tiny_encoder
from commerce.packages.recommender.text_encoder import FrozenTextEncoder
from commerce.packages.recommender.z_cache import ZCache

FIXTURES = Path(__file__).resolve().parents[2] / "data_adapters" / "tests" / "fixtures"


def source_items() -> list[dict]:
    """Catalog items as each adapter writes them: Instacart, Dunnhumby and live."""
    folder = FIXTURES / "instacart_small"
    instacart = load_instacart(folder, load_assignment(folder / "assignment.csv")).catalog_items
    dunnhumby = [dunnhumby_item("dh-store-990367", "990101", Product("TEST DEPT", "TEST CAT DAIRY", "TEST TYPE", "990 G")),
                 dunnhumby_item("dh-store-990367", "990102", Product("TEST DEPT", "TEST CAT DAIRY", None, None))]
    live = json.loads((FIXTURES / "live" / "catalog_items.json").read_text(encoding="utf-8"))
    return instacart + dunnhumby + live


class TextRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.encoder_dir = write_tiny_encoder(self.root / "encoder")

    def test_every_source_has_the_lab_text_and_z_in_the_service(self):
        items = source_items()
        self.assertEqual({i["source"] for i in items}, {"instacart", "dunnhumby", "live"})
        lab_texts = {(i["seller_id"], i["item_id_local"]): catalog_item_text(i) for i in items}
        lab = ZCache(self.root / "lab.sqlite", FrozenTextEncoder(self.encoder_dir, SERVICE_ENCODER), self.encoder_dir)
        self.addCleanup(lab.close)
        keys = sorted(lab_texts)
        lab_z = dict(zip(keys, lab.vectors([lab_texts[k] for k in keys])))
        by_seller: dict[str, list[dict]] = {}
        for item in items:
            by_seller.setdefault(item["seller_id"], []).append(item)
        for seller, own in by_seller.items():
            runtime = SellerRuntime(seller, self.root / seller / "features.sqlite", self.root / seller / "models",
                                    text=FrozenText(self.encoder_dir))
            for seq, item in enumerate(own, start=1):
                runtime.upsert_catalog_item(item, seq)
            snap = runtime.store.snapshot()
            catalog = runtime._catalog(snap)
            for row, item_id in enumerate(catalog.items):
                self.assertEqual(catalog_item_text(snap.catalog[item_id]), lab_texts[(seller, item_id)])
                np.testing.assert_allclose(catalog.z[row].numpy(), lab_z[(seller, item_id)], atol=1e-6,
                                           err_msg="%s %s" % (seller, item_id))
            self.assertEqual(set(catalog.items), {i["item_id_local"] for i in own})


if __name__ == "__main__":
    unittest.main()
