"""preprocessing_version names these outputs (z_cache.PREPROCESSING_VERSION).

If a test here fails, the text builder or the relation features now produce something
else. Give PREPROCESSING_VERSION a new value, so releases and z caches built on the old
output stop matching, and then update the expected values below. All input is made up.
"""
import hashlib
import random
import unittest

from commerce.packages.contracts.ids import canonical_json
from commerce.packages.data_adapters.text import catalog_item_text
from commerce.packages.recommender.relations import build_relations
from commerce.packages.recommender.tests import item_id, visits_of
from commerce.packages.recommender.z_cache import preprocessing_version


def item(source, title, description=None, path=None):
    return {"schema_version": "catalog_item.v1", "seller_id": "seller-990001", "item_id_local": "p-990001",
            "source": source, "title_text": title, "description_text": description, "category_path": path,
            "listing_status": "active", "first_listed_at": None}


TEXTS = [
    (item("live", "  유기농   우유 1L ", "무항생제\n목장 원유", [" 식품", "유제품 "]),
     "[NAME] 유기농 우유 1L [DESC] 무항생제 목장 원유 [CAT] 식품 > 유제품"),
    (item("live", "Café Crème 200 g"), "[NAME] Café Crème 200 g"),
    (item("live", "무가당 두유 950ml", "", []), "[NAME] 무가당 두유 950ml"),
    (item("instacart", "Organic Oat Milk 1/2 gal", "ignored", ["dairy eggs", "milk"]),
     "[NAME] Organic Oat Milk 1/2 gal [AISLE] milk [DEPT] dairy eggs"),
    (item("instacart", "Test Snack", None, ["snacks"]), "[NAME] Test Snack [DEPT] snacks"),
    (item("dunnhumby", "TEST TYPE WHOLE", "990 G", ["TEST DEPT", "TEST CAT DAIRY"]),
     "[CAT] TEST CAT DAIRY [TYPE] TEST TYPE WHOLE [SIZE] 990 G"),
    (item("dunnhumby", "TEST CAT DAIRY", None, ["TEST DEPT", "TEST CAT DAIRY"]), "[CAT] TEST CAT DAIRY"),
]

# canonical_json of the relations of relations_input(), floats rounded (4 places, time shares 3).
RELATIONS_DIGEST = "d676602f903a7b38ab116abcf41f48eba3bafedc97e92857e66490f7f4056935"


def relations_input():
    rng = random.Random(20261007)
    items = tuple(item_id(i) for i in range(1, 13))
    visits = {}
    for c in range(6):
        baskets = [sorted({rng.randint(1, 12) for _ in range(rng.randint(1, 4))}) for _ in range(rng.randint(2, 6))]
        gaps = {k: float(rng.choice([1, 3, 7, 14, 30])) for k in range(2, len(baskets) + 1)}
        visits["u%d" % c] = visits_of("u%d" % c, baskets, gaps=gaps)
    return visits, {item: row for row, item in enumerate(items)}, len(items)


def relations_digest():
    r = build_relations(*relations_input())
    shares = lambda t, places: [[[round(float(x), places) for x in s] for s in row] for row in t.float().tolist()]
    view = {"neighbor": r.neighbor.tolist(), "has_neighbor": r.has_neighbor.tolist(),
            "features": shares(r.features, 4), "forward": shares(r.time_forward, 3),
            "backward": shares(r.time_backward, 3)}
    return hashlib.sha256(canonical_json(view)).hexdigest()


class PreprocessingOutputs(unittest.TestCase):
    def test_the_version_is_the_one_the_first_releases_name(self):
        self.assertEqual(preprocessing_version(), "9a2ead06485fcceda0250a11ff06c32f621b79efcb606a31adc870251ef46004")

    def test_product_texts(self):
        for payload, expected in TEXTS:
            self.assertEqual(catalog_item_text(payload), expected)

    def test_relation_features(self):
        self.assertEqual(relations_digest(), RELATIONS_DIGEST)


if __name__ == "__main__":
    unittest.main()
