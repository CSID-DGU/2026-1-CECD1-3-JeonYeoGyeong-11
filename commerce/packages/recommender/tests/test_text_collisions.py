import unittest

from commerce.evaluation.text_collisions import collision_shares, per_seller


def tokenize(texts):
    """A stand-in tokenizer: one ID per word, cut at three IDs."""
    return [[len(w) for w in t.split()][:3] for t in texts]


def item(seller, item_id, title):
    return {"schema_version": "catalog_item.v1", "seller_id": seller, "item_id_local": item_id, "source": "live",
            "title_text": title, "description_text": None, "category_path": None, "listing_status": "active",
            "first_listed_at": None}


class Collisions(unittest.TestCase):
    def test_text_and_truncated_token_duplicates(self):
        shares = collision_shares({"a": "milk 1L", "b": "milk 1L", "c": "oat milk 1L", "d": "oat milk 2L x"},
                                  tokenize, max_length=3)
        self.assertEqual(shares["items"], 4)
        self.assertEqual(shares["text_duplicate"], 0.5)  # a and b
        # The cut makes c and d one token list too: [3, 4, 2] for both.
        self.assertEqual(shares["token_duplicate"], 1.0)
        self.assertEqual(shares["truncated"], 0.5)

    def test_duplicates_count_within_each_seller_only(self):
        items = [item("s1", "1", "우유"), item("s1", "2", "우유"), item("s2", "3", "우유"), item("s2", "4", "버터")]
        out = per_seller(items, tokenize, max_length=3)
        self.assertEqual(out["sellers"], 2)
        self.assertEqual(out["text_duplicate_macro"], 0.5)  # s1 all, s2 none
        self.assertEqual(out["token_duplicate_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
