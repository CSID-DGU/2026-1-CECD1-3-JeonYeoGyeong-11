import json
import unicodedata
import unittest

from commerce.packages.contracts.errors import FeatureNotImplemented
from commerce.packages.data_adapters.tests import CONTRACT_FIXTURES, LIVE
from commerce.packages.data_adapters.text import (
    EmptyProductText, audit_texts, build_product_text, catalog_item_text, dunnhumby_fields,
    instacart_fields, live_fields, normalize_field,
)


def contract_example(contract, name):
    return json.loads((CONTRACT_FIXTURES / contract / "valid" / name).read_text(encoding="utf-8"))


class Normalization(unittest.TestCase):
    def test_nfc_and_whitespace(self):
        decomposed = unicodedata.normalize("NFD", "유기농 우유")
        self.assertNotEqual(decomposed, "유기농 우유")
        self.assertEqual(normalize_field(decomposed), "유기농 우유")
        self.assertEqual(normalize_field("  Oat\tDrink\n 1　L "), "Oat Drink 1 L")

    def test_missing_is_none_not_text(self):
        for value in (None, "", "   ", "\t\n"):
            self.assertIsNone(normalize_field(value))
        with self.assertRaises(TypeError):  # a CSV reader's NaN must not become 'nan'
            normalize_field(float("nan"))

    def test_missing_fields_are_omitted(self):
        text = build_product_text(instacart_fields("Oat Drink", None, "  "))
        self.assertEqual(text, "[NAME] Oat Drink")
        self.assertNotIn("None", text)
        self.assertNotIn("nan", text)

    def test_all_missing_is_an_error_not_an_empty_text(self):
        with self.assertRaises(EmptyProductText):
            build_product_text(live_fields(None, "", []))


class SourceTexts(unittest.TestCase):
    def test_instacart_order(self):
        self.assertEqual(build_product_text(instacart_fields("Soda", "soft drinks", "beverages")),
                         "[NAME] Soda [AISLE] soft drinks [DEPT] beverages")

    def test_dunnhumby_order(self):
        self.assertEqual(build_product_text(dunnhumby_fields("FLUID MILK", "WHOLE", "1 GA")),
                         "[CAT] FLUID MILK [TYPE] WHOLE [SIZE] 1 GA")

    def test_live_contract_example(self):
        item = contract_example("catalog_item.v1", "live_new_item_zero_history.json")
        self.assertEqual(catalog_item_text(item),
                         "[NAME] 유기농 통밀 식빵 450g [DESC] 국산 통밀 100%로 구운 식빵이다. "
                         "[CAT] 식품 > 베이커리 > 식빵")

    def test_instacart_contract_example_ignores_derived_description(self):
        item = contract_example("catalog_item.v1", "instacart_inactive_item.json")
        self.assertEqual(catalog_item_text(item),
                         build_product_text(instacart_fields("Soda", "soft drinks", "beverages")))

    def test_dunnhumby_catalog_text_is_not_built_yet(self):
        item = contract_example("catalog_item.v1", "dunnhumby_generated_text.json")
        with self.assertRaises(FeatureNotImplemented):
            catalog_item_text(item)

    def test_sizes_numbers_and_korean_survive(self):
        for title in ("Greek Style Yogurt 1.5 kg", "Whole Milk 1/2 gal", "무가당 두유 950ml",
                      "그릭 요거트 Greek Yogurt 0.5kg", "Café Crème Biscuits 200 g"):
            self.assertIn(title, build_product_text(live_fields(title, None, None)))


class LiveFixtureTexts(unittest.TestCase):
    def setUp(self):
        items = json.loads((LIVE / "catalog_items.json").read_text(encoding="utf-8"))
        self.texts = {item["item_id_local"]: catalog_item_text(item) for item in items}

    def test_pairs_that_differ_only_in_size_or_sugar_stay_distinct(self):
        for a, b in (("item-1", "item-2"), ("item-3", "item-4"), ("item-5", "item-6")):
            self.assertNotEqual(self.texts[a], self.texts[b])

    def test_empty_description_and_absent_path_are_omitted(self):
        self.assertEqual(self.texts["item-7"], "[NAME] 그릭 요거트 Greek Yogurt 0.5kg")
        self.assertEqual(self.texts["item-8"], "[NAME] 제주 감귤 주스 1/2 L [CAT] 식품 > 음료")


class Audit(unittest.TestCase):
    def test_counts(self):
        audit = audit_texts({
            "a": live_fields("우유  1L", None, None),
            "b": live_fields("우유 1L", None, None),  # same text after normalization
            "c": live_fields("우유 1L", None, None),  # same raw values as b
            "d": live_fields(None, " ", None),  # quarantined
        })
        self.assertEqual(audit.items, 4)
        self.assertEqual(audit.empty, 1)
        self.assertEqual(audit.missing_by_marker, {"[CAT]": 4, "[DESC]": 4, "[NAME]": 1})
        self.assertEqual(audit.duplicate_raw, 0.5)
        self.assertEqual(audit.duplicate_text, 1.0)
        self.assertEqual(audit.hangul_in, 6)
        self.assertEqual(audit.hangul_out, audit.hangul_in)
