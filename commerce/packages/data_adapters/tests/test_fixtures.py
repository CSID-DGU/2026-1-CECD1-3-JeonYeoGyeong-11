import json
from pathlib import Path
import shutil
import tempfile
import unittest

from commerce.packages.data_adapters.baskets import basket_from_event
from commerce.packages.data_adapters.synthetic import write_fixtures
from commerce.packages.data_adapters.tests import FIXTURES, LIVE
from commerce.packages.data_adapters.validation import check_payload


class Reproducible(unittest.TestCase):
    def test_generator_rewrites_the_committed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            shutil.copy(FIXTURES / "config.json", out / "config.json")
            write_fixtures(out)
            generated = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
            committed = sorted(p.relative_to(FIXTURES) for p in FIXTURES.rglob("*")
                               if p.is_file() and p.suffix in (".csv", ".json"))
            self.assertEqual(generated, committed)
            for rel in generated:  # text mode: a CRLF checkout on Windows compares equal
                self.assertEqual((out / rel).read_text(encoding="utf-8"),
                                 (FIXTURES / rel).read_text(encoding="utf-8"), str(rel))


class LivePayloads(unittest.TestCase):
    def test_live_payloads_pass_the_contracts(self):
        for item in json.loads((LIVE / "catalog_items.json").read_text(encoding="utf-8")):
            check_payload("catalog_item.v1", item)
        for event in json.loads((LIVE / "purchase_events.json").read_text(encoding="utf-8")):
            basket_from_event(event)
