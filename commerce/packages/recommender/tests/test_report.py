import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from commerce.evaluation.d0022_report import main, paired_bootstrap, verdict

SELLERS = 6


def result(value, n=SELLERS):
    per = [{"ndcg@10": value + 0.01 * i, "recall@20": value, "hr@10": value} for i in range(n)]
    return {"sellers": n, "examples": 10 * n, "macro": {"ndcg@10": value + 0.025, "recall@20": value, "hr@10": value},
            "per_seller": per}


def parts(value):
    return {"all": result(value), "repeat": result(value + 0.1), "explore": result(value / 2),
            "cnew": result(value / 3)}


def fl_record(variant, seed, value, frac=0.0, protocol="next_visit", **settings):
    metrics = {"%s FL" % variant: parts(value), "popularity": parts(0.1), "P-TopFreq": parts(0.4)}
    return {"run": "D0022 federated_lab_sim",
            "settings": {"variant": variant, "target": "basket", "seed": seed, "holdout_frac": frac,
                         "protocol": protocol, **settings},
            "metrics": {"final_round": metrics, "best_round": metrics, "primary": "final_round"},
            "training": {"best_round": 480, "plateaued": True}}


class Bootstrap(unittest.TestCase):
    def test_a_constant_difference_has_a_point_interval(self):
        mean, low, high = paired_bootstrap(np.full(10, 0.3), np.full(10, 0.1))
        self.assertAlmostEqual(mean, 0.2)
        self.assertAlmostEqual(low, 0.2)
        self.assertAlmostEqual(high, 0.2)

    def test_an_interval_holding_zero_is_no_difference(self):
        self.assertEqual(verdict(-0.01, 0.02), "차이를 확인하지 못했다")
        self.assertEqual(verdict(0.01, 0.02), "R_hx가 높다")


class Report(unittest.TestCase):
    def test_tables_from_records(self):
        with tempfile.TemporaryDirectory() as root:
            records = [fl_record(v, s, x) for v, x in (("T_hx", 0.12), ("R_hx", 0.15)) for s in (0, 1, 2)]
            records.append(fl_record("R_hx", 0, 0.14, frac=0.1))
            records.append(fl_record("T_hx", 0, 0.5, protocol="gci"))  # must stay out of the main tables
            records.append(fl_record("T_hx", 0, 0.3, protocol="gci", menu_size=200))  # its own gci table
            records.append(fl_record("T_hx", 0, 0.9, seller_size=200))  # small sellers: their own table
            for i, record in enumerate(records):
                folder = Path(root) / str(i)
                folder.mkdir()
                (folder / "record.json").write_text(json.dumps(record), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["--runs", root])
        text = out.getvalue()
        self.assertIn("| R_hx | FL | 3 | 0.1750 |", text)
        self.assertIn("| 1차 대비 | R_hx − T_hx | 3 | +0.0300 | [+0.0300, +0.0300] | R_hx가 높다 |", text)
        self.assertIn("### 콜드스타트", text)
        self.assertIn("| T_hx | FL | 3 | 0.1450 |", text)  # the gci run did not join the seed mean
        self.assertIn("### HAREX 재현 조건", text)
        self.assertIn("| T_hx | 0.5000 | 0.5250 |", text)  # the Ulsan-like mean holds no BBQ-like run
        self.assertIn("메뉴 200개", text)
        self.assertIn("| T_hx | 0.3000 | 0.3250 |", text)
        self.assertIn("### basket — 판매자 100곳, train 주문 200건", text)
        self.assertIn("| T_hx | FL | 1 | 0.9250 |", text)
        self.assertLess(text.index("train 주문 1,040건"), text.index("train 주문 200건"))  # the main one first


if __name__ == "__main__":
    unittest.main()
