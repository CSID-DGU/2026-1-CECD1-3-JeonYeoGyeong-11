from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np

from commerce.packages.data_adapters import rds
from commerce.packages.data_adapters.rds import RVector, UnsupportedRData, data_frame, read_rda, read_rds
from commerce.packages.data_adapters.tests.rds_writer import (
    CompactIntSeq, DeferredString, WrapReal, factor, posixct, rda_bytes, rds_bytes, strings,
)
from commerce.packages.data_adapters.tests.rds_writer import data_frame as make_frame


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path


class Vectors(Base):
    def frame(self):
        return make_frame({
            "id": strings(["990001", "990002", None]),
            "n": RVector("integer", [1, rds.INT_NA, 3]),
            "x": RVector("double", [0.5, float("nan"), 2.0]),
            "ok": RVector("logical", [1, 0, rds.INT_NA]),
            "kind": factor(["b", None, "a"]),
            "at": posixct([datetime(2017, 1, 2, 5, 5, tzinfo=timezone.utc)] * 3),
        })

    def test_every_compression_and_both_versions_read_the_same(self):
        for version in (2, 3):
            for how in ("gzip", "bzip2", "xz", "none"):
                path = self.write("t.rds", rds_bytes(self.frame(), version=version, compress=how))
                columns, attrs, n = data_frame(read_rds(path))
                self.assertEqual(n, 3)
                self.assertEqual(columns["id"], ["990001", "990002", None])
                self.assertEqual(columns["n"].tolist(), [1, rds.INT_NA, 3])
                self.assertTrue(np.isnan(columns["x"][1]))
                self.assertEqual(columns["ok"].tolist(), [1, 0, rds.INT_NA])
                self.assertEqual(columns["kind"], ["b", None, "a"])
                self.assertEqual(attrs["at"]["tzone"].data, ["America/New_York"])
                self.assertEqual(columns["at"][0], datetime(2017, 1, 2, 5, 5, tzinfo=timezone.utc).timestamp())

    def test_plain_data_frame_row_names_and_rda_contents(self):
        frame = make_frame({"id": strings(["990001"])}, tibble=False, compact=False)
        path = self.write("p.rda", rda_bytes({"products": frame, "other": RVector("integer", [7])}))
        objects = read_rda(path)
        self.assertEqual(list(objects), ["products", "other"])
        self.assertEqual(data_frame(objects["products"])[0]["id"], ["990001"])
        self.assertEqual(objects["other"].data.tolist(), [7])

    def test_strings_keep_their_encoding(self):
        values = strings([("무가당 두유 950ml", rds.UTF8_MASK), ("Café", rds.LATIN1_MASK), ("plain", rds.UTF8_MASK)])
        out = read_rds(self.write("s.rds", rds_bytes(values)))
        self.assertEqual(out.data, ["무가당 두유 950ml", "Café", "plain"])

    def test_altrep_forms_r_writes(self):
        frame = make_frame({
            "seq": CompactIntSeq(4),
            "wrapped": WrapReal(RVector("double", [1.5, 2.5, 3.5, 4.5], {"tzone": strings(["UTC"])})),
            "deferred": DeferredString(RVector("integer", [990001, 990002, rds.INT_NA, 990004])),
        })
        columns, attrs, n = data_frame(read_rds(self.write("a.rds", rds_bytes(frame))))
        self.assertEqual(n, 4)
        self.assertEqual(columns["seq"].tolist(), [1, 2, 3, 4])
        self.assertEqual(columns["wrapped"].tolist(), [1.5, 2.5, 3.5, 4.5])
        self.assertEqual(attrs["wrapped"]["tzone"].data, ["UTC"])
        self.assertEqual(columns["deferred"], ["990001", "990002", None, "990004"])

    def test_what_is_not_read_says_so(self):
        with self.assertRaises(UnsupportedRData):
            read_rds(self.write("ascii.rds", b"A\n2\n"))
        with self.assertRaises(UnsupportedRData):
            data_frame(RVector("integer", [1]))
        with self.assertRaises(UnsupportedRData):
            read_rda(self.write("not.rda", rds_bytes(RVector("integer", [1]))))


if __name__ == "__main__":
    unittest.main()
