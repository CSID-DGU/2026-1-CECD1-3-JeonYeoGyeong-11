"""Writes small R serialization files: the generator of the synthetic Dunnhumby-shaped input.

It lays bytes out as R's serialize.c does for what rds.py reads (XDR, version 2
or 3; symbols written once and then referenced; ALTREP compact sequences,
wrap_* and deferred strings), so the reader is tested on R's layout without R.
Every value is made up and numeric IDs start at 990000 (data.md §6).
"""
import bz2
from dataclasses import dataclass
from datetime import datetime
import gzip
import lzma
import struct

import numpy as np

from commerce.packages.data_adapters import rds
from commerce.packages.data_adapters.dunnhumby import week_of
from commerce.packages.data_adapters.rds import RVector

_TYPES = {"logical": rds.LGLSXP, "integer": rds.INTSXP, "double": rds.REALSXP, "character": rds.STRSXP,
          "list": rds.VECSXP}


@dataclass(frozen=True)
class CompactIntSeq:
    """ALTREP compact_intseq 1..n, as R writes the row names of a large data frame."""
    n: int


@dataclass(frozen=True)
class WrapReal:
    """ALTREP wrap_real around a double vector."""
    vector: RVector


@dataclass(frozen=True)
class DeferredString:
    """ALTREP deferred_string: an integer vector R turns into text when read."""
    source: RVector


class _Writer:
    def __init__(self, version: int):
        self.out = bytearray()
        self.symbols: dict[str, int] = {}
        self.out += b"X\n"
        self.int(version)
        self.int(0x040300)  # written by R 4.3.0
        self.int(0x030500 if version == 3 else 0x020300)
        if version == 3:
            self.int(5)
            self.out += b"UTF-8"

    def int(self, value: int) -> None:
        self.out += struct.pack(">i", value)

    def charsxp(self, text: str | None, levels: int = rds.UTF8_MASK) -> None:
        if text is None:
            self.int(rds.CHARSXP)
            self.int(-1)
            return
        raw = text.encode("latin-1" if levels == rds.LATIN1_MASK else "utf-8")
        if levels == rds.UTF8_MASK and text.isascii():
            levels = rds.ASCII_MASK  # R marks pure ASCII strings so
        self.int(rds.CHARSXP | levels << 12)
        self.int(len(raw))
        self.out += raw

    def symbol(self, name: str) -> None:
        if name in self.symbols:
            self.int(self.symbols[name] << 8 | rds.REFSXP)
            return
        self.int(rds.SYMSXP)
        self.charsxp(name)
        self.symbols[name] = len(self.symbols) + 1

    def pairlist(self, items, *, tagged: bool) -> None:
        for tag, value in items:
            self.int(rds.LISTSXP | (rds.HAS_TAG if tagged else 0))
            if tagged:
                self.symbol(tag)
            self.value(value)
        self.int(rds.NILVALUE_SXP)

    def altrep(self, name: str, kind: int, state, attributes=None) -> None:
        self.int(rds.ALTREP_SXP)
        self.int(rds.LISTSXP)
        self.symbol(name)
        self.int(rds.LISTSXP)
        self.symbol("base")
        self.int(rds.LISTSXP)
        self.value(RVector("integer", [kind]))
        self.int(rds.NILVALUE_SXP)
        state()
        if attributes:
            self.pairlist(attributes.items(), tagged=True)
        else:
            self.int(rds.NILVALUE_SXP)

    def value(self, value) -> None:
        if value is None:
            self.int(rds.NILVALUE_SXP)
        elif isinstance(value, CompactIntSeq):
            self.altrep("compact_intseq", rds.INTSXP, lambda: self.value(RVector("double", [value.n, 1, 1])))
        elif isinstance(value, WrapReal):
            inner = RVector(value.vector.kind, value.vector.data)
            self.altrep("wrap_real", rds.REALSXP,
                        lambda: self.value(RVector("list", [inner, RVector("integer", [0, 0])])),
                        value.vector.attributes)
        elif isinstance(value, DeferredString):
            def state():
                self.int(rds.LISTSXP)  # CONS(arg, info): a dotted pair
                self.value(value.source)
                self.value(RVector("integer", [0]))
            self.altrep("deferred_string", rds.STRSXP, state)
        else:
            self.vector(value)

    def vector(self, vector: RVector) -> None:
        flags = _TYPES[vector.kind]
        if vector.attributes:
            flags |= rds.HAS_ATTR
        if "class" in vector.attributes:
            flags |= rds.IS_OBJECT
        self.int(flags)
        self.int(len(vector.data))
        if vector.kind in ("logical", "integer"):
            self.out += np.asarray(vector.data, dtype=">i4").tobytes()
        elif vector.kind == "double":
            self.out += np.asarray(vector.data, dtype=">f8").tobytes()
        elif vector.kind == "character":
            for text in vector.data:
                if isinstance(text, tuple):  # (text, encoding levels)
                    self.charsxp(*text)
                else:
                    self.charsxp(text)
        else:
            for item in vector.data:
                self.value(item)
        if vector.attributes:
            self.pairlist(vector.attributes.items(), tagged=True)


def _compress(data: bytes, how: str) -> bytes:
    return {"gzip": gzip.compress, "bzip2": bz2.compress, "xz": lzma.compress, "none": bytes}[how](data)


def rds_bytes(value, *, version: int = 3, compress: str = "gzip") -> bytes:
    writer = _Writer(version)
    writer.value(value)
    return _compress(bytes(writer.out), compress)


def rda_bytes(objects: dict, *, version: int = 3, compress: str = "bzip2") -> bytes:
    writer = _Writer(version)
    writer.pairlist(objects.items(), tagged=True)
    return _compress(b"RDX%d\n" % version + bytes(writer.out), compress)


def strings(values) -> RVector:
    return RVector("character", list(values))


def _length(column) -> int:
    if isinstance(column, CompactIntSeq):
        return column.n
    inner = column.vector if isinstance(column, WrapReal) else column.source if isinstance(column, DeferredString)         else column
    return len(inner.data)


def data_frame(columns: dict, *, tibble: bool = True, compact: bool = True) -> RVector:
    n = _length(next(iter(columns.values())))
    cls = ["tbl_df", "tbl", "data.frame"] if tibble else ["data.frame"]
    rownames = CompactIntSeq(n) if compact else RVector("integer", [rds.INT_NA, -n])
    return RVector("list", list(columns.values()),
                   {"names": strings(columns), "row.names": rownames, "class": strings(cls)})


def factor(values: list[str | None]) -> RVector:
    levels = sorted({v for v in values if v is not None})
    codes = [rds.INT_NA if v is None else levels.index(v) + 1 for v in values]
    return RVector("integer", codes, {"levels": strings(levels), "class": strings(["factor"])})


def posixct(instants: list[datetime], tzone: str = "America/New_York") -> RVector:
    return RVector("double", [t.timestamp() for t in instants],
                   {"class": strings(["POSIXct", "POSIXt"]), "tzone": strings([tzone])})


def write_dunnhumby(folder, rows: list[dict], products: list[dict], *, tzone: str = "America/New_York",
                    week_override: dict[int, int] | None = None) -> None:
    """transactions.rds and products.rda in the completejourney layout.

    rows: household, store, basket, product (str), quantity, sales_value (float), when (aware datetime).
    The week column follows dunnhumby.week_of unless week_override (row index -> week) says otherwise.
    """
    weeks = [week_of(r["when"]) for r in rows]
    for index, week in (week_override or {}).items():
        weeks[index] = week
    zeros = RVector("double", [0.0] * len(rows))
    tx = data_frame({
        "household_id": strings(r["household"] for r in rows),
        "store_id": strings(r["store"] for r in rows),
        "basket_id": strings(r["basket"] for r in rows),
        "product_id": strings(r["product"] for r in rows),
        "quantity": RVector("double", [float(r["quantity"]) for r in rows]),
        "sales_value": RVector("double", [float(r["sales_value"]) for r in rows]),
        "retail_disc": zeros, "coupon_disc": zeros, "coupon_match_disc": zeros,
        "week": RVector("integer", weeks),
        "transaction_timestamp": posixct([r["when"] for r in rows], tzone),
    })
    pr = data_frame({
        "product_id": strings(p["product_id"] for p in products),
        "manufacturer_id": strings("990900" for _ in products),
        "department": strings(p.get("department") for p in products),
        "brand": factor(["National"] * len(products)),
        "product_category": strings(p.get("product_category") for p in products),
        "product_type": strings(p.get("product_type") for p in products),
        "package_size": strings(p.get("package_size") for p in products),
    })
    (folder / "transactions.rds").write_bytes(rds_bytes(tx))
    (folder / "products.rda").write_bytes(rda_bytes({"products": pr}))
