"""Read R's serialization format: enough for the data frames in .rds and .rda files.

Dunnhumby's Complete Journey ships as R files (transactions.rds, products.rda;
data.md §1). Reading them needs no R and nothing outside the lock (OQ12): this
module follows R's serialize.c for the binary XDR format, versions 2 and 3,
behind gzip, bzip2 or xz compression.

Supported: NULL, symbols, pairlists (attributes, .rda contents), environments
(only skipped through), logical/integer/double/character/list/raw vectors and
the ALTREP forms R writes for data frames: compact integer/real sequences,
deferred strings and wrap_* vectors. Anything else raises UnsupportedRData
rather than guessing.

A vector comes back as RVector: numpy arrays for logical, integer and double
(R's NA kept as INT_NA or NaN), a list of str or None for character, a list of
items for lists. data_frame() turns a data.frame or tibble into named columns,
with factors expanded to their level strings.
"""
from __future__ import annotations

import bz2
from dataclasses import dataclass, field
import gzip
import lzma
from pathlib import Path
import struct
from typing import Any

import numpy as np

INT_NA = -2147483648  # R's NA_integer_ and NA for logicals

# SEXP types (Rinternals.h) and the special codes of serialize.c.
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP, PROMSXP, LANGSXP = 0, 1, 2, 3, 4, 5, 6
CHARSXP, LGLSXP, INTSXP, REALSXP, CPLXSXP, STRSXP, DOTSXP = 9, 10, 13, 14, 15, 16, 17
VECSXP, EXPRSXP, RAWSXP, S4SXP = 19, 20, 24, 25
REFSXP, NILVALUE_SXP, GLOBALENV_SXP, UNBOUNDVALUE_SXP, MISSINGARG_SXP = 255, 254, 253, 252, 251
BASENAMESPACE_SXP, NAMESPACESXP, PACKAGESXP, PERSISTSXP = 250, 249, 248, 247
EMPTYENV_SXP, BASEENV_SXP, ATTRLANGSXP, ATTRLISTSXP, ALTREP_SXP = 242, 241, 240, 239, 238

IS_OBJECT, HAS_ATTR, HAS_TAG = 1 << 8, 1 << 9, 1 << 10
# A CHARSXP's encoding is in its levels, the flag bits above 12.
BYTES_MASK, LATIN1_MASK, UTF8_MASK, ASCII_MASK = 1 << 1, 1 << 2, 1 << 3, 1 << 6
_PAIRLIST_TYPES = (LISTSXP, LANGSXP, CLOSXP, PROMSXP, DOTSXP, ATTRLANGSXP, ATTRLISTSXP)
_MARKERS = {GLOBALENV_SXP: "globalenv", UNBOUNDVALUE_SXP: "unbound", MISSINGARG_SXP: "missingarg",
            BASENAMESPACE_SXP: "basenamespace", EMPTYENV_SXP: "emptyenv", BASEENV_SXP: "baseenv"}


class UnsupportedRData(ValueError):
    """The file uses a part of R's format this reader does not implement."""


@dataclass(frozen=True)
class RSymbol:
    name: str


@dataclass
class RVector:
    kind: str  # logical, integer, double, character, list, raw
    data: Any
    attributes: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class RPairlist:
    items: list[tuple[str | None, Any]]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RMarker:
    """An environment or other reference object; its contents are not kept."""
    name: str


def _decompress(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


class _Reader:
    def __init__(self, data: bytes):
        self.buf = memoryview(data)
        self.pos = 0
        self.refs: list[Any] = []
        self.native_encoding = "utf-8"

    def int(self) -> int:
        value = struct.unpack_from(">i", self.buf, self.pos)[0]
        self.pos += 4
        return value

    def length(self) -> int:
        n = self.int()
        if n == -1:  # a long vector: two more words
            upper, lower = self.int(), self.int()
            n = (upper << 32) + (lower & 0xFFFFFFFF)
        return n

    def take(self, n: int) -> memoryview:
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise UnsupportedRData("the stream ends inside an item")
        self.pos += n
        return out

    def header(self) -> None:
        if bytes(self.take(2)) != b"X\n":
            raise UnsupportedRData("only the binary XDR format is read (saveRDS/save default)")
        version = self.int()
        self.int()  # writer's R version
        self.int()  # oldest R version that can read it
        if version == 3:
            name = bytes(self.take(self.int())).decode("ascii")
            self.native_encoding = {"utf8": "utf-8", "": "utf-8"}.get(name.lower(), name.lower())
        elif version != 2:
            raise UnsupportedRData("serialization version %d" % version)

    def decode(self, raw: bytes, levels: int) -> str:
        if levels & (UTF8_MASK | ASCII_MASK):
            return raw.decode("utf-8")
        if levels & LATIN1_MASK:
            return raw.decode("latin-1")
        if levels & BYTES_MASK:
            raise UnsupportedRData("a bytes-encoded string")
        return raw.decode(self.native_encoding)

    def charsxp(self, flags: int) -> str | None:
        n = self.int()
        if n == -1:
            return None  # NA_character_
        return self.decode(bytes(self.take(n)), flags >> 12)

    def strings(self, n: int) -> list[str | None]:
        out: list[str | None] = []
        buf, unpack = self.buf, struct.unpack_from
        for _ in range(n):  # inline CHARSXP reads: these vectors hold millions of strings
            flags, size = unpack(">ii", buf, self.pos)
            self.pos += 8
            if flags & 0xFF != CHARSXP:
                raise UnsupportedRData("a character vector element of type %d" % (flags & 0xFF))
            if size == -1:
                out.append(None)
                continue
            raw = bytes(buf[self.pos:self.pos + size])
            self.pos += size
            levels = flags >> 12
            out.append(raw.decode("utf-8") if levels & (UTF8_MASK | ASCII_MASK) else self.decode(raw, levels))
        return out

    def attributes(self) -> dict[str, Any]:
        attrs = self.item()
        if not isinstance(attrs, RPairlist):
            raise UnsupportedRData("attributes are not a pairlist")
        return {tag: value for tag, value in attrs.items}

    def item(self) -> Any:
        flags = self.int()
        kind = flags & 0xFF
        if kind == NILVALUE_SXP:
            return None
        if kind in _MARKERS:
            return RMarker(_MARKERS[kind])
        if kind == REFSXP:
            index = flags >> 8 or self.int()
            return self.refs[index - 1]
        if kind == SYMSXP:
            name = self.item()
            symbol = RSymbol(name if isinstance(name, str) else "")
            self.refs.append(symbol)
            return symbol
        if kind in (PERSISTSXP, PACKAGESXP, NAMESPACESXP):
            if self.int() != 0:
                raise UnsupportedRData("a reference name vector")
            names = [self.item() for _ in range(self.int())]
            marker = RMarker("%d:%s" % (kind, ",".join(str(n) for n in names)))
            self.refs.append(marker)
            return marker
        if kind == ENVSXP:
            marker = RMarker("environment")
            self.refs.append(marker)  # registered before its contents, as R does
            self.int()  # locked
            for _ in range(4):  # enclosure, frame, hash table, attributes
                self.item()
            return marker
        if kind in _PAIRLIST_TYPES:
            return self.pairlist(flags)
        if kind == ALTREP_SXP:
            return self.altrep()
        if kind == CHARSXP:
            return self.charsxp(flags)
        if kind in (LGLSXP, INTSXP):
            n = self.length()
            data = np.frombuffer(self.take(4 * n), dtype=">i4").astype(np.int32)
            out = RVector("logical" if kind == LGLSXP else "integer", data)
        elif kind == REALSXP:
            n = self.length()
            out = RVector("double", np.frombuffer(self.take(8 * n), dtype=">f8").astype(np.float64))
        elif kind == STRSXP:
            out = RVector("character", self.strings(self.length()))
        elif kind in (VECSXP, EXPRSXP):
            out = RVector("list", [self.item() for _ in range(self.length())])
        elif kind == RAWSXP:
            out = RVector("raw", bytes(self.take(self.length())))
        elif kind == S4SXP:
            out = RVector("list", [])
        else:
            raise UnsupportedRData("SEXP type %d" % kind)
        if flags & HAS_ATTR:
            out.attributes = self.attributes()
        return out

    def pairlist(self, flags: int) -> RPairlist:
        items, attrs = [], {}
        while True:  # walk the cdr chain without recursing on it
            if flags & HAS_ATTR or flags & 0xFF in (ATTRLANGSXP, ATTRLISTSXP):
                pair_attrs = self.attributes()
                attrs = attrs or pair_attrs
            tag = self.item() if flags & HAS_TAG else None
            items.append((tag.name if isinstance(tag, RSymbol) else tag, self.item()))
            flags = self.int()
            if flags & 0xFF == NILVALUE_SXP:
                return RPairlist(items, attrs)
            if flags & 0xFF not in _PAIRLIST_TYPES:
                self.pos -= 4
                tail = self.item()  # a dotted pair: rare, kept as the last item
                items.append((None, tail))
                return RPairlist(items, attrs)

    def altrep(self) -> Any:
        info, state, attrs = self.item(), self.item(), self.item()
        if not isinstance(info, RPairlist) or not info.items or not isinstance(info.items[0][1], RSymbol):
            raise UnsupportedRData("an ALTREP object without a class")
        name = info.items[0][1].name
        attributes = {tag: value for tag, value in attrs.items} if isinstance(attrs, RPairlist) else {}
        if name in ("compact_intseq", "compact_realseq"):
            n, start, step = (float(v) for v in state.data[:3])
            values = start + step * np.arange(int(n))
            kind, dtype = ("integer", np.int32) if name == "compact_intseq" else ("double", np.float64)
            return RVector(kind, values.astype(dtype), attributes)
        if name == "deferred_string":
            source = state.items[0][1] if isinstance(state, RPairlist) else state
            if not isinstance(source, RVector):
                raise UnsupportedRData("a deferred string without its vector")
            if source.kind == "integer":
                strings = [None if v == INT_NA else str(int(v)) for v in source.data]
            elif source.kind == "double":
                strings = [None if np.isnan(v) else repr(float(v)) for v in source.data]
            else:
                strings = list(source.data)
            return RVector("character", strings, attributes)
        if name.startswith("wrap_"):
            inner = state.data[0] if isinstance(state, RVector) else state.items[0][1]
            if attributes:
                inner.attributes = attributes
            return inner
        raise UnsupportedRData("ALTREP class %s" % name)


def _read(data: bytes) -> Any:
    reader = _Reader(data)
    reader.header()
    return reader.item()


def read_rds(path: str | Path) -> Any:
    """The one object a saveRDS() file holds."""
    return _read(_decompress(Path(path).read_bytes()))


def read_rda(path: str | Path) -> dict[str, Any]:
    """Every named object a save() file holds."""
    data = _decompress(Path(path).read_bytes())
    if data[:5] not in (b"RDX2\n", b"RDX3\n"):
        raise UnsupportedRData("not an R save() file")
    contents = _read(data[5:])
    if not isinstance(contents, RPairlist):
        raise UnsupportedRData("save() contents are not a pairlist")
    return {tag: value for tag, value in contents.items}


def _class(value: Any) -> list[str]:
    cls = value.attributes.get("class") if isinstance(value, RVector) else None
    return list(cls.data) if isinstance(cls, RVector) else []


def column_values(column: RVector) -> Any:
    """A column as Python data: factor codes become their level strings (None for NA)."""
    if "factor" in _class(column):
        levels = column.attributes["levels"].data
        return [None if code == INT_NA else levels[code - 1] for code in column.data]
    return column.data


def data_frame(value: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    """(columns, attributes of each column, number of rows) of a data.frame or tibble."""
    if not isinstance(value, RVector) or value.kind != "list" or "data.frame" not in _class(value):
        raise UnsupportedRData("not a data.frame")
    names = value.attributes["names"].data
    columns, attrs = {}, {}
    for name, column in zip(names, value.data):
        if not isinstance(column, RVector) or column.kind == "list":
            raise UnsupportedRData("column %s is not an atomic vector" % name)
        columns[name] = column_values(column)
        attrs[name] = column.attributes
    lengths = {len(c) for c in columns.values()}
    if len(lengths) > 1:
        raise UnsupportedRData("columns of different lengths")
    return columns, attrs, lengths.pop() if lengths else 0
