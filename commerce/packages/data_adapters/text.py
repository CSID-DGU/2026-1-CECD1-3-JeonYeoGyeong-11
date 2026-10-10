"""Encoder input text for one product (data.md §3, model.md §2).

Historical adapters and live catalog_item share this builder, so an item that
goes through the service keeps the text it had in the lab (OQ03). Markers are
plain strings for the tokenizer and no special token is added. Case is kept
until the encoder is chosen, and nothing non-ASCII is dropped.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence
import unicodedata

from commerce.packages.contracts.types import Payload

Fields = Sequence[tuple[str, str | None]]  # (marker, raw value) in output order

CATEGORY_SEPARATOR = " > "


class EmptyProductText(ValueError):
    """Every text field is missing: quarantine the item instead of encoding ''."""


def normalize_field(value: str | None) -> str | None:
    """NFC and collapsed whitespace; missing or blank becomes None, never 'nan'/'None'."""
    if value is None:
        return None
    if not isinstance(value, str):
        # A float NaN from a CSV reader must not turn into the text 'nan'.
        raise TypeError("text field must be str or None, got %s" % type(value).__name__)
    text = " ".join(unicodedata.normalize("NFC", value).split())
    return text or None


def build_product_text(fields: Fields) -> str:
    parts = []
    for marker, value in fields:
        text = normalize_field(value)
        if text is not None:
            parts.append("%s %s" % (marker, text))
    if not parts:
        raise EmptyProductText("all product text fields are missing")
    return " ".join(parts)


def instacart_fields(product_name: str | None, aisle: str | None, department: str | None) -> Fields:
    return (("[NAME]", product_name), ("[AISLE]", aisle), ("[DEPT]", department))


def dunnhumby_fields(product_category: str | None, product_type: str | None,
                     package_size: str | None) -> Fields:
    return (("[CAT]", product_category), ("[TYPE]", product_type), ("[SIZE]", package_size))


def live_fields(title_text: str | None, description_text: str | None,
                category_path: Sequence[str] | None) -> Fields:
    path = None
    if category_path:
        segments = [s for s in (normalize_field(p) for p in category_path) if s is not None]
        path = CATEGORY_SEPARATOR.join(segments) or None
    return (("[NAME]", title_text), ("[DESC]", description_text), ("[CAT]", path))


def catalog_item_fields(item: Payload) -> Fields:
    """Fields of a validated catalog_item.v1, read by its source's rule.

    An Instacart item carries category_path [department, aisle] (instacart.py).
    Its description_text is ignored: Instacart has no description, so any value
    there was derived from aisle/department and would count them twice.
    """
    source = item["source"]
    if source == "live":
        return live_fields(item["title_text"], item["description_text"], item["category_path"])
    if source == "instacart":
        path = item["category_path"] or []
        if len(path) > 2:
            raise ValueError("instacart category_path is [department, aisle]")
        department = path[0] if path else None
        aisle = path[1] if len(path) == 2 else None
        return instacart_fields(item["title_text"], aisle, department)
    if source == "dunnhumby":
        # dunnhumby.catalog_item: path [department, product_category], title product_type
        # (product_category when the type is missing, so then there is no [TYPE]), size in description.
        path = item["category_path"] or []
        category = path[-1] if path else None
        product_type = None if item["title_text"] == category else item["title_text"]
        return dunnhumby_fields(category, product_type, item["description_text"])
    raise ValueError("unknown catalog_item source %r" % source)


def catalog_item_text(item: Payload) -> str:
    return build_product_text(catalog_item_fields(item))


def _hangul(text: str) -> int:
    return sum(1 for ch in text if unicodedata.name(ch, "").startswith("HANGUL"))


def _duplicate_share(keys: Sequence[object]) -> float:
    if not keys:
        return 0.0
    counts = Counter(keys)
    return sum(1 for key in keys if counts[key] > 1) / len(keys)


@dataclass(frozen=True)
class TextAudit:
    items: int
    empty: int  # quarantined: every field missing
    missing_by_marker: dict[str, int]
    duplicate_raw: float  # share of items whose raw field values equal another item's
    duplicate_text: float  # the same after normalization, over items that produced text
    hangul_in: int  # Hangul characters in the NFC raw values
    hangul_out: int  # Hangul characters in the built texts


def audit_texts(fields_by_item: Mapping[str, Fields]) -> TextAudit:
    """Raw-vs-normalized quality counts. Token truncation joins once the encoder is fixed."""
    missing: Counter[str] = Counter()
    raw_keys, texts = [], []
    empty = hangul_in = 0
    for fields in fields_by_item.values():
        raw_keys.append(tuple(value for _, value in fields))
        for marker, value in fields:
            if normalize_field(value) is None:
                missing[marker] += 1
            elif value is not None:
                hangul_in += _hangul(unicodedata.normalize("NFC", value))
        try:
            texts.append(build_product_text(fields))
        except EmptyProductText:
            empty += 1
    return TextAudit(
        items=len(fields_by_item), empty=empty, missing_by_marker=dict(sorted(missing.items())),
        duplicate_raw=_duplicate_share(raw_keys), duplicate_text=_duplicate_share(texts),
        hangul_in=hangul_in, hangul_out=sum(_hangul(t) for t in texts),
    )
