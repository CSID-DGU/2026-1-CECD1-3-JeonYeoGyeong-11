"""Display helpers shared by the screens and the demo media generator (no I/O).

There are no product photos in the demo data; thumbnails and generated post
graphics show an emoji picked from the product title (checked first: a
category such as "과일" is too coarse to tell strawberries from tangerines),
then from the catalog category, or nothing.
"""
from __future__ import annotations

import hashlib
from typing import Optional

_TITLE_EMOJI = (
    ("딸기", "🍓"), ("사과", "🍎"), ("감귤", "🍊"), ("한라봉", "🍊"), ("귤", "🍊"), ("토마토", "🍅"),
    ("상추", "🥬"), ("배추", "🥬"), ("오이", "🥒"), ("감자", "🥔"), ("고구마", "🍠"), ("당근", "🥕"),
    ("양파", "🧅"), ("마늘", "🧄"), ("버섯", "🍄"), ("옥수수", "🌽"), ("포도", "🍇"), ("복숭아", "🍑"),
    ("수박", "🍉"), ("참외", "🍈"), ("블루베리", "🫐"), ("키위", "🥝"), ("바나나", "🍌"), ("레몬", "🍋"),
    ("고등어", "🐟"), ("갈치", "🐟"), ("광어", "🐟"), ("연어", "🐟"), ("한치", "🦑"), ("오징어", "🦑"),
    ("문어", "🐙"), ("전복", "🐚"), ("소라", "🐚"), ("굴", "🦪"), ("미역", "🌿"), ("김 ", "🌿"),
    ("새우", "🦐"), ("게", "🦀"), ("치즈", "🧀"), ("요거트", "🥣"), ("그래놀라", "🥣"), ("베이글", "🥯"),
    ("크루아상", "🥐"), ("케이크", "🍰"), ("쿠키", "🍪"), ("스콘", "🥯"), ("녹차", "🍵"), ("말차", "🍵"),
    ("주스", "🧃"), ("콜드브루", "🧋"), ("라떼", "☕"), ("원두", "☕"), ("오겹살", "🥓"), ("삼겹살", "🥓"),
    ("한우", "🥩"), ("소고기", "🥩"), ("닭", "🍗"), ("오리", "🍗"), ("소시지", "🌭"), ("꿀", "🍯"),
    ("잼", "🍯"), ("두부", "🧈"), ("버터", "🧈"), ("우유", "🥛"), ("계란", "🥚"), ("달걀", "🥚"),
    ("쌀", "🍚"), ("현미", "🍚"), ("떡", "🍡"), ("만두", "🥟"), ("김치", "🥬"), ("식빵", "🍞"),
)
_CATEGORY_EMOJI = (
    ("우유", "🥛"), ("계란", "🥚"), ("유제품", "🧀"), ("생선", "🐟"), ("수산", "🐟"),
    ("커피", "☕"), ("음료", "🧃"), ("빵", "🍞"), ("베이커리", "🥐"), ("과일", "🍊"),
    ("채소", "🥬"), ("쌀", "🍚"), ("축산", "🥩"), ("정육", "🥩"), ("농산", "🌽"), ("반찬", "🍱"),
)

# Soft backgrounds for generated tiles: (light, deep) pairs that sit with the cucumber palette.
TONES = (
    ("#f1f7ea", "#c8e3ad"), ("#f3f6e4", "#dfe9bf"), ("#eaf5ef", "#cbe6d6"), ("#f6f2e4", "#eadfc0"),
    ("#fdf0e6", "#f6d3b8"), ("#eef2fb", "#cfd9f2"), ("#fbeff2", "#f1cfd8"), ("#f2f0fb", "#d8d2f2"),
)


def product_emoji(category_path: Optional[list[str]], title: Optional[str] = None) -> Optional[str]:
    for keyword, emoji in _TITLE_EMOJI:
        if title and keyword in title:
            return emoji
    for keyword, emoji in _CATEGORY_EMOJI:
        if keyword in (category_path or []):
            return emoji
    return None


def tone_for(key: str) -> tuple[str, str]:
    """A stable background pair for any key (same post -> same colors on every render)."""
    return TONES[int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(TONES)]


def mask_name(name: str | None) -> str:
    """Shown next to reviews and comments: 김하준 -> 김*준, 이준 -> 이*, a login id -> its first letters."""
    name = (name or "").strip()
    if not name:
        return "고객"
    if len(name) <= 2:
        return name[0] + "*"
    if name.isascii():
        return name[:3] + "*" * min(4, len(name) - 3)
    return name[0] + "*" * (len(name) - 2) + name[-1]
