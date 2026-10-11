"""Post media: seller uploads and generated demo graphics, stored per seller.

Files live next to the seller's orders.sqlite in media/ (A-only, Git-ignored
like the DB). A post's media list is JSON entries {"type", "name"}:
- "image" / "video": a seller upload. The type is read from the file's magic
  bytes, never from its name or the browser's content type; anything else is
  refused. Uploaded SVG is refused on purpose (it can carry script).
- "motion": an animated SVG this module generated for the demo (there are no
  real product photos or videos in the demo data). It plays as an <img>,
  needs no codec or network, and is served with a CSP that forbids script.
"""
from __future__ import annotations

import hashlib
import html
import re
import uuid
from pathlib import Path
from typing import Optional

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_FILES_PER_POST = 10
NAME_RE = re.compile(r"^[a-z0-9-]{8,80}\.(jpg|png|gif|webp|mp4|webm|mov|svg)$")
CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp",
                 "mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime", "svg": "image/svg+xml"}
SVG_HEADERS = {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
               "X-Content-Type-Options": "nosniff"}


class MediaError(ValueError):
    """An upload this module will not store (shown to the seller as is)."""


def media_dir(merchant_db_path: Path) -> Path:
    return merchant_db_path.parent / "media"


def sniff(data: bytes) -> Optional[tuple[str, str]]:
    """(media type, extension) from the first bytes, or None."""
    if data[:3] == b"\xff\xd8\xff":
        return "image", "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image", "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image", "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image", "webp"
    if data[4:8] == b"ftyp":
        return "video", "mov" if data[8:10] == b"qt" else "mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video", "webm"
    return None


def save_upload(folder: Path, data: bytes) -> dict[str, str]:
    if not data:
        raise MediaError("빈 파일입니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise MediaError("파일은 %dMB까지 올릴 수 있습니다." % (MAX_UPLOAD_BYTES // (1024 * 1024)))
    kind = sniff(data)
    if kind is None:
        raise MediaError("사진(JPG·PNG·GIF·WebP)이나 영상(MP4·WebM·MOV)만 올릴 수 있습니다.")
    folder.mkdir(parents=True, exist_ok=True)
    name = "up-%s.%s" % (uuid.uuid4().hex, kind[1])
    (folder / name).write_bytes(data)
    return {"type": kind[0], "name": name}


def resolve(folder: Path, name: str) -> Optional[Path]:
    """The file for a media name, or None for anything that is not a plain stored name."""
    if not NAME_RE.match(name):
        return None
    path = folder / name
    return path if path.is_file() else None


# --- generated demo graphics ----------------------------------------------------

_FONT = "Pretendard Variable, Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif"


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _lines(text: str, width: int, limit: int) -> list[str]:
    """Greedy wrap by character count (Korean has no reliable word spacing for this)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w[:width]
    if cur:
        lines.append(cur)
    return lines[:limit]


def photo_svg(title: str, subtitle: str, emoji: str, tone: tuple[str, str], badge: str = "") -> str:
    """A square product-photo stand-in: soft gradient, large emoji, caption card."""
    light, deep = tone
    title_lines = _lines(title, 15, 2)
    title_svg = "".join('<tspan x="96" dy="%d">%s</tspan>' % (0 if i == 0 else 72, _e(t)) for i, t in enumerate(title_lines))
    sub_y = 836 + 72 * len(title_lines) + 6
    badge_svg = ('<g><rect x="72" y="72" rx="34" height="68" width="%d" fill="#ffffff" fill-opacity=".85"/>'
                 '<text x="106" y="118" font-size="34" font-weight="700" fill="#2f6627" font-family="%s">%s</text></g>'
                 % (40 + 30 * len(badge), _FONT, _e(badge))) if badge else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{light}"/><stop offset="1" stop-color="{deep}"/></linearGradient></defs>
<rect width="1080" height="1080" fill="url(#g)"/>
<circle cx="880" cy="200" r="190" fill="#ffffff" fill-opacity=".25"/><circle cx="160" cy="620" r="120" fill="#ffffff" fill-opacity=".18"/>
{badge_svg}
<text x="540" y="470" font-size="360" text-anchor="middle" dominant-baseline="middle">{_e(emoji)}</text>
<rect x="56" y="744" width="968" height="{sub_y + 48 - 744}" rx="40" fill="#ffffff" fill-opacity=".9"/>
<text y="836" font-size="60" font-weight="800" fill="#18231b" font-family="{_FONT}">{title_svg}</text>
<text x="96" y="{sub_y}" font-size="38" fill="#4b5a4f" font-family="{_FONT}">{_e(subtitle)}</text>
</svg>"""


def reel_svg(title: str, lines: list[str], emoji: str, tone: tuple[str, str], seconds: int = 9) -> str:
    """A 9:16 animated short-form stand-in: bobbing emoji, captions that take turns, a progress bar."""
    light, deep = tone
    n = max(1, len(lines))
    step = seconds / n
    caption_css = "".join(
        f".c{i}{{opacity:0;animation:cap {seconds}s linear infinite;animation-delay:{i * step:.2f}s}}" for i in range(n))
    keyframe_end = 100 / n
    captions = "".join(
        f'<text class="c{i}" x="540" y="1460" font-size="66" font-weight="800" text-anchor="middle" fill="#ffffff" '
        f'font-family="{_FONT}">{_e(t)}</text>' for i, t in enumerate(lines or [title]))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1920" width="1080" height="1920">
<style>
.bob{{animation:bob 2.4s ease-in-out infinite;transform-origin:540px 820px}}
@keyframes bob{{0%,100%{{transform:translateY(0) rotate(-4deg)}}50%{{transform:translateY(-60px) rotate(4deg)}}}}
.pulse{{opacity:.2;animation:pulse 3s ease-in-out infinite both}}
@keyframes pulse{{0%,100%{{opacity:.15}}50%{{opacity:.4}}}}
.bar{{animation:bar {seconds}s linear infinite;transform-origin:60px 70px}}
@keyframes bar{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
@keyframes cap{{0%{{opacity:0}}2%{{opacity:1}}{keyframe_end - 2:.2f}%{{opacity:1}}{keyframe_end:.2f}%{{opacity:0}}100%{{opacity:0}}}}
{caption_css}
</style>
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{light}"/><stop offset=".55" stop-color="{deep}"/><stop offset="1" stop-color="#244e20"/></linearGradient></defs>
<rect width="1080" height="1920" fill="url(#g)"/>
<circle class="pulse" cx="260" cy="420" r="210" fill="#ffffff"/><circle class="pulse" cx="860" cy="1080" r="260" fill="#ffffff" style="animation-delay:1.5s"/>
<rect x="60" y="60" width="960" height="12" rx="6" fill="#ffffff" fill-opacity=".35"/>
<rect class="bar" x="60" y="60" width="960" height="12" rx="6" fill="#ffffff"/>
<g class="bob"><text x="540" y="820" font-size="460" text-anchor="middle" dominant-baseline="middle">{_e(emoji)}</text></g>
<text x="540" y="1300" font-size="56" font-weight="700" text-anchor="middle" fill="#ffffff" fill-opacity=".85" font-family="{_FONT}">{_e(title[:22])}</text>
{captions}
<text x="540" y="1800" font-size="36" text-anchor="middle" fill="#ffffff" fill-opacity=".7" font-family="{_FONT}">데모 영상 대체 그래픽</text>
</svg>"""


def write_generated(folder: Path, svg: str, kind: str) -> dict[str, str]:
    """Store a generated SVG under a content-addressed name; kind is "image" or "motion"."""
    folder.mkdir(parents=True, exist_ok=True)
    name = "gen-%s.svg" % hashlib.sha256(svg.encode("utf-8")).hexdigest()[:32]
    path = folder / name
    if not path.exists():
        path.write_text(svg, encoding="utf-8")
    return {"type": kind, "name": name}
