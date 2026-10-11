"""Explore-feed ranking for one store's posts (pure functions, no I/O).

score = (0.45 relevance + 0.25 engagement + 0.20 freshness + 0.10 momentum) x penalties,
then a diversity re-rank. Each part is in [0, 1]:

- relevance: how much the post's tagged products (and their categories) match
  what this customer bought (recent purchases weigh more), has in the cart, or
  liked in other posts; B's recommendation rank for those products adds to it
  (a post about an item the model expects them to buy next); hashtags they
  liked before add a little. Guests have no relevance; the other weights are
  rescaled.
- engagement: likes + 2 x comments + 0.05 x views, log-scaled against the
  store's most engaging post, so one viral post does not flatten the rest.
- freshness: exp(-age_days / 10).
- momentum: the share of a post's likes that came in the last 3 days.
- penalties: seen in the last 3 days x0.35, seen before that x0.7, already
  liked x0.6 -- the explore page should keep showing something new.
- diversity: picking greedily, a post that shares its main product or category
  with one of the last two picks is multiplied by 0.75.

Every ranked post carries a short reason for the screen ("내가 산 상품", ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PostSignals:
    post_id: str
    created_at: datetime
    items: tuple[str, ...]          # tagged products
    hashtags: tuple[str, ...]
    likes: int = 0
    comments: int = 0
    views: int = 0
    recent_likes: int = 0           # likes in the last 3 days


@dataclass
class ViewerSignals:
    item_affinity: dict[str, float] = field(default_factory=dict)   # bought (recency-weighted), cart, liked
    category_of: dict[str, str] = field(default_factory=dict)        # item -> main category
    model_rank: dict[str, int] = field(default_factory=dict)         # item -> 0-based rank in B's recommendation
    model_is_real: bool = False                                      # a model ranking, not a fallback
    hashtag_affinity: dict[str, float] = field(default_factory=dict)
    seen_at: dict[str, datetime] = field(default_factory=dict)
    liked: set[str] = field(default_factory=set)

    @property
    def is_guest(self) -> bool:
        return not (self.item_affinity or self.model_rank or self.hashtag_affinity)


@dataclass(frozen=True)
class Ranked:
    post_id: str
    score: float
    reason: str


def _norm(values: Mapping[str, float]) -> dict[str, float]:
    top = max(values.values(), default=0.0)
    return {k: v / top for k, v in values.items()} if top > 0 else {}


def purchase_affinity(purchases: Iterable[tuple[str, datetime]], now: datetime, half_life_days: float = 30.0) -> dict[str, float]:
    """item -> sum of 0.5^(age/half_life) over the customer's purchases of it."""
    out: dict[str, float] = {}
    for item, when in purchases:
        age = max((now - when).total_seconds() / 86400.0, 0.0)
        out[item] = out.get(item, 0.0) + 0.5 ** (age / half_life_days)
    return out


def _relevance(post: PostSignals, viewer: ViewerSignals, items: dict[str, float], cats: dict[str, float],
               tags: dict[str, float]) -> tuple[float, str]:
    if not post.items and not post.hashtags:
        return 0.0, ""
    item_part = max((items.get(i, 0.0) for i in post.items), default=0.0)
    cat_part = max((cats.get(viewer.category_of.get(i, ""), 0.0) for i in post.items), default=0.0)
    model_part = max((1.0 / (1.0 + viewer.model_rank[i]) for i in post.items if i in viewer.model_rank), default=0.0)
    tag_part = max((tags.get(h, 0.0) for h in post.hashtags), default=0.0)
    score = min(1.0, 0.55 * item_part + 0.20 * cat_part + 0.45 * model_part + 0.20 * tag_part)
    parts = [(item_part, "내가 산 상품"), (model_part * (1.0 if viewer.model_is_real else 0.6),
             "AI 추천 상품" if viewer.model_is_real else "많이 찾는 상품"),
             (cat_part * 0.8, "관심 카테고리"), (tag_part * 0.7, "관심 해시태그")]
    best = max(parts, key=lambda p: p[0])
    return score, best[1] if best[0] > 0 else ""


def rank(posts: Sequence[PostSignals], viewer: ViewerSignals, now: Optional[datetime] = None,
         limit: Optional[int] = None) -> list[Ranked]:
    now = now or datetime.now(timezone.utc)
    items, tags = _norm(viewer.item_affinity), _norm(viewer.hashtag_affinity)
    cat_raw: dict[str, float] = {}
    for item, w in viewer.item_affinity.items():
        cat = viewer.category_of.get(item)
        if cat:
            cat_raw[cat] = cat_raw.get(cat, 0.0) + w
    cats = _norm(cat_raw)
    raw_engagement = {p.post_id: p.likes + 2 * p.comments + 0.05 * p.views for p in posts}
    top_engagement = math.log1p(max(raw_engagement.values(), default=0.0)) or 1.0
    guest = viewer.is_guest
    weights = (0.0, 0.50, 0.35, 0.15) if guest else (0.45, 0.25, 0.20, 0.10)

    scored: list[tuple[PostSignals, float, str]] = []
    for p in posts:
        relevance, reason = (0.0, "") if guest else _relevance(p, viewer, items, cats, tags)
        engagement = math.log1p(raw_engagement[p.post_id]) / top_engagement
        age_days = max((now - p.created_at).total_seconds() / 86400.0, 0.0)
        freshness = math.exp(-age_days / 10.0)
        momentum = p.recent_likes / p.likes if p.likes else 0.0
        score = (weights[0] * relevance + weights[1] * engagement + weights[2] * freshness + weights[3] * momentum)
        seen = viewer.seen_at.get(p.post_id)
        if seen is not None:
            score *= 0.35 if now - seen < timedelta(days=3) else 0.7
        if p.post_id in viewer.liked:
            score *= 0.6
        if not reason:
            reason = ("새 게시물" if age_days < 3 else "인기 게시물" if engagement > 0.6 else "추천 게시물")
        scored.append((p, score, reason))

    # Greedy diversity re-rank.
    def main_key(p: PostSignals) -> tuple[str, str]:
        item = p.items[0] if p.items else ""
        return item, viewer.category_of.get(item, "")

    out: list[Ranked] = []
    recent: list[tuple[str, str]] = []
    pool = scored[:]
    while pool and (limit is None or len(out) < limit):
        def adjusted(entry):
            p, s, _ = entry
            item, cat = main_key(p)
            clash = any((item and item == ri) or (cat and cat == rc) for ri, rc in recent[-2:])
            return s * (0.75 if clash else 1.0)
        best = max(pool, key=lambda e: (adjusted(e), e[0].created_at, e[0].post_id))
        pool.remove(best)
        out.append(Ranked(best[0].post_id, round(adjusted(best), 6), best[2]))
        recent.append(main_key(best[0]))
    return out


def related(target: PostSignals, posts: Sequence[PostSignals], category_of: Mapping[str, str],
            co_likes: Mapping[str, int], limit: int = 6) -> list[str]:
    """Posts like `target`: shared products, categories and hashtags, plus customers who liked both."""
    def jac(a: Iterable[str], b: Iterable[str]) -> float:
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 0.0
    t_cats = {category_of.get(i, "") for i in target.items} - {""}
    top_co = max(co_likes.values(), default=0) or 1
    scored = []
    for p in posts:
        if p.post_id == target.post_id:
            continue
        cats = {category_of.get(i, "") for i in p.items} - {""}
        s = (1.0 * jac(target.items, p.items) + 0.6 * jac(t_cats, cats) + 0.5 * jac(target.hashtags, p.hashtags)
             + 0.8 * co_likes.get(p.post_id, 0) / top_co)
        if s > 0:
            scored.append((s, p.likes, p.post_id))
    scored.sort(reverse=True)
    return [pid for _, _, pid in scored[:limit]]
