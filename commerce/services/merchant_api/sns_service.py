"""Store SNS: seller posts (photos, reels, captions, hashtags, product tags),
customer likes / comments / views, and a ranked explore feed.

The explore feed ranks this store's posts for one customer (feed_ranking.py)
using signals that already live on the seller's server: their completed
orders, cart, likes and views, and B's recommendation for them (the same
predict_local call as the home screen; A's baseline when B cannot answer).
Nothing here is sent anywhere else.

A post needs at least one media item. When the seller uploads none, a graphic
is generated from the first tagged product (a still for a photo post, an
animated one for a reel), so a store without photos or videos can still post.
"""
from __future__ import annotations

import datetime as dt
import re
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import cart_db, feed_ranking, media, orders_db, orders_service, shop_db, sns_db
from commerce.services.merchant_api.presentation import product_emoji, tone_for

KINDS = {"article": "게시물", "short_video": "릴스"}
MAX_CAPTION = 2200
MAX_TAGGED = 5
HASHTAG_RE = re.compile(r"#([0-9A-Za-z_가-힣]{1,30})")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def parse_hashtags(*texts: Optional[str]) -> list[str]:
    seen: list[str] = []
    for text in texts:
        for tag in HASHTAG_RE.findall(text or ""):
            if tag not in seen:
                seen.append(tag)
    return seen[:15]


def _catalog(conn, seller_id: str) -> dict[str, dict[str, Any]]:
    return {i["item_id_local"]: i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}


def _category(item: dict[str, Any]) -> str:
    path = item.get("category_path") or []
    return path[-1] if path else ""


def _check_items(catalog: dict[str, dict], item_ids: Sequence[str]) -> list[str]:
    items = list(dict.fromkeys(i for i in item_ids if i))
    if len(items) > MAX_TAGGED:
        raise ContractError("SCHEMA_INVALID", "/item_ids")
    for i in items:
        if i not in catalog:
            raise ContractError("NOT_FOUND", "/item_ids")
    return items


def generated_media(folder: Path, kind: str, item: Optional[dict[str, Any]], caption: str, key: str) -> dict[str, str]:
    """The stand-in graphic for a post without uploaded media."""
    title = (item or {}).get("title_text") or (caption.splitlines()[0] if caption else "오늘의 소식")
    emoji = product_emoji((item or {}).get("category_path"), title) or "🥒"
    tone = tone_for(key)
    if kind == "short_video":
        lines = [l.strip() for l in re.split(r"[\n.!?]", caption) if l.strip() and not l.strip().startswith("#")][:4]
        return media.write_generated(folder, media.reel_svg(title, [l[:18] for l in lines] or [title[:18]], emoji, tone), "motion")
    price = "%s원" % format(item["display_price_minor"], ",") if item else ""
    tags = " ".join("#" + t for t in parse_hashtags(caption)[:2])
    return media.write_generated(folder, media.photo_svg(title, " · ".join(x for x in (price, tags) if x), emoji, tone), "image")


def create_post(conn, folder: Path, *, seller_id: str, kind: str, caption: str, item_ids: Sequence[str] = (),
                uploads: Sequence[bytes] = (), created_at: Optional[str] = None) -> dict[str, Any]:
    if kind not in KINDS:
        raise ContractError("INVALID_ENUM_VALUE", "/kind")
    caption = (caption or "").strip()
    if not caption:
        raise ContractError("MISSING_REQUIRED_FIELD", "/caption")
    if len(caption) > MAX_CAPTION:
        raise ContractError("SCHEMA_INVALID", "/caption")
    if len(uploads) > media.MAX_FILES_PER_POST:
        raise ContractError("SCHEMA_INVALID", "/media")
    catalog = _catalog(conn, seller_id)
    items = _check_items(catalog, item_ids)
    stored = [media.save_upload(folder, data) for data in uploads if data]  # MediaError -> caller
    post_id = uuid.uuid4().hex
    if not stored:
        stored = [generated_media(folder, kind, catalog.get(items[0]) if items else None, caption, post_id)]
    title = caption.splitlines()[0][:60]
    with conn:
        sns_db.insert_post(conn, seller_id, post_id, kind, title, caption, stored, parse_hashtags(caption),
                           created_at or _iso(_now()))
        sns_db.set_post_products(conn, seller_id, post_id, items)
    return sns_db.fetch_post(conn, seller_id, post_id)


def update_post(conn, *, seller_id: str, post_id: str, caption: str, item_ids: Sequence[str]) -> None:
    if sns_db.fetch_post(conn, seller_id, post_id) is None:
        raise ContractError("NOT_FOUND", "/post_id")
    caption = (caption or "").strip()
    if not caption or len(caption) > MAX_CAPTION:
        raise ContractError("SCHEMA_INVALID", "/caption")
    items = _check_items(_catalog(conn, seller_id), item_ids)
    with conn:
        sns_db.update_post(conn, seller_id, post_id, caption.splitlines()[0][:60], caption, parse_hashtags(caption),
                           _iso(_now()))
        sns_db.set_post_products(conn, seller_id, post_id, items)


def delete_post(conn, *, seller_id: str, post_id: str) -> None:
    with conn:
        if not sns_db.soft_delete_post(conn, seller_id, post_id, _iso(_now())):
            raise ContractError("NOT_FOUND", "/post_id")


def toggle_like(conn, *, seller_id: str, post_id: str, customer_id_local: str, at: Optional[str] = None) -> bool:
    if sns_db.fetch_post(conn, seller_id, post_id) is None:
        raise ContractError("NOT_FOUND", "/post_id")
    with conn:
        return sns_db.toggle_like(conn, seller_id, post_id, customer_id_local, at or _iso(_now()))


def add_comment(conn, *, seller_id: str, post_id: str, body: str, customer_id_local: Optional[str] = None,
                at: Optional[str] = None) -> None:
    """customer_id_local None means the seller is replying."""
    if sns_db.fetch_post(conn, seller_id, post_id) is None:
        raise ContractError("NOT_FOUND", "/post_id")
    body = (body or "").strip()
    if not body or len(body) > 500:
        raise ContractError("SCHEMA_INVALID", "/body")
    with conn:
        sns_db.insert_comment(conn, seller_id, uuid.uuid4().hex, post_id,
                              "seller" if customer_id_local is None else "customer", customer_id_local, body,
                              at or _iso(_now()))


def delete_comment(conn, *, seller_id: str, comment_id: str, customer_id_local: Optional[str] = None) -> None:
    with conn:
        if not sns_db.delete_comment(conn, seller_id, comment_id, customer_id_local):
            raise ContractError("NOT_FOUND", "/comment_id")


def record_view(conn, *, seller_id: str, post_id: str, viewer: str, at: Optional[str] = None) -> None:
    with conn:
        sns_db.record_view(conn, seller_id, post_id, viewer, at or _iso(_now()))


# --- reading -----------------------------------------------------------------

def _signals(conn, seller_id: str, posts: list[dict], now: dt.datetime) -> dict[str, feed_ranking.PostSignals]:
    tags = sns_db.post_products(conn, seller_id)
    likes, comments, views = (sns_db.like_counts(conn, seller_id), sns_db.comment_counts(conn, seller_id),
                              sns_db.view_counts(conn, seller_id))
    recent: dict[str, int] = {}
    cutoff = now - dt.timedelta(days=3)
    for like in sns_db.likes_with_time(conn, seller_id):
        if _parse(like["created_at"]) >= cutoff:
            recent[like["post_id"]] = recent.get(like["post_id"], 0) + 1
    return {p["post_id"]: feed_ranking.PostSignals(
        p["post_id"], _parse(p["created_at"]), tuple(tags.get(p["post_id"], ())), tuple(p["hashtags"]),
        likes.get(p["post_id"], 0), comments.get(p["post_id"], 0), views.get(p["post_id"], 0),
        recent.get(p["post_id"], 0)) for p in posts}


def viewer_signals(conn, *, seller_id: str, customer_id_local: Optional[str], catalog: dict[str, dict],
                   posts_by_id: dict[str, dict], tags: dict[str, list[str]], runtime=None,
                   now: Optional[dt.datetime] = None) -> feed_ranking.ViewerSignals:
    now = now or _now()
    viewer = feed_ranking.ViewerSignals(category_of={i: _category(it) for i, it in catalog.items()})
    if not customer_id_local:
        return viewer
    purchases = []
    for order in orders_db.list_orders_by_customer(conn, seller_id, customer_id_local):
        if order["status"] == "completed" and order.get("completed_at"):
            when = _parse(order["completed_at"])
            purchases += [(i["item_id_local"], when) for i in order["items"]]
    viewer.item_affinity = feed_ranking.purchase_affinity(purchases, now)
    for line in cart_db.list_items(conn, seller_id, customer_id_local):
        viewer.item_affinity[line["item_id_local"]] = viewer.item_affinity.get(line["item_id_local"], 0.0) + 0.5
    for item in shop_db.wishlist(conn, seller_id, customer_id_local):
        viewer.item_affinity[item] = viewer.item_affinity.get(item, 0.0) + 0.6
    viewer.liked = sns_db.liked_by(conn, seller_id, customer_id_local)
    for post_id in viewer.liked:
        for item in tags.get(post_id, ()):
            viewer.item_affinity[item] = viewer.item_affinity.get(item, 0.0) + 0.4
        for tag in (posts_by_id.get(post_id) or {}).get("hashtags", ()):
            viewer.hashtag_affinity[tag] = viewer.hashtag_affinity.get(tag, 0.0) + 1.0
    viewer.seen_at = {p: _parse(t) for p, t in sns_db.viewed_by(conn, seller_id, customer_id_local).items()}
    rec = orders_service.get_recommendations_for_display(conn, seller_id=seller_id, customer_id_local=customer_id_local,
                                                         top_n=20, runtime=runtime)
    viewer.model_rank = {e["item_id_local"]: n for n, e in enumerate(rec["items"])}
    viewer.model_is_real = orders_service.recommendation_label(rec) is None
    return viewer


def _decorate(post: dict, seller_id: str, catalog: dict[str, dict], tags: dict[str, list[str]],
              signals: feed_ranking.PostSignals, liked: set[str]) -> dict[str, Any]:
    out = dict(post)
    out["media"] = [dict(m, url="/media/%s/%s" % (seller_id, m["name"])) for m in post["media"]]
    out["items"] = [catalog[i] for i in tags.get(post["post_id"], ()) if i in catalog]
    out.update(likes=signals.likes, comments=signals.comments, views=signals.views,
               liked_by_me=post["post_id"] in liked, kind_label=KINDS.get(post["kind"], post["kind"]))
    return out


def explore(conn, *, seller_id: str, customer_id_local: Optional[str], runtime=None, limit: Optional[int] = None,
            now: Optional[dt.datetime] = None) -> list[dict[str, Any]]:
    now = now or _now()
    posts = sns_db.list_posts(conn, seller_id)
    if not posts:
        return []
    catalog, tags = _catalog(conn, seller_id), sns_db.post_products(conn, seller_id)
    by_id = {p["post_id"]: p for p in posts}
    signals = _signals(conn, seller_id, posts, now)
    viewer = viewer_signals(conn, seller_id=seller_id, customer_id_local=customer_id_local, catalog=catalog,
                            posts_by_id=by_id, tags=tags, runtime=runtime, now=now)
    ranked = feed_ranking.rank(list(signals.values()), viewer, now=now, limit=limit)
    return [dict(_decorate(by_id[r.post_id], seller_id, catalog, tags, signals[r.post_id], viewer.liked),
                 reason=r.reason, score=r.score) for r in ranked]


def post_detail(conn, *, seller_id: str, post_id: str, customer_id_local: Optional[str]) -> Optional[dict[str, Any]]:
    posts = sns_db.list_posts(conn, seller_id)
    by_id = {p["post_id"]: p for p in posts}
    if post_id not in by_id:
        return None
    now = _now()
    catalog, tags = _catalog(conn, seller_id), sns_db.post_products(conn, seller_id)
    signals = _signals(conn, seller_id, posts, now)
    liked = sns_db.liked_by(conn, seller_id, customer_id_local) if customer_id_local else set()
    co_likes: dict[str, int] = {}
    likers = {l["customer_id_local"] for l in sns_db.likes_with_time(conn, seller_id) if l["post_id"] == post_id}
    for like in sns_db.likes_with_time(conn, seller_id):
        if like["customer_id_local"] in likers and like["post_id"] != post_id:
            co_likes[like["post_id"]] = co_likes.get(like["post_id"], 0) + 1
    related_ids = feed_ranking.related(signals[post_id], list(signals.values()),
                                       {i: _category(it) for i, it in catalog.items()}, co_likes)
    post = _decorate(by_id[post_id], seller_id, catalog, tags, signals[post_id], liked)
    post["comment_list"] = sns_db.list_comments(conn, seller_id, post_id)
    post["related"] = [_decorate(by_id[r], seller_id, catalog, tags, signals[r], liked) for r in related_ids]
    return post


def posts_for_item(conn, *, seller_id: str, item_id_local: str, limit: int = 6) -> list[dict[str, Any]]:
    posts = sns_db.list_posts(conn, seller_id)
    catalog, tags = _catalog(conn, seller_id), sns_db.post_products(conn, seller_id)
    signals = _signals(conn, seller_id, posts, _now())
    hits = [p for p in posts if item_id_local in tags.get(p["post_id"], ())]
    hits.sort(key=lambda p: (signals[p["post_id"]].likes, p["created_at"]), reverse=True)
    return [_decorate(p, seller_id, catalog, tags, signals[p["post_id"]], set()) for p in hits[:limit]]


def insights(conn, *, seller_id: str) -> list[dict[str, Any]]:
    """Per post: reach and engagement, plus orders of its tagged products placed after it went up."""
    posts = sns_db.list_posts(conn, seller_id)
    catalog, tags = _catalog(conn, seller_id), sns_db.post_products(conn, seller_id)
    signals = _signals(conn, seller_id, posts, _now())
    orders = orders_db.list_orders(conn, seller_id)
    out = []
    for p in posts:
        items = set(tags.get(p["post_id"], ()))
        after = [o for o in orders if o["created_at"] >= p["created_at"] and o["status"] != "cancelled"
                 and items & {i["item_id_local"] for i in o["items"]}]
        row = _decorate(p, seller_id, catalog, tags, signals[p["post_id"]], set())
        row["orders_after"] = len(after)
        row["engagement_rate"] = round(100 * (row["likes"] + row["comments"]) / row["views"], 1) if row["views"] else None
        out.append(row)
    return out
