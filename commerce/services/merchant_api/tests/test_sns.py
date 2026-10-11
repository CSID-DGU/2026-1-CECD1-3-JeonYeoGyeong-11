"""SNS feed: the explore ranking (pure), media checks, posts/likes/comments/views and insights,
and the additive migration that lets an older orders.sqlite pick up the new columns."""
from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import (cart_db, feed_ranking as fr, media, orders_db, orders_service, shop_db, sns_db,
                                            sns_service, social_db)
from commerce.services.merchant_api.migrate import add_columns

SELLER = "seller-sns"
NOW = dt.datetime(2026, 10, 8, 12, 0, tzinfo=dt.timezone.utc)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _post(post_id, days_old, items=(), tags=(), likes=0, comments=0, views=0, recent=0):
    return fr.PostSignals(post_id, NOW - dt.timedelta(days=days_old), tuple(items), tuple(tags), likes, comments, views, recent)


class RankingTest(unittest.TestCase):
    def test_guest_sees_engaging_and_fresh_posts_first(self):
        posts = [_post("old-quiet", 40), _post("new", 0.5), _post("popular", 5, likes=50, comments=10, views=400)]
        order = [r.post_id for r in fr.rank(posts, fr.ViewerSignals(), now=NOW)]
        self.assertEqual(order[-1], "old-quiet")
        self.assertEqual(set(order[:2]), {"new", "popular"})

    def test_buyer_sees_posts_about_what_they_buy_first(self):
        posts = [_post("fish", 2, items=["sku-fish"], likes=1), _post("bread", 2, items=["sku-bread"], likes=30, views=300)]
        viewer = fr.ViewerSignals(item_affinity={"sku-fish": 2.0}, category_of={"sku-fish": "수산", "sku-bread": "식품"})
        ranked = fr.rank(posts, viewer, now=NOW)
        self.assertEqual(ranked[0].post_id, "fish")
        self.assertEqual(ranked[0].reason, "내가 산 상품")

    def test_model_ranking_is_labelled_only_when_it_is_a_real_model(self):
        posts = [_post("p", 1, items=["sku-egg"])]
        real = fr.rank(posts, fr.ViewerSignals(model_rank={"sku-egg": 0}, model_is_real=True), now=NOW)
        popular = fr.rank(posts, fr.ViewerSignals(model_rank={"sku-egg": 0}, model_is_real=False), now=NOW)
        self.assertEqual(real[0].reason, "AI 추천 상품")
        self.assertEqual(popular[0].reason, "많이 찾는 상품")

    def test_recently_seen_posts_sink(self):
        posts = [_post("a", 1, likes=5), _post("b", 1, likes=5)]
        viewer = fr.ViewerSignals(item_affinity={"x": 1.0}, seen_at={"a": NOW - dt.timedelta(hours=2)})
        self.assertEqual([r.post_id for r in fr.rank(posts, viewer, now=NOW)], ["b", "a"])

    def test_diversity_keeps_one_product_from_filling_the_top(self):
        posts = [_post("f1", 1, items=["fish"], likes=20), _post("f2", 1, items=["fish"], likes=19),
                 _post("f3", 1, items=["fish"], likes=18), _post("egg", 1, items=["egg"], likes=12)]
        viewer = fr.ViewerSignals(item_affinity={"zzz": 1.0}, category_of={"fish": "수산", "egg": "축산"})
        top3 = [r.post_id for r in fr.rank(posts, viewer, now=NOW)][:3]
        self.assertIn("egg", top3)

    def test_purchase_affinity_halves_every_half_life(self):
        aff = fr.purchase_affinity([("a", NOW), ("b", NOW - dt.timedelta(days=30))], NOW)
        self.assertAlmostEqual(aff["a"], 1.0)
        self.assertAlmostEqual(aff["b"], 0.5)

    def test_related_prefers_shared_products_and_co_likes(self):
        target = _post("t", 1, items=["fish"], tags=["제철"])
        others = [target, _post("same", 1, items=["fish"]), _post("tag", 1, tags=["제철"]), _post("none", 1, items=["egg"])]
        out = fr.related(target, others, {"fish": "수산", "egg": "축산"}, co_likes={"none": 0})
        self.assertEqual(out[0], "same")
        self.assertNotIn("t", out)
        self.assertNotIn("none", out)


class MediaTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)

    def test_upload_type_comes_from_bytes_not_the_name(self):
        self.assertEqual(media.save_upload(self.folder, PNG)["type"], "image")
        self.assertEqual(media.save_upload(self.folder, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8)["type"], "video")
        for bad in (b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"MZ\x90\x00", b""):
            with self.assertRaises(media.MediaError):
                media.save_upload(self.folder, bad)

    def test_resolve_only_returns_stored_names(self):
        stored = media.save_upload(self.folder, PNG)["name"]
        self.assertIsNotNone(media.resolve(self.folder, stored))
        for name in ("../orders.sqlite", "up-x.png", "a/b.png", "evil.html"):
            self.assertIsNone(media.resolve(self.folder, name))

    def test_generated_graphics_escape_text_and_are_content_addressed(self):
        svg = media.photo_svg("<script>alert(1)</script>", "1,000원", "🥒", ("#000", "#fff"))
        self.assertNotIn("<script>", svg)
        a = media.write_generated(self.folder, svg, "image")
        b = media.write_generated(self.folder, svg, "image")
        self.assertEqual(a, b)
        self.assertIn("@keyframes", media.reel_svg("릴스", ["한 줄"], "🐟", ("#000", "#fff")))


class SnsServiceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name) / "media"
        self.conn = orders_db.connect(Path(tmp.name) / "orders.sqlite", schemas=(sns_db.ensure_schema, shop_db.ensure_schema, cart_db.ensure_schema))
        self.addCleanup(self.conn.close)
        for item, title, path in (("sku-fish", "은갈치", ["수산", "생선"]), ("sku-bread", "식빵", ["식품", "빵"])):
            orders_service.register_catalog_item(self.conn, seller_id=SELLER, item_id_local=item, title_text=title,
                                                 category_path=path, display_price_minor=10000)

    def _post(self, caption, items=(), uploads=(), kind="article"):
        return sns_service.create_post(self.conn, self.folder, seller_id=SELLER, kind=kind, caption=caption,
                                       item_ids=items, uploads=uploads)

    def test_post_without_upload_gets_a_generated_graphic(self):
        post = self._post("오늘 들어온 갈치 #제철 #수산", ["sku-fish"])
        self.assertEqual(post["hashtags"], ["제철", "수산"])
        self.assertEqual(len(post["media"]), 1)
        self.assertTrue(post["media"][0]["name"].startswith("gen-"))
        reel = self._post("손질 영상", ["sku-fish"], kind="short_video")
        self.assertEqual(reel["media"][0]["type"], "motion")

    def test_uploaded_photo_is_kept(self):
        post = self._post("사진", uploads=[PNG])
        self.assertEqual(post["media"][0]["type"], "image")
        self.assertTrue(post["media"][0]["name"].startswith("up-"))

    def test_invalid_posts_are_refused(self):
        with self.assertRaises(ContractError):
            self._post("   ")
        with self.assertRaises(ContractError):
            self._post("태그", ["no-such-item"])
        with self.assertRaises(ContractError):
            self._post("종류", kind="story")
        with self.assertRaises(media.MediaError):
            self._post("파일", uploads=[b"not media"])

    def test_like_toggles_and_comment_ownership(self):
        post = self._post("좋아요 테스트")
        pid = post["post_id"]
        self.assertTrue(sns_service.toggle_like(self.conn, seller_id=SELLER, post_id=pid, customer_id_local="c1"))
        self.assertFalse(sns_service.toggle_like(self.conn, seller_id=SELLER, post_id=pid, customer_id_local="c1"))
        sns_service.add_comment(self.conn, seller_id=SELLER, post_id=pid, body="맛있겠다", customer_id_local="c1")
        comment = sns_db.list_comments(self.conn, SELLER, pid)[0]
        with self.assertRaises(ContractError):  # someone else's comment
            sns_service.delete_comment(self.conn, seller_id=SELLER, comment_id=comment["comment_id"], customer_id_local="c2")
        sns_service.delete_comment(self.conn, seller_id=SELLER, comment_id=comment["comment_id"])  # the seller may
        self.assertEqual(sns_db.list_comments(self.conn, SELLER, pid), [])

    def test_deleted_post_disappears_from_the_feed(self):
        pid = self._post("지울 글")["post_id"]
        sns_service.delete_post(self.conn, seller_id=SELLER, post_id=pid)
        self.assertEqual(sns_service.explore(self.conn, seller_id=SELLER, customer_id_local=None), [])
        with self.assertRaises(ContractError):
            sns_service.toggle_like(self.conn, seller_id=SELLER, post_id=pid, customer_id_local="c1")

    def test_explore_puts_a_buyers_own_products_first(self):
        self._post("빵 소식", ["sku-bread"])
        fish = self._post("갈치 소식", ["sku-fish"])
        order = orders_service.place_order(self.conn, seller_id=SELLER, customer_id_local="c1", idempotency_key="k1",
                                           items=[{"item_id_local": "sku-fish", "quantity": 1, "unit_price_minor": 10000}],
                                           currency="KRW")
        orders_service.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="accept", expected_status_version=1)
        orders_service.transition_order(self.conn, seller_id=SELLER, order_id=order["order_id"], action="complete", expected_status_version=2)
        feed = sns_service.explore(self.conn, seller_id=SELLER, customer_id_local="c1")
        self.assertEqual(feed[0]["post_id"], fish["post_id"])
        self.assertEqual(feed[0]["reason"], "내가 산 상품")

    def test_insights_count_views_and_orders_after_the_post(self):
        pid = self._post("갈치", ["sku-fish"])["post_id"]
        sns_service.record_view(self.conn, seller_id=SELLER, post_id=pid, viewer="c1")
        sns_service.record_view(self.conn, seller_id=SELLER, post_id=pid, viewer="c1")
        sns_service.toggle_like(self.conn, seller_id=SELLER, post_id=pid, customer_id_local="c1")
        orders_service.place_order(self.conn, seller_id=SELLER, customer_id_local="c1", idempotency_key="k2",
                                   items=[{"item_id_local": "sku-fish", "quantity": 1, "unit_price_minor": 10000}], currency="KRW")
        row = next(r for r in sns_service.insights(self.conn, seller_id=SELLER) if r["post_id"] == pid)
        self.assertEqual((row["views"], row["likes"], row["orders_after"]), (2, 1, 1))
        self.assertEqual(row["engagement_rate"], 50.0)


class MigrationTest(unittest.TestCase):
    def test_old_tables_gain_new_columns_and_keep_rows(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "orders.sqlite"
        old = sqlite3.connect(path)
        old.executescript("""
            CREATE TABLE posts (seller_id TEXT NOT NULL, post_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
                                body TEXT, media_path TEXT, created_at TEXT NOT NULL, PRIMARY KEY (seller_id, post_id));
            CREATE TABLE group_buys (seller_id TEXT NOT NULL, group_buy_id TEXT NOT NULL, item_id_local TEXT NOT NULL,
                                     target_quantity INTEGER NOT NULL, unit_price_minor INTEGER NOT NULL,
                                     deadline_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL,
                                     PRIMARY KEY (seller_id, group_buy_id));
            INSERT INTO posts VALUES ('s', 'p1', 'article', '예전 글', '본문', 'old.png', '2026-01-01T00:00:00Z');
        """)
        old.commit()
        old.close()
        conn = orders_db.connect(path, schemas=(social_db.ensure_schema, sns_db.ensure_schema))
        self.addCleanup(conn.close)
        post = sns_db.fetch_post(conn, "s", "p1")
        self.assertEqual(post["media"], [{"type": "image", "name": "old.png"}])
        cols = {r[1] for r in conn.execute("PRAGMA table_info(group_buys)")}
        self.assertTrue({"proposer_customer_id", "message", "closed_reason"} <= cols)
        add_columns(conn, "group_buys", {"message": "TEXT"})  # again: a no-op


if __name__ == "__main__":
    unittest.main()
