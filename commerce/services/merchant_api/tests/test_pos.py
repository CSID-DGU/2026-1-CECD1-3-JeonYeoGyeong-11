"""The restaurant POS: table checks, kitchen, members by phone (never stored), split and cash
payments, settling into one A order that reaches B with only menu items and time, refunds,
day closing; each store's own floor plan (zones, table names, seats) and counter style; the POS
JSON calls (CSRF) and the owner's pages; the restaurant dataset."""
from __future__ import annotations

import dataclasses
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import fulfillment, orders_db, orders_service, pos_db, pos_layout, pos_service as pos, session
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.tests.fakes import FakeRecommenderRuntime, ScriptedRuntime

S = "rest-1"
MENU = [("m-kimchi", "김치찌개", 9000, "찌개"), ("m-rice", "공기밥", 1000, "사이드"), ("m-cola", "콜라", 2000, "음료")]


def _db(root: Path):
    conn = orders_db.connect(root / "orders.sqlite", schemas=(fulfillment.ensure_schema, pos_db.ensure_schema))
    for item, title, price, section in MENU:
        orders_service.register_catalog_item(conn, seller_id=S, item_id_local=item, title_text=title,
                                             category_path=["식당", section], display_price_minor=price)
    return conn


class PosServiceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.conn = _db(Path(tmp.name))
        self.addCleanup(self.conn.close)
        self.runtime = FakeRecommenderRuntime(S)

    def _ticket(self, table=1):
        return pos.open_ticket(self.conn, S, table_no=table, guests=2)["ticket_id"]

    def test_a_table_has_one_open_check_and_lines_merge_until_sent(self):
        t = self._ticket(3)
        self.assertEqual(pos.open_ticket(self.conn, S, table_no=3)["ticket_id"], t)
        pos.add_line(self.conn, S, t, "m-kimchi")
        pos.add_line(self.conn, S, t, "m-kimchi")
        self.assertEqual(pos.send_to_kitchen(self.conn, S, t), 1)
        pos.add_line(self.conn, S, t, "m-kimchi")  # after sending, a new kitchen order
        lines = pos.view(self.conn, S, t)["lines"]
        self.assertEqual([(l["quantity"], l["state"]) for l in lines], [(2, "sent"), (1, "new")])
        with self.assertRaises(ContractError):
            pos.open_ticket(self.conn, S, table_no=99)

    def _plan(self, **rename):
        """The default hall with some tables renamed / dropped (value None) and a second zone."""
        p = pos_layout.plan(self.conn, S)
        items = []
        for i in p["items"]:
            if i["kind"] == "table" and i["name"] in rename:
                if rename[i["name"]] is None:
                    continue
                i = dict(i, name=rename[i["name"]])
            items.append(i)
        items.append({"no": None, "kind": "table", "zone": "룸", "name": "룸1", "seats": 10, "shape": "round", "x": 2, "y": 2, "w": 5, "h": 5})
        return {"zones": ["홀", "룸"], "items": items}

    def test_a_store_without_a_plan_gets_its_old_table_count_as_a_hall(self):
        p = pos_layout.plan(self.conn, S)
        tables = [i for i in p["items"] if i["kind"] == "table"]
        self.assertEqual(p["zones"], ["홀"])
        self.assertEqual([t["name"] for t in tables][:3], ["1", "2", "3"])
        self.assertEqual(len(tables), 12)
        self.assertEqual(pos.floor(self.conn, S)["table_count"], 12)

    def test_renaming_and_moving_tables_keeps_checks_and_old_receipts(self):
        paid = self._ticket(3)
        pos.add_line(self.conn, S, paid, "m-kimchi")
        pos.pay(self.conn, S, paid, method="card", runtime=self.runtime)
        live = self._ticket(5)
        pos.add_line(self.conn, S, live, "m-cola")
        pos_layout.save(self.conn, S, self._plan(**{"3": "A3", "5": "창가1", "12": None}))
        self.assertEqual(pos.view(self.conn, S, paid)["label"], "3번 테이블")  # the receipt keeps the name it was paid under
        self.assertEqual(pos.view(self.conn, S, live)["label"], "창가1")      # the open check follows the table
        floor = pos.floor(self.conn, S)
        self.assertEqual([z["name"] for z in floor["zones"]], ["홀", "룸"])
        self.assertEqual(floor["table_count"], 12)  # 12 - 1 dropped + 룸1
        room = pos_layout.tables(self.conn, S)[-1]
        self.assertEqual((room["name"], room["seats"], room["zone"]), ("룸1", 10, "룸"))
        t = pos.open_ticket(self.conn, S, table_no=room["no"], guests=8)
        self.assertEqual(pos.view(self.conn, S, t["ticket_id"])["label"], "룸1")
        with self.assertRaises(ContractError):
            pos.open_ticket(self.conn, S, table_no=12)  # dropped from the plan
        eleven = next(t["no"] for t in pos_layout.tables(self.conn, S) if t["name"] == "11")
        pos.move_table(self.conn, S, live, eleven)
        self.assertEqual(pos.view(self.conn, S, live)["label"], "11번 테이블")

    def test_a_plan_that_cannot_work_is_refused_with_a_reason(self):
        busy = self._ticket(2)
        with self.assertRaisesRegex(pos_layout.LayoutError, "아직 결제하지 않은"):
            pos_layout.save(self.conn, S, self._plan(**{"2": None}))
        pos.void_ticket(self.conn, S, busy)
        bad = self._plan(**{"4": "1"})
        with self.assertRaisesRegex(pos_layout.LayoutError, "두 번"):
            pos_layout.save(self.conn, S, bad)
        bad = self._plan()
        bad["items"][1] = dict(bad["items"][1], x=bad["items"][0]["x"], y=bad["items"][0]["y"])
        with self.assertRaisesRegex(pos_layout.LayoutError, "겹쳐"):
            pos_layout.save(self.conn, S, bad)
        bad = self._plan()
        bad["items"][0] = dict(bad["items"][0], x=pos_layout.COLS - 1)
        with self.assertRaisesRegex(pos_layout.LayoutError, "도면 밖"):
            pos_layout.save(self.conn, S, bad)
        self.assertEqual(len(pos_layout.tables(self.conn, S)), 12)  # nothing half-saved

    def test_counter_orders_get_the_days_call_numbers_and_prepay_sends_with_payment(self):
        pos.save_settings(self.conn, S, point_rate=5, open_hours="", service_mode="prepay", time_limit=0)
        a = pos.open_ticket(self.conn, S, table_no=None)["ticket_id"]
        b = pos.open_ticket(self.conn, S, table_no=None, kind="dine_in")["ticket_id"]
        self.assertEqual((pos.view(self.conn, S, a)["label"], pos.view(self.conn, S, b)["label"]), ("포장 1", "매장 2"))
        pos.add_line(self.conn, S, b, "m-kimchi")
        pos.pay(self.conn, S, b, method="card", runtime=self.runtime)
        self.assertEqual([t["label"] for t in pos.kitchen_queue(self.conn, S)], ["매장 2"])  # paid, still cooking
        self.assertEqual(pos.settings(self.conn, S)["service_mode"], "prepay")
        with self.assertRaises(ContractError):
            pos.save_settings(self.conn, S, point_rate=5, open_hours="", service_mode="buffet")

    def test_a_sent_line_is_voided_not_deleted_and_kitchen_sees_only_live_lines(self):
        t = self._ticket()
        pos.add_line(self.conn, S, t, "m-kimchi", 2)
        pos.send_to_kitchen(self.conn, S, t)
        line = pos.view(self.conn, S, t)["lines"][0]["line_id"]
        self.assertEqual(len(pos.kitchen_queue(self.conn, S)), 1)
        pos.set_line_quantity(self.conn, S, t, line, 0)
        v = pos.view(self.conn, S, t)
        self.assertEqual((v["lines"][0]["state"], v["total"]), ("void", 0))
        self.assertEqual(pos.kitchen_queue(self.conn, S), [])

    def test_sold_out_menu_cannot_be_added(self):
        fulfillment.set_stock(self.conn, S, "m-cola", 0)
        with self.assertRaises(ContractError):
            pos.add_line(self.conn, S, self._ticket(), "m-cola")

    def test_member_phone_is_only_a_salted_hash(self):
        t = self._ticket()
        m = pos.attach_member(self.conn, S, t, "010-1234-5678")
        self.assertEqual(m["last4"], "5678")
        dump = "\n".join(str(tuple(r)) for r in self.conn.execute("SELECT * FROM pos_members"))
        self.assertNotIn("01012345678", dump)
        self.assertEqual(pos.member_for_phone(self.conn, S, "01012345678")["member_id"], m["member_id"])
        with self.assertRaises(ContractError):
            pos.attach_member(self.conn, S, t, "02-123")

    def test_split_and_cash_settle_one_order_that_b_sees_without_payment_or_phone(self):
        t = self._ticket()
        pos.add_line(self.conn, S, t, "m-kimchi", 2)
        pos.add_line(self.conn, S, t, "m-rice", 2)
        m = pos.attach_member(self.conn, S, t, "01055556666")
        v = pos.pay(self.conn, S, t, method="card", amount=10000, runtime=self.runtime)
        self.assertEqual((v["status"], v["due"]), ("open", 10000))
        with self.assertRaises(ContractError):
            pos.pay(self.conn, S, t, method="cash", received=5000, runtime=self.runtime)  # not enough cash
        v = pos.pay(self.conn, S, t, method="cash", received=20000, runtime=self.runtime)
        self.assertEqual((v["status"], v["change"], v["points_earned"]), ("paid", 10000, 1000))  # 5% of 20,000
        self.assertTrue(v["receipt_no"].endswith("-0001"))
        events = list(self.runtime.ingested_events.values())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["customer_id_local"], m["member_id"])
        self.assertEqual(sorted((i["item_id_local"], i["quantity_observed"]) for i in events[0]["items"]),
                         [("m-kimchi", 2), ("m-rice", 2)])
        self.assertNotIn("5555", repr(events[0]))
        self.assertEqual(pos.member(self.conn, S, m["member_id"])["visits"], 1)

    def test_walk_ins_are_separate_customers_and_points_pay(self):
        a, b = self._ticket(1), self._ticket(2)
        for t in (a, b):
            pos.add_line(self.conn, S, t, "m-cola")
            pos.pay(self.conn, S, t, method="easy", runtime=self.runtime)
        ids = {e["customer_id_local"] for e in self.runtime.ingested_events.values()}
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(i.startswith("guest-") for i in ids))
        t = self._ticket(3)
        pos.add_line(self.conn, S, t, "m-kimchi")
        pos.attach_member(self.conn, S, t, "01077778888")
        with self.assertRaises(ContractError):
            pos.pay(self.conn, S, t, method="points", amount=1000)  # no points yet

    def test_refund_takes_points_back_and_void_reaches_nobody(self):
        t = self._ticket()
        pos.add_line(self.conn, S, t, "m-kimchi")
        m = pos.attach_member(self.conn, S, t, "01011112222")
        pos.pay(self.conn, S, t, method="card", runtime=self.runtime)
        pos.refund(self.conn, S, t)
        self.assertEqual(pos.member(self.conn, S, m["member_id"])["points"], 0)
        self.assertEqual(pos.view(self.conn, S, t)["status"], "refunded")
        t2 = self._ticket(2)
        pos.add_line(self.conn, S, t2, "m-cola")
        pos.void_ticket(self.conn, S, t2)
        self.assertEqual(len(self.runtime.ingested_events), 1)

    def test_day_summary_and_closing_need_no_open_checks(self):
        t = self._ticket()
        pos.add_line(self.conn, S, t, "m-kimchi")
        open_t = self._ticket(2)
        pos.add_line(self.conn, S, open_t, "m-cola")
        pos.pay(self.conn, S, t, method="card", runtime=self.runtime)
        day = pos.business_date(pos.view(self.conn, S, t)["paid_at"])
        s = pos.day_summary(self.conn, S, day)
        self.assertEqual((s["revenue"], s["count"], s["by_method"]), (9000, 1, [("카드", 9000)]))
        with self.assertRaises(ContractError):
            pos.close_day(self.conn, S, day)
        pos.void_ticket(self.conn, S, open_t)
        self.assertEqual(pos.close_day(self.conn, S, day)["revenue"], 9000)


class PosScreensTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        conn = _db(root)
        conn.close()
        self.runtime = ScriptedRuntime(S)
        env = mock.patch.dict("os.environ", {})
        env.start()
        self.addCleanup(env.stop)
        from commerce.services.merchant_api.main import create_app
        settings = MerchantSettings(seller_id=S, feature_db_path=root / "features.sqlite", model_dir=root / "models",
                                    merchant_db_path=root / "orders.sqlite")
        self.client = TestClient(create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=lambda *a: self.runtime)),
                                 follow_redirects=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        r = self.client.post(f"/seller/{S}/signup", data={"username": "owner", "display_name": "오이네 한식당", "password": "pw-1234",
                                                         "business_reg_no": "1234567890", "business_open_date": "20200101",
                                                         "business_rep_name": "홍길동"})
        self.assertEqual(r.status_code, 303, r.text)

    def _h(self):
        return {"X-CSRF-Token": session.csrf_token(self.client.cookies.get(session.seller_cookie(S)))}

    def test_pos_needs_login_and_comes_back_after_it(self):
        self.client.cookies.clear()
        r = self.client.get(f"/pos/{S}/")
        self.assertIn(f"/seller/{S}/login?next=%2Fpos%2F{S}%2F", r.headers["location"])
        r = self.client.post(f"/seller/{S}/login", data={"username": "owner", "password": "pw-1234", "next": f"/pos/{S}/"})
        self.assertEqual(r.headers["location"], f"/pos/{S}/")
        r = self.client.post(f"/seller/{S}/login", data={"username": "owner", "password": "pw-1234", "next": "https://evil.example/"})
        self.assertEqual(r.headers["location"], f"/seller/{S}/today")

    def test_full_table_flow_over_the_pos_api(self):
        floor = self.client.get(f"/pos/{S}/")
        self.assertEqual(floor.status_code, 200)
        self.assertIn("fp-table", floor.text)
        r = self.client.get(f"/pos/{S}/tables/4")
        ticket = r.headers["location"].rsplit("/", 1)[1]
        page = self.client.get(r.headers["location"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("김치찌개", page.text)
        api = f"/pos/{S}/api/tickets/{ticket}"
        self.assertEqual(self.client.post(api + "/lines", json={"item_id": "m-kimchi"}).status_code, 403)  # no CSRF
        j = self.client.post(api + "/lines", json={"item_id": "m-kimchi", "quantity": 2}, headers=self._h()).json()
        self.assertEqual(j["ticket"]["total"], 18000)
        j = self.client.post(api + "/member", json={"phone": "01012345678"}, headers=self._h()).json()
        self.assertEqual(j["ticket"]["member"]["label"], "****-5678")
        self.assertTrue(j["member_new"])
        j = self.client.post(api + "/send", headers=self._h()).json()
        self.assertEqual(j["sent"], 1)
        self.assertIn("김치찌개", self.client.get(f"/pos/{S}/kitchen").text)
        bad = self.client.post(api + "/pay", json={"method": "cash", "received": 1000, "amount": 18000}, headers=self._h())
        self.assertEqual(bad.status_code, 422)
        self.assertIn("받은 현금", bad.json()["error"])
        j = self.client.post(api + "/pay", json={"method": "card"}, headers=self._h()).json()
        self.assertEqual(j["redirect"], f"/pos/{S}/receipts/{ticket}")
        receipt = self.client.get(j["redirect"]).text
        self.assertIn("영수증", receipt)
        self.assertIn("****-5678", receipt)
        self.assertEqual(len(self.runtime.ingested_events), 1)
        self.assertEqual(self.client.get(f"/pos/{S}/tickets/{ticket}").headers["location"], f"/pos/{S}/receipts/{ticket}")

    def test_counter_recommendation_skips_sold_out_and_ordered_menus(self):
        self.runtime.recommendation = {
            "schema_version": "recommendation.v1", "seller_id": S, "customer_id_local": "x", "as_of": "2026-10-06T00:00:00.000000Z",
            "model_version": "ic100-R_lm-r500", "score_semantics": "next_purchase", "horizon_days": None, "is_cold_start": False,
            "fallback_reason": None, "items": [{"item_id_local": i, "score": 1.0 - n / 10} for n, (i, *_) in enumerate(MENU)]}
        r = self.client.get(f"/pos/{S}/tables/1")
        ticket = r.headers["location"].rsplit("/", 1)[1]
        api = f"/pos/{S}/api/tickets/{ticket}"
        self.client.post(f"/seller/{S}/menu/m-rice/soldout", data={"sold_out": "1", "csrf_token": self._h()["X-CSRF-Token"]})
        self.client.post(api + "/member", json={"phone": "01099998888"}, headers=self._h())
        j = self.client.post(api + "/lines", json={"item_id": "m-kimchi"}, headers=self._h()).json()
        self.assertEqual([m["item_id"] for m in j["recommend"]["items"]], ["m-cola"])
        self.assertTrue(j["recommend"]["personal"])

    def test_owner_pages_open_and_menu_add_points_to_new_menu_targets(self):
        for page in ("today", "sales", "menu", "guests", "fl", "settings"):
            r = self.client.get(f"/seller/{S}/{page}")
            self.assertEqual(r.status_code, 200, (page, r.text[:300]))
        self.assertNotIn("/buyer/", self.client.get(f"/seller/{S}/today").text)  # no link out to a buyer web or other store
        r = self.client.post(f"/seller/{S}/menu", data={"title_text": "들기름 막국수", "display_price_minor": "9000", "section": "식사",
                                                       "description_text": "메밀면", "csrf_token": self._h()["X-CSRF-Token"]})
        self.assertIn("/guests?item=menu-", r.headers["location"])
        self.assertIn("들기름 막국수", self.client.get(r.headers["location"]).text)
        r = self.client.post(f"/seller/{S}/settings", data={"point_rate": "3", "open_hours": "11~21", "service_mode": "prepay",
                                                           "time_limit": "90", "store_intro": "", "csrf_token": self._h()["X-CSRF-Token"]})
        self.assertEqual(r.status_code, 303)
        floor = self.client.get(f"/pos/{S}/").text
        self.assertIn("＋ 매장 주문", floor)   # prepay counter
        self.assertNotIn("＋ 포장 주문", floor)  # takeout box left unticked
        self.assertIn("90분 넘음", floor)

    def test_owner_draws_the_floor_and_the_pos_shows_it(self):
        self.assertEqual(self.client.get(f"/seller/{S}/floor").status_code, 200)
        plan = {"zones": ["1층", "2층"], "items": [
            {"no": None, "kind": "table", "zone": "1층", "name": "A1", "seats": 2, "shape": "round", "x": 0, "y": 0, "w": 2, "h": 2},
            {"no": None, "kind": "table", "zone": "2층", "name": "201", "seats": 6, "shape": "booth", "x": 4, "y": 4, "w": 4, "h": 2},
            {"no": None, "kind": "fixture", "zone": "1층", "name": "", "seats": 0, "shape": "kitchen", "x": 10, "y": 0, "w": 5, "h": 3}]}
        self.assertEqual(self.client.post(f"/seller/{S}/floor", json=plan).status_code, 403)  # no CSRF
        r = self.client.post(f"/seller/{S}/floor", json=plan, headers=self._h())
        self.assertEqual(r.status_code, 200, r.text)
        saved = r.json()["plan"]
        self.assertEqual(saved["zones"], ["1층", "2층"])
        self.assertEqual(sorted(i["name"] for i in saved["items"]), ["201", "A1", "주방"])
        floor = self.client.get(f"/pos/{S}/").text
        self.assertIn("2층", floor)
        self.assertIn(">A1<", floor)
        self.assertEqual(floor.count("fp-table"), 2)
        bad = dict(plan, items=plan["items"] + [dict(plan["items"][0], x=1)])
        r = self.client.post(f"/seller/{S}/floor", json=bad, headers=self._h())
        self.assertEqual(r.status_code, 422)
        self.assertIn("두 번", r.json()["error"])
        no = next(i["no"] for i in saved["items"] if i["name"] == "201")
        r = self.client.get(f"/pos/{S}/tables/{no}")
        self.assertIn("201", self.client.get(r.headers["location"]).text)


class LegacyGateTest(unittest.TestCase):
    def test_buyer_web_and_lab_are_off_when_asked(self):
        import importlib
        from commerce.services.merchant_api import main
        self.addCleanup(importlib.reload, main)  # runs after the env patch below is gone
        with mock.patch.dict("os.environ", {"LEGACY_BUYER_SCREENS": "0"}):
            main = importlib.reload(main)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        settings = MerchantSettings(seller_id=S, feature_db_path=root / "f.sqlite", model_dir=root / "m",
                                    merchant_db_path=root / "orders.sqlite")
        runtime = FakeRecommenderRuntime(S)
        with TestClient(main.create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=lambda *a: runtime)),
                        follow_redirects=False) as c:
            self.assertEqual(c.get(f"/buyer/{S}/").status_code, 404)
            self.assertEqual(c.get(f"/seller/{S}/compare").status_code, 404)
            self.assertEqual(c.get("/").headers["location"], f"/seller/{S}/today")


class RestaurantDatasetTest(unittest.TestCase):
    def test_small_store_is_deterministic_and_complete(self):
        from commerce.services.merchant_api import platform_dataset, restaurant_dataset as rd
        store = dataclasses.replace(rd.STORES[0], members=10, per_day=(4, 6), history_days=12)
        exports = []
        for _ in range(2):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            conn = orders_db.connect(Path(tmp.name) / "orders.sqlite")
            self.addCleanup(conn.close)
            rd.seed_store(conn, "merchant-1", store)
            exports.append(platform_dataset.export_seller_input(conn, "merchant-1"))
        self.assertEqual(exports[0], exports[1])
        s = rd.summary(conn, "merchant-1")
        self.assertGreater(s["tickets_paid"], 30)
        self.assertGreater(s["member_events"], 0)
        self.assertEqual(rd.seed_store(conn, "merchant-1", store)["tickets"], 0)  # a re-run adds nothing
        names = {r[0] for r in conn.execute("SELECT customer_id_local FROM orders")}
        self.assertTrue(any(n.startswith("m-") for n in names) and any(n.startswith("guest-") for n in names))

    def test_each_store_has_its_own_floor_and_counter_style(self):
        from commerce.services.merchant_api import restaurant_dataset as rd
        for store in rd.STORES:  # every drawn plan is one the editor would accept
            pos_layout._validate(rd.PLANS[store.key]())
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = orders_db.connect(Path(tmp.name) / "orders.sqlite")
        self.addCleanup(conn.close)
        hansik = dataclasses.replace(rd.STORES[0], members=8, per_day=(4, 6), history_days=6)
        bunsik = dataclasses.replace(rd.STORES[2], members=8, per_day=(4, 6), history_days=6)
        rd.seed_store(conn, "merchant-1", hansik)
        rd.seed_store(conn, "merchant-3", bunsik)
        self.assertEqual(pos_layout.plan(conn, "merchant-1")["zones"], ["홀", "방"])
        seats = {t["no"]: t["seats"] for t in pos_layout.tables(conn, "merchant-1")}
        for no, guests in conn.execute("SELECT table_no, guests FROM pos_tickets WHERE seller_id = 'merchant-1' AND table_no IS NOT NULL"):
            self.assertLessEqual(guests, seats[no])  # a party sits at a table that seats it
        self.assertEqual(pos.settings(conn, "merchant-3")["service_mode"], "prepay")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = 'merchant-3' AND table_no IS NOT NULL").fetchone()[0], 0)
        self.assertGreater(conn.execute("SELECT MAX(call_no) FROM pos_tickets WHERE seller_id = 'merchant-3'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
