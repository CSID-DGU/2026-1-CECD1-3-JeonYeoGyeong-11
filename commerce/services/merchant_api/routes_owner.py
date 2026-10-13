"""The owner's web (/seller/{seller_id}/...) for a restaurant: today, sales, menu, guests (what
the shared model says about this store's members), federated learning, settings.

Everything here reads this store's orders.sqlite and asks this store's B runtime; there is no
link to any other store and no central view.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import RedirectResponse

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import (fulfillment, model_status, orders_service, personalization, pos_service as pos,
                                            sales, social_db)
from commerce.services.merchant_api.orders_service import build_recommendation_request
from commerce.services.merchant_api.routes_sns import ScreenDeps

SHARED_GROUPS = {
    "text_only": ["text_proj", "fusion", "position", "sequence", "query_proj", "scorer"],
    "text_relation": ["text_proj", "fusion", "position", "sequence", "query_proj", "scorer", "time_mlp", "relation_mlp", "relation_pool"],
}


def _today() -> str:
    return dt.datetime.now(pos.KST).date().isoformat()


def _hours_chart(hours: dict[int, int]) -> dict[str, Any]:
    series = [{"date": "%02d" % h, "label": "%d시" % h, "revenue": hours.get(h, 0), "orders": 0} for h in range(10, 23)]
    return sales.bar_chart(series, max((s["revenue"] for s in series), default=0), width=640, height=170)


def release_info(model_dir: Optional[Path]) -> list[dict[str, Any]]:
    out = []
    installed = model_status.installed_models(model_dir)
    for variant, version in installed["variants"].items():
        info = {"variant": variant, "label": "텍스트+관계" if variant == "text_relation" else "텍스트", "version": version,
                "size_kb": None, "groups": SHARED_GROUPS[variant]}
        if version and model_dir:
            try:
                rel = json.loads((Path(model_dir) / "base" / variant / version / "release.json").read_text(encoding="utf-8"))
                info["size_kb"] = round(rel.get("weights_size_bytes", 0) / 1024)
            except (OSError, ValueError):
                pass
        out.append(info)
    return out


def menu_targets(conn, seller_id: str, item_id: str, runtime, members: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
    """Who to tell about a menu: members ranked by how often they ordered dishes of the same section
    (counted from this store's checks, not a model output), plus where the shared model ranks this
    dish for a few regulars (the AI's view of it next to the store's other dishes)."""
    menu = {m["item_id"]: m for m in pos.menu(conn, seller_id)}
    item = menu.get(item_id)
    if item is None:
        return {"ok": False, "reason": "판매 중인 메뉴가 아니에요"}
    same = [i for i, m in menu.items() if m["section"] == item["section"]]
    counts: dict[str, int] = {}
    marks = ",".join("?" * len(same))
    for r in conn.execute("SELECT t.member_id, SUM(l.quantity) AS n FROM pos_lines l JOIN pos_tickets t USING (seller_id, ticket_id) "
                          "WHERE l.seller_id = ? AND t.status = 'paid' AND t.member_id IS NOT NULL AND l.voided_at IS NULL "
                          "AND l.item_id_local IN (%s) GROUP BY t.member_id" % marks, (seller_id, *same)):
        counts[r["member_id"]] = r["n"]
    ranked = sorted((dict(m, same_count=counts.get(m["member_id"], 0)) for m in members if counts.get(m["member_id"])),
                    key=lambda m: (-m["same_count"], -m["visits"]))
    ai_rank = None
    if runtime is not None and members:
        active = list(menu)
        ranks = []
        for m in members[:10]:
            try:
                rec = runtime.predict_local(build_recommendation_request(seller_id, m["member_id"], candidate_item_ids=active, top_n=len(active)))
            except Exception:
                break
            if rec.get("fallback_reason") and rec["fallback_reason"] != "no_customer_history":
                break
            ids = [e["item_id_local"] for e in rec["items"]]
            if item_id in ids:
                ranks.append(ids.index(item_id) + 1)
        if ranks:
            ranks.sort()
            ai_rank = {"median": ranks[len(ranks) // 2], "best": ranks[0], "of": len(active)}
    return {"ok": True, "members": ranked[:limit], "section": item["section"], "ai": ai_rank}


def register(app: FastAPI, d: ScreenDeps) -> None:
    T = d.seller_templates

    @app.get("/seller/{seller_id}/today")
    def owner_today(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        today = _today()
        yesterday = (dt.date.fromisoformat(today) - dt.timedelta(days=1)).isoformat()
        s_today, s_yday = pos.day_summary(conn, seller_id, today), pos.day_summary(conn, seller_id, yesterday)
        orders = orders_service.list_orders(conn, seller_id=seller_id)
        titles = {i["item_id_local"]: i["title_text"] for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
        stats = sales.summary(orders, titles)
        floor = pos.floor(conn, seller_id)
        context = request.app.state.merchant
        probe = orders_service.get_recommendations_for_display(
            conn, seller_id=seller_id, customer_id_local=model_status.probe_customer(conn, seller_id), runtime=context.runtime)
        members = pos.members_overview(conn, seller_id)
        return T.TemplateResponse(request, "today.html", {
            "seller_id": seller_id, "active_tab": "today", "staff": staff, "today": s_today, "yesterday": s_yday,
            "stats": stats, "chart": sales.bar_chart(stats["series"], stats["peak"]), "hours": _hours_chart(s_today["hours"]),
            "floor": floor, "kitchen": len(pos.kitchen_queue(conn, seller_id)),
            "model_ok": probe["model_version"] != orders_service.MOCK_MODEL_VERSION and not probe.get("fallback_reason"),
            "model_version": probe["model_version"], "members": len(members),
            "regulars": sum(1 for m in members if m["visits"] >= 3),
            "low_stock": [(titles.get(i, i), n) for i, n in fulfillment.stock_levels(conn, seller_id).items() if n is not None and n <= 3]})

    @app.get("/seller/{seller_id}/sales")
    def owner_sales(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller), day: Optional[str] = None):
        try:
            day = dt.date.fromisoformat(day).isoformat() if day else _today()
        except ValueError:
            day = _today()
        prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
        nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
        return T.TemplateResponse(request, "sales.html", {
            "seller_id": seller_id, "active_tab": "sales", "staff": staff, "day": day, "prev": prev,
            "next": nxt if nxt <= _today() else None, "summary": pos.day_summary(conn, seller_id, day),
            "tickets": pos.paid_tickets(conn, seller_id, day), "hours": _hours_chart(pos.day_summary(conn, seller_id, day)["hours"])})

    @app.get("/seller/{seller_id}/menu")
    def owner_menu(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        items = pos.menu(conn, seller_id)
        inactive = [i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id) if i["listing_status"] != "active"]
        orders = orders_service.list_orders(conn, seller_id=seller_id)
        titles = {i["item_id_local"]: i["title_text"] for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
        sold = {t["item_id_local"]: t["quantity"] for t in sales.summary(orders, titles)["top_items"]}
        counts: dict[str, int] = {}
        for o in orders:
            if o["status"] == "completed":
                for it in o["items"]:
                    counts[it["item_id_local"]] = counts.get(it["item_id_local"], 0) + it["quantity"]
        return T.TemplateResponse(request, "menu.html", {
            "seller_id": seller_id, "active_tab": "menu", "staff": staff, "items": items, "sections": pos.sections(items),
            "inactive": inactive, "counts": counts, "sold30": sold, "saved": request.query_params.get("saved")})

    @app.post("/seller/{seller_id}/menu")
    def owner_menu_add(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                       title_text: str = Form(...), display_price_minor: int = Form(...), section: str = Form("식사"),
                       description_text: str = Form("")):
        if display_price_minor < 0 or not title_text.strip():
            raise ContractError("INVALID_TYPE", "/display_price_minor")
        item_id = "menu-" + uuid.uuid4().hex[:8]
        orders_service.register_catalog_item(
            conn, seller_id=seller_id, item_id_local=item_id, title_text=title_text.strip(),
            description_text=description_text.strip() or None, category_path=["식당", section.strip()[:20] or "식사"],
            display_price_minor=display_price_minor, runtime=request.app.state.merchant.runtime)
        return RedirectResponse(f"/seller/{seller_id}/guests?item={item_id}#new-menu", status_code=303)

    @app.post("/seller/{seller_id}/menu/{item_id_local}/soldout")
    def owner_soldout(seller_id: str, item_id_local: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                      sold_out: str = Form("1")):
        fulfillment.set_stock(conn, seller_id, item_id_local, 0 if sold_out == "1" else None)
        return RedirectResponse(f"/seller/{seller_id}/menu?saved=stock#m-{item_id_local}", status_code=303)

    @app.get("/seller/{seller_id}/guests")
    def owner_guests(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller),
                     item: Optional[str] = None):
        members = pos.members_overview(conn, seller_id)
        menu = pos.menu(conn, seller_id)
        catalog = {i["item_id_local"]: i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
        newest = sorted(menu, key=lambda m: catalog[m["item_id"]]["first_listed_at"], reverse=True)
        target = item or (newest[0]["item_id"] if newest else None)
        return T.TemplateResponse(request, "guests.html", {
            "seller_id": seller_id, "active_tab": "guests", "staff": staff, "members": members,
            "regulars": sum(1 for m in members if m["visits"] >= 3), "pairs": pos.menu_pairs(conn, seller_id),
            "menu": newest, "target": target, "target_title": next((m["title"] for m in menu if m["item_id"] == target), None),
            "targets": menu_targets(conn, seller_id, target, request.app.state.merchant.runtime, members) if target else None})

    @app.get("/seller/{seller_id}/guests/{member_id}")
    def owner_guest(seller_id: str, member_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        m = pos.member(conn, seller_id, member_id)
        if m is None:
            raise ContractError("NOT_FOUND", "/member_id")
        rec = orders_service.get_recommendations_for_display(conn, seller_id=seller_id, customer_id_local=member_id, top_n=5,
                                                            runtime=request.app.state.merchant.runtime)
        catalog = {i["item_id_local"]: i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
        return T.TemplateResponse(request, "guest.html", {
            "seller_id": seller_id, "active_tab": "guests", "staff": staff, "member": m,
            "history": pos.member_history(conn, seller_id, member_id), "favourites": pos.favourite_items(conn, seller_id, member_id),
            "rec": [catalog[e["item_id_local"]] for e in rec["items"] if e["item_id_local"] in catalog],
            "rec_label": orders_service.recommendation_label(rec), "rec_model": rec.get("model_version")})

    @app.get("/seller/{seller_id}/fl")
    def owner_fl(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        context = request.app.state.merchant
        kept = {
            "orders": conn.execute("SELECT COUNT(*) FROM orders WHERE seller_id = ?", (seller_id,)).fetchone()[0],
            "tickets": conn.execute("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = ? AND status IN ('paid', 'refunded')", (seller_id,)).fetchone()[0],
            "members": conn.execute("SELECT COUNT(*) FROM pos_members WHERE seller_id = ?", (seller_id,)).fetchone()[0],
            "payments": conn.execute("SELECT COUNT(*) FROM pos_payments WHERE seller_id = ?", (seller_id,)).fetchone()[0],
        }
        fl_cfg = getattr(getattr(context.fl_client, "config", None), "enabled", False)
        return T.TemplateResponse(request, "fl.html", {
            "seller_id": seller_id, "active_tab": "fl", "staff": staff, "releases": release_info(context.model_dir),
            "encoder": model_status.installed_models(context.model_dir)["encoder"], "kept": kept,
            "delivery": orders_service.delivery_summary(conn, seller_id=seller_id), "fl_enabled": bool(fl_cfg),
            "personalization": personalization.last_run(seller_id), "notice": request.query_params.get("p")})

    @app.get("/seller/{seller_id}/settings")
    def owner_settings(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        return T.TemplateResponse(request, "settings.html", {
            "seller_id": seller_id, "active_tab": "settings", "staff": staff, "pos": pos.settings(conn, seller_id),
            "store": social_db.get_store_settings(conn, seller_id), "saved": request.query_params.get("saved")})

    @app.post("/seller/{seller_id}/settings")
    def owner_settings_save(seller_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                            tables: int = Form(12), point_rate: int = Form(5), open_hours: str = Form(""), store_intro: str = Form("")):
        pos.save_settings(conn, seller_id, tables=tables, point_rate=point_rate, open_hours=open_hours)
        current = social_db.get_store_settings(conn, seller_id)
        with conn:
            social_db.save_store_settings(conn, seller_id, current["group_discount_pct"], current["group_min_target"], store_intro)
        return RedirectResponse(f"/seller/{seller_id}/settings?saved=1", status_code=303)
