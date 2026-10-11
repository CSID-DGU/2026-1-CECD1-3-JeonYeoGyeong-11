"""Order details, notifications, the store's own page and follows, the buyer's saved
delivery details, and the seller's order/delivery/photo/stock tools (see fulfillment.py,
notifications.py)."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import (fulfillment, media, notifications, orders_service, shop_db, social_db,
                                            sns_service)
from commerce.services.merchant_api.routes_sns import ScreenDeps, _read_uploads

STATUS_LABELS = {"requested": "접수", "accepted": "준비 중", "completed": "완료", "cancelled": "취소"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _order_view(conn, seller_id: str, order: dict) -> dict:
    titles = {i["item_id_local"]: i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
    lines = [dict(i, title=titles.get(i["item_id_local"], {}).get("title_text", i["item_id_local"]),
                  category_path=titles.get(i["item_id_local"], {}).get("category_path"),
                  line_total=i["quantity"] * i["unit_price_minor"]) for i in order["items"]]
    subtotal = sum(l["line_total"] for l in lines)
    detail = fulfillment.of_order(conn, seller_id, order["order_id"])
    fee = detail["shipping_fee"] if detail else 0
    return {"order": order, "lines": lines, "subtotal": subtotal, "fee": fee, "total": subtotal + fee,
            "fulfillment": detail, "timeline": fulfillment.timeline(conn, seller_id, order),
            "method_label": fulfillment.METHODS.get(detail["method"]) if detail else None,
            "status_label": STATUS_LABELS[order["status"]]}


def register(app: FastAPI, d: ScreenDeps) -> None:
    def folder(request: Request):
        return media.media_dir(request.app.state.merchant.merchant_db_path)

    # --- buyer -------------------------------------------------------------------

    @app.get("/buyer/{seller_id}/orders/{order_id}")
    def buyer_order(seller_id: str, order_id: str, request: Request, conn=Depends(d.get_conn),
                    customer=Depends(d.require_customer)):
        order = orders_service.get_order(conn, seller_id=seller_id, order_id=order_id)
        if order["customer_id_local"] != customer["customer_id_local"]:
            raise ContractError("NOT_FOUND", "/order_id")
        return d.buyer_templates.TemplateResponse(request, "order_detail.html", {
            "seller_id": seller_id, "active_tab": "orders", "customer": customer,
            "terms": fulfillment.store_terms(conn, seller_id), **_order_view(conn, seller_id, order)})

    @app.get("/buyer/{seller_id}/notifications")
    def buyer_notifications(seller_id: str, request: Request, conn=Depends(d.get_conn),
                            customer=Depends(d.require_customer)):
        items = notifications.recent(conn, seller_id, customer["customer_id_local"])
        notifications.mark_all_read(conn, seller_id, customer["customer_id_local"])
        return d.buyer_templates.TemplateResponse(request, "notifications.html", {
            "seller_id": seller_id, "active_tab": "notifications", "customer": customer, "items": items,
            "unread_notifications": 0})

    @app.get("/buyer/{seller_id}/store")
    def buyer_store(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.current_customer)):
        catalog = [i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id) if i["listing_status"] == "active"]
        ratings = shop_db.rating_summary(conn, seller_id)
        total_reviews = sum(n for _, n in ratings.values())
        avg = round(sum(r * n for r, n in ratings.values()) / total_reviews, 1) if total_reviews else None
        sold = {}
        for o in orders_service.list_orders(conn, seller_id=seller_id):
            if o["status"] == "completed":
                for i in o["items"]:
                    sold[i["item_id_local"]] = sold.get(i["item_id_local"], 0) + i["quantity"]
        best = sorted(catalog, key=lambda i: -sold.get(i["item_id_local"], 0))[:8]
        posts = sns_service.explore(conn, seller_id=seller_id, customer_id_local=None, limit=9)
        settings = social_db.get_store_settings(conn, seller_id)
        first = conn.execute("SELECT MIN(created_at) FROM orders WHERE seller_id = ?", (seller_id,)).fetchone()[0]
        return d.buyer_templates.TemplateResponse(request, "store.html", {
            "seller_id": seller_id, "active_tab": "store", "customer": customer, "intro": settings.get("store_intro"),
            "product_count": len(catalog), "rating": avg, "review_count": total_reviews,
            "followers": len(notifications.followers(conn, seller_id)),
            "following": notifications.is_following(conn, seller_id, customer["customer_id_local"] if customer else None),
            "orders_done": sum(sold.values()), "since": (first or "")[:10], "best": best, "ratings": ratings,
            "posts": posts, "terms": fulfillment.store_terms(conn, seller_id), "photos": shop_db.photos(conn, seller_id)})

    @app.post("/buyer/{seller_id}/store/follow")
    def buyer_follow(seller_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                     back: str = Form("store")):
        notifications.toggle_follow(conn, seller_id, customer["customer_id_local"])
        return RedirectResponse(f"/buyer/{seller_id}/feed" if back == "feed" else f"/buyer/{seller_id}/store", status_code=303)

    @app.get("/buyer/{seller_id}/me")
    def buyer_me(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.require_customer),
                 saved: Optional[str] = None, e: Optional[str] = None):
        cid = customer["customer_id_local"]
        orders = orders_service.list_orders_by_customer(conn, seller_id=seller_id, customer_id_local=cid)
        return d.buyer_templates.TemplateResponse(request, "me.html", {
            "seller_id": seller_id, "active_tab": "me", "customer": customer,
            "profile": fulfillment.profile(conn, seller_id, cid), "saved": saved, "error": e,
            "counts": {s: sum(1 for o in orders if o["status"] == s) for s in STATUS_LABELS},
            "wishes": len(shop_db.wishlist(conn, seller_id, cid)),
            "following": notifications.is_following(conn, seller_id, cid)})

    @app.post("/buyer/{seller_id}/me")
    def buyer_me_save(seller_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                      recipient: str = Form(""), phone: str = Form(""), address: str = Form(""), method: str = Form("pickup")):
        try:
            details = fulfillment.validate(method, recipient, phone, address, "", "")
        except ContractError as exc:
            return RedirectResponse(f"/buyer/{seller_id}/me?e={exc.field_path.strip('/')}", status_code=303)
        with conn:
            conn.execute(
                """INSERT INTO customer_profiles (seller_id, customer_id_local, method, recipient, phone, address, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(seller_id, customer_id_local) DO UPDATE SET method = excluded.method,
                   recipient = excluded.recipient, phone = excluded.phone, address = excluded.address, updated_at = excluded.updated_at""",
                (seller_id, customer["customer_id_local"], details["method"], details["recipient"], details["phone"],
                 (address or "").strip()[:200] or None, _now()))
        return RedirectResponse(f"/buyer/{seller_id}/me?saved=1", status_code=303)

    # --- seller ------------------------------------------------------------------

    @app.get("/seller/{seller_id}/orders/{order_id}")
    def seller_order(seller_id: str, order_id: str, request: Request, conn=Depends(d.get_conn),
                     staff=Depends(d.require_seller)):
        order = orders_service.get_order(conn, seller_id=seller_id, order_id=order_id)
        customer = conn.execute("SELECT display_name FROM customers WHERE seller_id = ? AND customer_id_local = ?",
                                (seller_id, order["customer_id_local"])).fetchone()
        history = [o for o in orders_service.list_orders_by_customer(conn, seller_id=seller_id,
                                                                      customer_id_local=order["customer_id_local"])
                   if o["status"] == "completed"]
        return d.seller_templates.TemplateResponse(request, "order_detail.html", {
            "seller_id": seller_id, "active_tab": "orders", "staff": staff,
            "customer_name": customer["display_name"] if customer else order["customer_id_local"],
            "past_orders": len(history), "delivery_status": orders_service.purchase_event_status_by_order(
                conn, seller_id=seller_id).get(order_id), **_order_view(conn, seller_id, order)})

    @app.post("/seller/{seller_id}/orders/{order_id}/tracking")
    def seller_tracking(seller_id: str, order_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                        tracking_no: str = Form("")):
        fulfillment.set_tracking(conn, seller_id, order_id, tracking_no)
        order = orders_service.get_order(conn, seller_id=seller_id, order_id=order_id)
        if tracking_no.strip():
            notifications.notify(conn, seller_id, [order["customer_id_local"]], "order",
                                 "택배가 출발했어요. 송장번호 %s" % tracking_no.strip()[:30],
                                 f"/buyer/{seller_id}/orders/{order_id}", dedupe_key="tracking:%s:%s" % (order_id, tracking_no.strip()))
        return RedirectResponse(f"/seller/{seller_id}/orders/{order_id}", status_code=303)

    @app.post("/seller/{seller_id}/delivery-settings")
    def seller_delivery_settings(seller_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                                 pickup_address: str = Form(""), pickup_hours: str = Form(""),
                                 delivery_fee: int = Form(fulfillment.DEFAULT_DELIVERY_FEE),
                                 free_delivery_over: int = Form(fulfillment.DEFAULT_FREE_OVER)):
        fulfillment.save_store_terms(conn, seller_id, pickup_address=pickup_address, pickup_hours=pickup_hours,
                                     delivery_fee=delivery_fee, free_delivery_over=free_delivery_over)
        return RedirectResponse(f"/seller/{seller_id}/group-buys?saved=delivery", status_code=303)

    @app.post("/seller/{seller_id}/products/{item_id_local}/photos")
    def seller_add_photos(seller_id: str, item_id_local: str, request: Request, conn=Depends(d.get_conn),
                          staff=Depends(d.require_seller_form), files: list[UploadFile] = File(default=[])):
        if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local) is None:
            raise ContractError("NOT_FOUND", "/item_id_local")
        have = len(shop_db.photos(conn, seller_id).get(item_id_local, []))
        try:
            for data in _read_uploads(files)[:max(0, shop_db.MAX_PHOTOS - have)]:
                if not data:
                    continue
                if (media.sniff(data) or ("",))[0] != "image":
                    raise media.MediaError("상품 사진은 JPG·PNG·GIF·WebP만 올릴 수 있습니다.")
                stored = media.save_upload(folder(request), data)
                with conn:
                    shop_db.add_photo(conn, seller_id, item_id_local, stored["name"], _now())
        except media.MediaError:
            return RedirectResponse(f"/seller/{seller_id}/products/{item_id_local}?photo=bad", status_code=303)
        return RedirectResponse(f"/seller/{seller_id}/products/{item_id_local}?photo=ok", status_code=303)

    @app.post("/seller/{seller_id}/products/{item_id_local}/photos/{name}/delete")
    def seller_remove_photo(seller_id: str, item_id_local: str, name: str, conn=Depends(d.get_conn),
                            staff=Depends(d.require_seller_form)):
        with conn:
            shop_db.remove_photo(conn, seller_id, item_id_local, name)
        return RedirectResponse(f"/seller/{seller_id}/products/{item_id_local}", status_code=303)

    @app.post("/seller/{seller_id}/products/{item_id_local}/stock")
    def seller_stock(seller_id: str, item_id_local: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                     stock: str = Form("")):
        if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local) is None:
            raise ContractError("NOT_FOUND", "/item_id_local")
        value = stock.strip()
        fulfillment.set_stock(conn, seller_id, item_id_local, int(value) if value.isdigit() else None)
        return RedirectResponse(f"/seller/{seller_id}/products/{item_id_local}?saved=stock", status_code=303)
