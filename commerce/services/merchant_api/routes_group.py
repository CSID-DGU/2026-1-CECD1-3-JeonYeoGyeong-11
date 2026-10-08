"""Group-buy screens: buyers propose and join, the seller sets pricing and can close a proposal."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import RedirectResponse

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_service, social_service
from commerce.services.merchant_api.routes_sns import ScreenDeps

# ContractError code -> what the buyer reads after a failed proposal or join.
MESSAGES = {
    "DUPLICATE_EVENT": "이미 참여했거나, 같은 상품의 공동구매가 진행 중이에요. 진행 중인 공동구매에 참여해 주세요.",
    "ILLEGAL_STATE_TRANSITION": "이미 마감됐거나 성사된 공동구매예요.",
    "NOT_FOUND": "지금 판매하지 않는 상품이에요.",
    "INVALID_TYPE": "수량·목표·기간을 확인해 주세요.",
}


def _decorated(conn, seller_id: str, customer_id: Optional[str]) -> list[dict]:
    titles = {i["item_id_local"]: i for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
    out = []
    for gb in social_service.list_group_buys(conn, seller_id=seller_id):
        item = titles.get(gb["item_id_local"], {})
        joined = social_service.list_group_buy_participants(conn, seller_id=seller_id, group_buy_id=gb["group_buy_id"])
        out.append(dict(gb, title=item.get("title_text", gb["item_id_local"]), item=item,
                        regular_price=item.get("display_price_minor"),
                        percent=min(100, int(100 * gb["joined_quantity"] / gb["target_quantity"])) if gb["target_quantity"] else 0,
                        participants=len(joined),
                        mine=next((p["quantity"] for p in joined if p["customer_id_local"] == customer_id), None)))
    order = {"open": 0, "succeeded": 1, "failed": 2}
    out.sort(key=lambda g: (order.get(g["status"], 3), g["deadline_at"]))
    return out


def register(app: FastAPI, d: ScreenDeps) -> None:
    @app.get("/buyer/{seller_id}/group-buys")
    def buyer_group_buys(seller_id: str, request: Request, conn=Depends(d.get_conn), customer=Depends(d.current_customer),
                         e: Optional[str] = None, ok: Optional[str] = None, item: Optional[str] = None):
        settings = social_service.get_store_settings(conn, seller_id=seller_id)
        catalog = [dict(i, group_price=int(round(i["display_price_minor"] * (100 - settings["group_discount_pct"]) / 100, -1)))
                   for i in orders_service.list_catalog_for_display(conn, seller_id=seller_id) if i["listing_status"] == "active"]
        customer_id = customer["customer_id_local"] if customer else None
        return d.buyer_templates.TemplateResponse(request, "group_buys.html", {
            "seller_id": seller_id, "active_tab": "group_buys", "customer": customer, "settings": settings,
            "group_buys": _decorated(conn, seller_id, customer_id), "catalog": catalog,
            "error": MESSAGES.get(e or "", "요청을 처리하지 못했어요.") if e else None, "ok": ok, "preselect": item,
            "max_days": social_service.MAX_GROUP_BUY_DAYS,
        })

    @app.post("/buyer/{seller_id}/group-buys")
    def buyer_propose(seller_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                      item_id_local: str = Form(...), target_quantity: int = Form(...), quantity: int = Form(1),
                      days: int = Form(7), message: str = Form("")):
        try:
            social_service.propose_group_buy(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                             item_id_local=item_id_local, target_quantity=target_quantity,
                                             quantity=quantity, days=days, message=message)
        except ContractError as exc:
            return RedirectResponse(f"/buyer/{seller_id}/group-buys?e={exc.code}", status_code=303)
        return RedirectResponse(f"/buyer/{seller_id}/group-buys?ok=proposed", status_code=303)

    @app.post("/buyer/{seller_id}/group-buys/{group_buy_id}/join")
    def buyer_join(seller_id: str, group_buy_id: str, conn=Depends(d.get_conn), customer=Depends(d.require_customer_form),
                   quantity: int = Form(...)):
        try:
            gb = social_service.join_group_buy(conn, seller_id=seller_id, group_buy_id=group_buy_id,
                                               customer_id_local=customer["customer_id_local"], quantity=quantity)
        except ContractError as exc:
            return RedirectResponse(f"/buyer/{seller_id}/group-buys?e={exc.code}", status_code=303)
        done = "succeeded" if gb["status"] == "succeeded" else "joined"
        return RedirectResponse(f"/buyer/{seller_id}/group-buys?ok={done}", status_code=303)

    @app.get("/seller/{seller_id}/group-buys")
    def seller_group_buys(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller),
                          saved: Optional[str] = None):
        return d.seller_templates.TemplateResponse(request, "group_buys.html", {
            "seller_id": seller_id, "active_tab": "group_buys", "staff": staff,
            "group_buys": _decorated(conn, seller_id, None),
            "settings": social_service.get_store_settings(conn, seller_id=seller_id), "saved": saved,
        })

    @app.post("/seller/{seller_id}/group-buys/{group_buy_id}/close")
    def seller_close(seller_id: str, group_buy_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form)):
        social_service.close_group_buy(conn, seller_id=seller_id, group_buy_id=group_buy_id)
        return RedirectResponse(f"/seller/{seller_id}/group-buys", status_code=303)

    @app.post("/seller/{seller_id}/store-settings")
    def seller_settings(seller_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form),
                        group_discount_pct: int = Form(...), group_min_target: int = Form(...),
                        store_intro: str = Form("")):
        social_service.save_store_settings(conn, seller_id=seller_id, group_discount_pct=group_discount_pct,
                                           group_min_target=group_min_target, store_intro=store_intro)
        return RedirectResponse(f"/seller/{seller_id}/group-buys?saved=1", status_code=303)
