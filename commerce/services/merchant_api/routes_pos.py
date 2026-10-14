"""POS screens (/pos/{seller_id}/...) for the store's staff, and the JSON calls the order screen makes.

The POS is a seller staff session like the owner web (same login), so each store's POS only
ever sees its own tables, menu and members. JSON calls carry the session's CSRF token in the
X-CSRF-Token header.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from commerce.packages.contracts.errors import ContractError
from commerce.services.merchant_api import orders_service, pos_service as pos, session
from commerce.services.merchant_api.routes_sns import ScreenDeps

ERRORS = {
    "/stock": "품절된 메뉴예요.", "/phone": "휴대폰 번호 11자리를 확인해 주세요.", "/amount": "받을 금액을 확인해 주세요.",
    "/received": "받은 현금이 결제 금액보다 적어요.", "/points": "쓸 수 있는 포인트보다 많아요.", "/status": "이미 결제가 끝난 주문이에요.",
    "/table_no": "그 테이블은 지금 다른 주문이 있어요.", "/lines": "담긴 메뉴가 없어요.", "/discount": "할인 금액을 확인해 주세요.",
    "/member": "포인트를 쓴 주문은 회원을 바꿀 수 없어요.", "/open_tickets": "아직 결제하지 않은 테이블이 있어요.",
    "/table": "지금 배치도에 없는 테이블이에요. 사장님 화면의 테이블 배치를 확인해 주세요.",
}


def _today() -> str:
    return dt.datetime.now(pos.KST).date().isoformat()


def recommendations(conn, seller_id: str, member_id: Optional[str], runtime, exclude: set[str] = frozenset()) -> dict[str, Any]:
    """Up to four menus to suggest at the counter: B's ranking for this member, or the store's
    popular ones for a walk-in. Sold-out menus are skipped (A knows stock, B does not)."""
    rec = orders_service.get_recommendations_for_display(conn, seller_id=seller_id, customer_id_local=member_id or "walk-in",
                                                        top_n=10, runtime=runtime)
    menu = {m["item_id"]: m for m in pos.menu(conn, seller_id)}
    picks = [menu[e["item_id_local"]] for e in rec["items"]
             if e["item_id_local"] in menu and not menu[e["item_id_local"]]["sold_out"] and e["item_id_local"] not in exclude][:4]
    label = orders_service.recommendation_label(rec)
    usual = pos.usual_menu(conn, seller_id, member_id) if member_id else []
    return {"items": picks, "label": label, "model": rec.get("model_version"), "personal": bool(member_id) and label is None,
            "usual": usual}


def register(app: FastAPI, d: ScreenDeps, templates: Jinja2Templates) -> None:
    def require_api(seller_id: str, request: Request) -> dict:
        staff = d.require_seller(seller_id, request)
        if not session.check_csrf(request.cookies.get(session.seller_cookie(seller_id)), request.headers.get("x-csrf-token")):
            raise ContractError("FORBIDDEN", "/csrf_token")
        return staff

    def state(conn, request: Request, seller_id: str, ticket_id: str, extra: Optional[dict] = None) -> JSONResponse:
        v = pos.view(conn, seller_id, ticket_id)
        rec = recommendations(conn, seller_id, v["member_id"], request.app.state.merchant.runtime,
                              {l["item_id_local"] for l in v["lines"] if not l["voided_at"]})
        return JSONResponse({"ticket": v, "recommend": rec, **(extra or {})})

    def api(fn):
        """Run fn and turn a ContractError into a JSON message the order screen shows."""
        try:
            return fn()
        except ContractError as exc:
            return JSONResponse({"error": ERRORS.get(exc.field_path, "처리하지 못했어요 (%s)" % exc.code), "code": exc.code},
                                status_code=409 if exc.code in ("ILLEGAL_STATE_TRANSITION", "DUPLICATE_EVENT") else 422)

    # --- screens ------------------------------------------------------------------

    @app.get("/pos/{seller_id}/")
    def pos_floor(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        return templates.TemplateResponse(request, "floor.html", {
            "seller_id": seller_id, "staff": staff, "screen": "floor", "floor": pos.floor(conn, seller_id),
            "kitchen": len(pos.kitchen_queue(conn, seller_id)), "today": pos.day_summary(conn, seller_id, _today())})

    @app.get("/pos/{seller_id}/tables/{table_no}")
    def pos_table(seller_id: str, table_no: int, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        t = pos.open_ticket(conn, seller_id, table_no=table_no)
        return RedirectResponse(f"/pos/{seller_id}/tickets/{t['ticket_id']}", status_code=303)

    @app.get("/pos/{seller_id}/takeout")
    def pos_takeout(seller_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller), kind: str = "takeout"):
        t = pos.open_ticket(conn, seller_id, table_no=None, kind="dine_in" if kind == "dine_in" else "takeout")
        return RedirectResponse(f"/pos/{seller_id}/tickets/{t['ticket_id']}", status_code=303)

    @app.get("/pos/{seller_id}/tickets/{ticket_id}")
    def pos_ticket(seller_id: str, ticket_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        v = pos.view(conn, seller_id, ticket_id)
        if v["status"] != "open":
            return RedirectResponse(f"/pos/{seller_id}/receipts/{ticket_id}", status_code=303)
        items = pos.menu(conn, seller_id)
        rec = recommendations(conn, seller_id, v["member_id"], request.app.state.merchant.runtime,
                              {l["item_id_local"] for l in v["lines"] if not l["voided_at"]})
        return templates.TemplateResponse(request, "ticket.html", {
            "seller_id": seller_id, "staff": staff, "screen": "ticket", "ticket": v, "menu": items,
            "sections": pos.sections(items), "recommend": rec, "memos": pos.MEMOS,
            "table_choices": pos.table_choices(conn, seller_id), "settings": pos.settings(conn, seller_id),
            "methods": pos.METHODS})

    @app.get("/pos/{seller_id}/receipts/{ticket_id}")
    def pos_receipt(seller_id: str, ticket_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        return templates.TemplateResponse(request, "receipt.html", {
            "seller_id": seller_id, "staff": staff, "screen": "receipt", "ticket": pos.view(conn, seller_id, ticket_id),
            "store": staff.get("display_name") or seller_id})

    @app.get("/pos/{seller_id}/kitchen")
    def pos_kitchen(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        return templates.TemplateResponse(request, "kitchen.html", {
            "seller_id": seller_id, "staff": staff, "screen": "kitchen", "queue": pos.kitchen_queue(conn, seller_id),
            "now": dt.datetime.now(dt.timezone.utc)})

    @app.get("/pos/{seller_id}/close")
    def pos_close_screen(seller_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller),
                         day: Optional[str] = None):
        day = day or _today()
        return templates.TemplateResponse(request, "close.html", {
            "seller_id": seller_id, "staff": staff, "screen": "close", "summary": pos.day_summary(conn, seller_id, day),
            "open_tables": conn.execute("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = ? AND status = 'open'",
                                        (seller_id,)).fetchone()[0],
            "error": request.query_params.get("e")})

    @app.post("/pos/{seller_id}/close")
    def pos_close(seller_id: str, conn=Depends(d.get_conn), staff=Depends(d.require_seller_form)):
        try:
            pos.close_day(conn, seller_id, _today())
        except ContractError:
            return RedirectResponse(f"/pos/{seller_id}/close?e=open", status_code=303)
        return RedirectResponse(f"/pos/{seller_id}/close", status_code=303)

    # --- JSON calls from the order screen ----------------------------------------------

    @app.get("/pos/{seller_id}/api/tickets/{ticket_id}")
    def api_ticket(seller_id: str, ticket_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(d.require_seller)):
        return state(conn, request, seller_id, ticket_id)

    def mutate(seller_id: str, ticket_id: str, request: Request, conn, action, extra=None):
        def run():
            out = action()
            return state(conn, request, seller_id, ticket_id, extra(out) if extra else None)
        return api(run)

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/lines")
    def api_add(seller_id: str, ticket_id: str, request: Request, payload: dict = Body(...), conn=Depends(d.get_conn),
                staff=Depends(require_api)):
        return mutate(seller_id, ticket_id, request, conn, lambda: pos.add_line(
            conn, seller_id, ticket_id, str(payload.get("item_id")), int(payload.get("quantity") or 1), payload.get("memo")))

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/lines/{line_id}")
    def api_line(seller_id: str, ticket_id: str, line_id: str, request: Request, payload: dict = Body(...),
                 conn=Depends(d.get_conn), staff=Depends(require_api)):
        def act():
            if "memo" in payload:
                pos.set_line_memo(conn, seller_id, ticket_id, line_id, payload.get("memo"))
            if "quantity" in payload:
                pos.set_line_quantity(conn, seller_id, ticket_id, line_id, int(payload["quantity"]))
        return mutate(seller_id, ticket_id, request, conn, act)

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/send")
    def api_send(seller_id: str, ticket_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(require_api)):
        return mutate(seller_id, ticket_id, request, conn, lambda: pos.send_to_kitchen(conn, seller_id, ticket_id),
                      lambda n: {"sent": n})

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/member")
    def api_member(seller_id: str, ticket_id: str, request: Request, payload: dict = Body(...), conn=Depends(d.get_conn),
                   staff=Depends(require_api)):
        if payload.get("phone"):
            return mutate(seller_id, ticket_id, request, conn, lambda: pos.attach_member(conn, seller_id, ticket_id, payload["phone"]),
                          lambda m: {"member_new": m["visits"] == 0})
        return mutate(seller_id, ticket_id, request, conn, lambda: pos.detach_member(conn, seller_id, ticket_id))

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/details")
    def api_details(seller_id: str, ticket_id: str, request: Request, payload: dict = Body(...), conn=Depends(d.get_conn),
                    staff=Depends(require_api)):
        def act():
            if "guests" in payload:
                pos.set_guests(conn, seller_id, ticket_id, int(payload["guests"]))
            if "discount" in payload or "discount_percent" in payload:
                pos.set_discount(conn, seller_id, ticket_id, amount=int(payload.get("discount") or 0),
                                 percent=int(payload.get("discount_percent") or 0))
            if payload.get("table_no"):
                pos.move_table(conn, seller_id, ticket_id, int(payload["table_no"]))
        return mutate(seller_id, ticket_id, request, conn, act)

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/void")
    def api_void(seller_id: str, ticket_id: str, request: Request, payload: dict = Body(default={}), conn=Depends(d.get_conn),
                 staff=Depends(require_api)):
        return mutate(seller_id, ticket_id, request, conn, lambda: pos.void_ticket(conn, seller_id, ticket_id, payload.get("reason", "")),
                      lambda _: {"redirect": f"/pos/{seller_id}/"})

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/pay")
    def api_pay(seller_id: str, ticket_id: str, request: Request, payload: dict = Body(...), conn=Depends(d.get_conn),
                staff=Depends(require_api)):
        runtime = request.app.state.merchant.runtime

        def act():
            amount = payload.get("amount")
            received = payload.get("received")
            return pos.pay(conn, seller_id, ticket_id, method=str(payload.get("method")),
                           amount=int(amount) if amount not in (None, "") else None,
                           received=int(received) if received not in (None, "") else None, runtime=runtime)

        def extra(v):
            out = {"change": v.get("change")}
            if v["status"] == "paid":
                out["redirect"] = f"/pos/{seller_id}/receipts/{ticket_id}"
            return out
        return mutate(seller_id, ticket_id, request, conn, act, extra)

    @app.post("/pos/{seller_id}/api/lines/{line_id}/served")
    def api_served(seller_id: str, line_id: str, conn=Depends(d.get_conn), staff=Depends(require_api)):
        return api(lambda: (pos.serve_line(conn, seller_id, line_id), JSONResponse({"ok": True}))[1])

    @app.post("/pos/{seller_id}/api/tickets/{ticket_id}/refund")
    def api_refund(seller_id: str, ticket_id: str, request: Request, conn=Depends(d.get_conn), staff=Depends(require_api)):
        return mutate(seller_id, ticket_id, request, conn, lambda: pos.refund(conn, seller_id, ticket_id))
