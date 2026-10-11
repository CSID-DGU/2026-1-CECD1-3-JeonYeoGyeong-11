"""Merchant app: health, lifecycle, order/catalog domain routes, and screens.

The plain JSON routes under /sellers/{seller_id}/... have no caller
authentication -- who is allowed to act as a given seller_id there is exactly
OQ13 (auth error code) and OQ15 (order_id in seller-origin URLs), both open
human decisions (docs/design/open-questions.md), kept for local/integration
testing only.

The screen routes under /buyer/{seller_id}/... and /seller/{seller_id}/...
check caller identity via signed session cookies (accounts_service.py,
session.py): every /seller/... route except signup/login requires a verified
seller_accounts login, and every buyer action that needs an identity (placing
an order, joining a group-buy, messaging, viewing order history) requires a
customer login. Browsing (catalog, feed, group-buy listing) stays open. Every
state-changing form of a logged-in user carries a session-bound CSRF token
(require_*_form). Prices come from the server's catalog or group-buy row,
never from the form. OQ13's error-code question and OQ15 stay open.
"""
import asyncio
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel  # ships with fastapi; no separate lock entry needed

from commerce.packages.contracts.errors import ContractError
from commerce.packages.fl_client.lifecycle import FLClientConfig
from commerce.services.merchant_api import (accounts_db, accounts_service, cart_db, chatbot, fulfillment, model_status,
                                            notifications, orders_db, orders_service, personalization, routes_chat,
                                            routes_group, routes_shop, routes_sns, routes_store, sales, session, shop_db,
                                            shop_service, sns_db, sns_service, social_db, social_service)
from commerce.services.merchant_api.accounts_service import AccountError
from commerce.services.merchant_api.context import MerchantContext, MerchantSettings, build_context, merchant_db_path_from_env
from commerce.services.merchant_api.presentation import mask_name, product_emoji

_APPS_DIR = Path(__file__).resolve().parents[2] / "apps"
_log = logging.getLogger(__name__)


def _csrf_context(cookie_for_seller: Callable[[str], str]):
    def processor(request: Request) -> dict:
        seller_id = request.path_params.get("seller_id", "")
        return {"csrf_token": session.csrf_token(request.cookies.get(cookie_for_seller(seller_id)))}
    return processor


def _store_name(request: Request) -> dict:
    """The store's display name (its first staff account's), for screens that name the seller."""
    seller_id = request.path_params.get("seller_id", "")
    merchant = getattr(request.app.state, "merchant", None)
    name = None
    if merchant is not None and seller_id == merchant.seller_id:
        conn = orders_db.connect(merchant.merchant_db_path, schemas=(accounts_db.ensure_schema,))
        try:
            row = conn.execute("SELECT display_name FROM seller_accounts WHERE seller_id = ? ORDER BY rowid LIMIT 1",
                               (seller_id,)).fetchone()
            name = row["display_name"] if row else None
        finally:
            conn.close()
    return {"store_name": name or seller_id}


def _buyer_badges(request: Request) -> dict:
    """Unread notifications for the bell in the buyer header (0 when logged out)."""
    seller_id = request.path_params.get("seller_id", "")
    merchant = getattr(request.app.state, "merchant", None)
    account = session.unsign(request.cookies.get(session.customer_cookie(seller_id))) if seller_id else None
    if merchant is None or seller_id != merchant.seller_id or not account or account.get("seller_id") != seller_id:
        return {"unread_notifications": 0}
    conn = orders_db.connect(merchant.merchant_db_path, schemas=(notifications.ensure_schema,))
    try:
        return {"unread_notifications": notifications.unread_count(conn, seller_id, account["customer_id_local"])}
    finally:
        conn.close()


_seller_templates = Jinja2Templates(directory=str(_APPS_DIR / "seller" / "templates"),
                                    context_processors=[_csrf_context(session.seller_cookie)])
_buyer_templates = Jinja2Templates(directory=str(_APPS_DIR / "buyer" / "templates"),
                                   context_processors=[_csrf_context(session.customer_cookie), _store_name, _buyer_badges])

# Platform name shown on every screen. Placeholder until the team settles the
# "OO" part; change it here only.
BRAND_NAME = "오이OO"

# Thumbnail emoji: shared with the demo media generator (presentation.py).
_product_emoji = product_emoji


for _templates in (_seller_templates, _buyer_templates):
    _templates.env.globals["brand_name"] = BRAND_NAME
    _templates.env.globals["product_emoji"] = _product_emoji
    _templates.env.filters["mask_name"] = mask_name

_STATUS_LABELS = {"requested": "접수", "accepted": "처리중", "completed": "완료", "cancelled": "취소"}


def _with_total(order: dict) -> dict:
    """Screen-only total_amount, derived from order items (not part of commerce_order.v1)."""
    total = sum(item["quantity"] * item["unit_price_minor"] for item in order["items"])
    return {**order, "total_amount": total}


def _catalog_titles(conn, seller_id: str) -> dict[str, str]:
    """item_id_local -> title_text, so screens show product names instead of IDs."""
    return {item["item_id_local"]: item["title_text"]
            for item in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}


def _price_chart_points(history: list[dict]) -> Optional[str]:
    """SVG polyline 'x,y x,y ...' for M30's price sparkline, or None when there's
    nothing to draw. Computed here, not in Jinja2, since point-scaling math is
    awkward to express as template filters."""
    if len(history) < 2:
        return None
    prices = [row["price_minor"] for row in history]
    low, high = min(prices), max(prices)
    span = (high - low) or 1
    n = len(history)
    points = []
    for i, price in enumerate(prices):
        x = i * (300 / (n - 1))
        y = 75 - (price - low) / span * 70
        points.append("%.1f,%.1f" % (x, y))
    return " ".join(points)

# interfaces.md §8 HTTP mapping. Codes this service cannot yet produce (manifest/
# tensor/round codes are B<->C only) are omitted rather than guessed.
# ComparisonArm.unavailable_reason (contracts/types.py UnavailableReason) -> screen text.
_UNAVAILABLE_LABELS = {
    "model_not_ready": "공유 모델이 아직 설치되지 않음",
    "personalization_not_ready": "이 매장의 개인화를 아직 실행하지 않음",
    "insufficient_data": "개인화에 필요한 이력이 부족함",
    "validation_rejected": "개인화 결과가 검증에서 공통 모델보다 낫지 않아 쓰지 않음",
    "base_mismatch": "개인화가 현재 공통 모델과 다른 버전에서 만들어짐",
}

_HTTP_STATUS_BY_CODE = {
    "SCHEMA_INVALID": 422,
    "UNKNOWN_FIELD": 422,
    "MISSING_REQUIRED_FIELD": 422,
    "INVALID_TYPE": 422,
    "INVALID_ENUM_VALUE": 422,
    "VERSION_MISMATCH": 422,
    "DUPLICATE_EVENT": 409,
    "DUPLICATE_IDEMPOTENCY_KEY": 409,
    "ILLEGAL_STATE_TRANSITION": 409,
    "STALE_STATUS_VERSION": 409,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
}


class LoginRequired(Exception):
    """Raised by require_customer/require_seller; the exception handler below
    turns this into a redirect to the right login page instead of a 500."""

    def __init__(self, login_url: str):
        self.login_url = login_url


_REDELIVERY_JOIN_SECONDS = 60
# Tables that share orders.sqlite with the order domain; created once per file (orders_db.connect).
_SCREEN_SCHEMAS = (social_db.ensure_schema, sns_db.ensure_schema, accounts_db.ensure_schema, cart_db.ensure_schema,
                   shop_db.ensure_schema, chatbot.ensure_schema, fulfillment.ensure_schema, notifications.ensure_schema)


_CHECKOUT_ERRORS = {
    "recipient": "택배로 받으려면 받는 분 이름을 입력해 주세요.",
    "phone": "연락처를 010-1234-5678 형식으로 입력해 주세요.",
    "address": "택배 받을 주소를 입력해 주세요.",
    "method": "수령 방법을 골라 주세요.",
    "stock": "남은 수량보다 많이 주문할 수 없어요.",
}


def _checkout_error(exc: ContractError) -> str:
    field = exc.field_path.strip("/").split("/")[0]
    return field if field in _CHECKOUT_ERRORS else "method"


def _redeliver_outbox(context: MerchantContext) -> None:
    """Hand B whatever was committed but never acknowledged (interfaces.md §2),
    including everything the demo seed wrote with no runtime attached. A
    recommender failure here is logged, never a reason not to start.

    Runs on a background thread (see lifespan): B commits each event on its
    own, so a few hundred seeded orders take tens of seconds, and the app must
    answer /healthz and serve pages meanwhile. Request paths that deliver
    concurrently are safe: B accepts an identical redelivery as success."""
    try:
        conn = orders_db.connect(context.merchant_db_path)
        try:
            orders_service.deliver_all_pending(conn, seller_id=context.seller_id, runtime=context.runtime)
        finally:
            conn.close()
    except Exception as exc:
        _log.warning("startup outbox redelivery failed: %s", exc)


def settings_from_env() -> MerchantSettings:
    enabled = os.environ.get("FL_ENABLED", "false").lower()
    if enabled not in ("true", "false"):
        raise ValueError("FL_ENABLED must be true or false")
    feature_db_path = Path(os.environ["FEATURE_DB_PATH"])
    return MerchantSettings(
        seller_id=os.environ["MERCHANT_ID"],
        feature_db_path=feature_db_path,
        model_dir=Path(os.environ["MODEL_DIR"]),
        merchant_db_path=merchant_db_path_from_env(),
        fl=FLClientConfig(enabled=enabled == "true", mode=os.environ.get("FL_MODE", "protected"),
                          model_variant=os.environ.get("FL_MODEL_VARIANT", "text_relation")),
    )


class OrderItemIn(BaseModel):
    item_id_local: str
    quantity: int
    unit_price_minor: int


class PlaceOrderIn(BaseModel):
    customer_id_local: str
    idempotency_key: str
    items: list[OrderItemIn]
    currency: str


class TransitionIn(BaseModel):
    expected_status_version: int


class CatalogItemIn(BaseModel):
    item_id_local: str
    title_text: str
    description_text: Optional[str] = None
    category_path: Optional[list[str]] = None
    listing_status: str = "active"
    display_price_minor: int = 0


def create_app(settings: MerchantSettings | None = None, *, context_factory: Callable[[MerchantSettings], MerchantContext] = build_context) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        context = context_factory(settings if settings is not None else settings_from_env())
        app.state.merchant = context
        redelivery = threading.Thread(target=_redeliver_outbox, args=(context,), name="outbox-redelivery", daemon=True)
        redelivery.start()
        try:
            await context.fl_client.start()
            yield
        finally:
            try:
                await context.fl_client.stop()
            finally:
                # Let a redelivery in progress finish its current batch; a daemon
                # thread never keeps the process alive past this.
                await asyncio.to_thread(redelivery.join, _REDELIVERY_JOIN_SECONDS)
                await asyncio.to_thread(context.jobs.close)
                close = getattr(context.runtime, "close", None)  # not part of the Protocol; B's SellerRuntime has it
                if callable(close):
                    await asyncio.to_thread(close)
                del app.state.merchant

    app = FastAPI(title="Merchant service scaffold", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ContractError)
    async def handle_contract_error(request: Request, exc: ContractError):
        return JSONResponse(status_code=_HTTP_STATUS_BY_CODE.get(exc.code, 400), content=exc.to_payload())

    @app.exception_handler(LoginRequired)
    async def handle_login_required(request: Request, exc: "LoginRequired"):
        return RedirectResponse(exc.login_url, status_code=303)

    def get_conn(request: Request):
        context: MerchantContext = request.app.state.merchant
        conn = orders_db.connect(context.merchant_db_path, schemas=_SCREEN_SCHEMAS)
        try:
            yield conn
        finally:
            conn.close()

    def current_customer(seller_id: str, request: Request) -> Optional[dict]:
        account = session.unsign(request.cookies.get(session.customer_cookie(seller_id)))
        if account is None or account.get("seller_id") != seller_id or account.get("role") != "customer":
            return None
        return account

    def require_customer(seller_id: str, request: Request) -> dict:
        account = current_customer(seller_id, request)
        if account is None:
            raise LoginRequired(f"/buyer/{seller_id}/login")
        return account

    def current_seller_staff(seller_id: str, request: Request) -> Optional[dict]:
        account = session.unsign(request.cookies.get(session.seller_cookie(seller_id)))
        if account is None or account.get("seller_id") != seller_id or account.get("role") != "seller":
            return None
        return account

    def require_seller(seller_id: str, request: Request) -> dict:
        account = current_seller_staff(seller_id, request)
        if account is None:
            raise LoginRequired(f"/seller/{seller_id}/login")
        return account

    # POST forms of a logged-in user: login plus a CSRF token bound to that session cookie.
    def require_customer_form(seller_id: str, request: Request, csrf_token: Optional[str] = Form(None)) -> dict:
        account = require_customer(seller_id, request)
        if not session.check_csrf(request.cookies.get(session.customer_cookie(seller_id)), csrf_token):
            raise ContractError("FORBIDDEN", "/csrf_token")
        return account

    def require_seller_form(seller_id: str, request: Request, csrf_token: Optional[str] = Form(None)) -> dict:
        account = require_seller(seller_id, request)
        if not session.check_csrf(request.cookies.get(session.seller_cookie(seller_id)), csrf_token):
            raise ContractError("FORBIDDEN", "/csrf_token")
        return account

    @app.get("/healthz")
    def healthz():
        return {"service": "merchant", "status": "scaffold", "ready": False}

    @app.post("/sellers/{seller_id}/orders", status_code=201)
    def create_order(seller_id: str, body: PlaceOrderIn, conn=Depends(get_conn)):
        return orders_service.place_order(
            conn, seller_id=seller_id, customer_id_local=body.customer_id_local,
            idempotency_key=body.idempotency_key,
            items=[item.model_dump() for item in body.items], currency=body.currency,
        )

    @app.get("/sellers/{seller_id}/orders/{order_id}")
    def read_order(seller_id: str, order_id: str, conn=Depends(get_conn)):
        return orders_service.get_order(conn, seller_id=seller_id, order_id=order_id)

    def transition(seller_id: str, order_id: str, action: str, body: TransitionIn, request: Request, conn):
        context: MerchantContext = request.app.state.merchant
        return orders_service.transition_order(
            conn, seller_id=seller_id, order_id=order_id, action=action,
            expected_status_version=body.expected_status_version, runtime=context.runtime,
        )

    @app.post("/sellers/{seller_id}/orders/{order_id}/accept")
    def accept_order(seller_id: str, order_id: str, body: TransitionIn, request: Request, conn=Depends(get_conn)):
        return transition(seller_id, order_id, "accept", body, request, conn)

    @app.post("/sellers/{seller_id}/orders/{order_id}/complete")
    def complete_order(seller_id: str, order_id: str, body: TransitionIn, request: Request, conn=Depends(get_conn)):
        return transition(seller_id, order_id, "complete", body, request, conn)

    @app.post("/sellers/{seller_id}/orders/{order_id}/cancel")
    def cancel_order(seller_id: str, order_id: str, body: TransitionIn, request: Request, conn=Depends(get_conn)):
        return transition(seller_id, order_id, "cancel", body, request, conn)

    @app.post("/sellers/{seller_id}/catalog-items", status_code=201)
    def upsert_catalog_item(seller_id: str, body: CatalogItemIn, request: Request, conn=Depends(get_conn)):
        context: MerchantContext = request.app.state.merchant
        return orders_service.register_catalog_item(
            conn, seller_id=seller_id, item_id_local=body.item_id_local, title_text=body.title_text,
            description_text=body.description_text, category_path=body.category_path,
            listing_status=body.listing_status, display_price_minor=body.display_price_minor,
            runtime=context.runtime,
        )

    app.mount("/static", StaticFiles(directory=str(_APPS_DIR / "static")), name="static")

    @app.get("/")
    def root(request: Request):
        context: MerchantContext = request.app.state.merchant
        return RedirectResponse(f"/buyer/{context.seller_id}/")

    # --- Seller auth (business-registration gated signup) ------------------

    @app.get("/seller/{seller_id}/signup")
    def seller_signup_form(seller_id: str, request: Request):
        return _seller_templates.TemplateResponse(request, "signup.html", {
            "seller_id": seller_id, "active_tab": "", "error": None, "form": {},
        })

    @app.post("/seller/{seller_id}/signup")
    def seller_signup(
        seller_id: str, request: Request, conn=Depends(get_conn),
        username: str = Form(...), display_name: str = Form(...), password: str = Form(...),
        business_reg_no: str = Form(...), business_open_date: str = Form(...), business_rep_name: str = Form(...),
    ):
        try:
            accounts_service.signup_seller(
                conn, seller_id=seller_id, username=username, display_name=display_name, password=password,
                business_reg_no=business_reg_no, business_open_date=business_open_date,
                business_rep_name=business_rep_name,
            )
        except AccountError as exc:
            return _seller_templates.TemplateResponse(request, "signup.html", {
                "seller_id": seller_id, "active_tab": "", "error": str(exc),
                "form": {"username": username, "display_name": display_name, "business_reg_no": business_reg_no,
                         "business_open_date": business_open_date, "business_rep_name": business_rep_name},
            })
        token = session.sign({"role": "seller", "seller_id": seller_id, "username": username, "display_name": display_name})
        response = RedirectResponse(f"/seller/{seller_id}/overview", status_code=303)
        response.set_cookie(session.seller_cookie(seller_id), token, httponly=True, samesite="lax")
        return response

    @app.get("/seller/{seller_id}/login")
    def seller_login_form(seller_id: str, request: Request):
        return _seller_templates.TemplateResponse(request, "login.html", {
            "seller_id": seller_id, "active_tab": "", "error": None,
        })

    @app.post("/seller/{seller_id}/login")
    def seller_login(
        seller_id: str, request: Request, conn=Depends(get_conn),
        username: str = Form(...), password: str = Form(...),
    ):
        try:
            account = accounts_service.authenticate_seller(conn, seller_id=seller_id, username=username, password=password)
        except AccountError as exc:
            return _seller_templates.TemplateResponse(request, "login.html", {
                "seller_id": seller_id, "active_tab": "", "error": str(exc),
            })
        token = session.sign({"role": "seller", "seller_id": seller_id, "username": username,
                               "display_name": account["display_name"]})
        response = RedirectResponse(f"/seller/{seller_id}/overview", status_code=303)
        response.set_cookie(session.seller_cookie(seller_id), token, httponly=True, samesite="lax")
        return response

    @app.get("/seller/{seller_id}/logout")
    def seller_logout(seller_id: str):
        response = RedirectResponse(f"/seller/{seller_id}/login", status_code=303)
        response.delete_cookie(session.seller_cookie(seller_id))
        return response

    # --- Seller screens ---------------------------------------------------

    @app.get("/seller/{seller_id}/overview")
    def seller_overview(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        orders = orders_service.list_orders(conn, seller_id=seller_id)
        status_counts = {"requested": 0, "accepted": 0, "completed": 0, "cancelled": 0}
        for order in orders:
            status_counts[order["status"]] += 1
        context: MerchantContext = request.app.state.merchant
        # A probe for the dashboard only, for a customer who has bought here, so B answers with
        # its model if one is installed, or says why it cannot (model_status.probe_customer).
        probe = orders_service.get_recommendations_for_display(
            conn, seller_id=seller_id, customer_id_local=model_status.probe_customer(conn, seller_id),
            runtime=context.runtime,
        )
        return _seller_templates.TemplateResponse(request, "overview.html", {
            "seller_id": seller_id, "active_tab": "overview", "staff": staff,
            "product_count": len(catalog), "status_counts": status_counts,
            "model_connected": probe["model_version"] != orders_service.MOCK_MODEL_VERSION,
            "model_version": probe["model_version"], "model_fallback": probe.get("fallback_reason"),
            "fallback_label": model_status.FALLBACK_LABELS.get(probe.get("fallback_reason") or ""),
            "installed": model_status.installed_models(context.model_dir),
            "delivery": orders_service.delivery_summary(conn, seller_id=seller_id),
            "sales": (stats := sales.summary(orders, _catalog_titles(conn, seller_id))),
            "chart": sales.bar_chart(stats["series"], stats["peak"]),
            "low_stock": [(i, n) for i, n in fulfillment.stock_levels(conn, seller_id).items() if n is not None and n <= 3],
            "titles": _catalog_titles(conn, seller_id),
        })

    @app.get("/seller/{seller_id}/products")
    def seller_products(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        return _seller_templates.TemplateResponse(request, "products.html", {
            "seller_id": seller_id, "active_tab": "products", "catalog": catalog, "staff": staff,
            "stock": fulfillment.stock_levels(conn, seller_id), "photos": shop_db.photos(conn, seller_id),
        })

    @app.post("/seller/{seller_id}/products")
    def seller_create_product(
        seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller_form),
        title_text: str = Form(...), display_price_minor: int = Form(...),
        item_id_local: str = Form(...),
    ):
        context: MerchantContext = request.app.state.merchant
        if display_price_minor < 0:
            raise ContractError("INVALID_TYPE", "/display_price_minor")
        orders_service.register_catalog_item(
            conn, seller_id=seller_id, item_id_local=item_id_local, title_text=title_text,
            display_price_minor=display_price_minor, runtime=context.runtime,
        )
        return RedirectResponse(f"/seller/{seller_id}/products", status_code=303)

    @app.get("/seller/{seller_id}/orders")
    def seller_orders(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        orders = [_with_total(o) for o in orders_service.list_orders(conn, seller_id=seller_id)]
        return _seller_templates.TemplateResponse(request, "orders.html", {
            "seller_id": seller_id, "active_tab": "orders", "orders": orders,
            "status_labels": _STATUS_LABELS, "staff": staff, "titles": _catalog_titles(conn, seller_id),
            "delivery_by_order": orders_service.purchase_event_status_by_order(conn, seller_id=seller_id),
            "fulfillment": fulfillment.of_orders(conn, seller_id), "methods": fulfillment.METHODS,
            "status_filter": request.query_params.get("status"),
        })

    def seller_transition_screen(seller_id: str, order_id: str, action: str, request: Request, conn, expected_status_version: int):
        context: MerchantContext = request.app.state.merchant
        order = orders_service.transition_order(
            conn, seller_id=seller_id, order_id=order_id, action=action,
            expected_status_version=expected_status_version, runtime=context.runtime,
        )
        # A-local follow-ups: the timeline's accept time, held stock back on cancel, the buyer's bell.
        with conn:
            fulfillment.log_status(conn, seller_id, order_id, order["status"])
        if action == "cancel":
            fulfillment.release_stock(conn, seller_id, order_id)
        notifications.on_order(conn, seller_id, order, action)
        back = request.query_params.get("back")
        return RedirectResponse(f"/seller/{seller_id}/orders/{order_id}" if back == "detail" else f"/seller/{seller_id}/orders",
                                status_code=303)

    @app.post("/seller/{seller_id}/orders/{order_id}/accept")
    def seller_accept_order(
        seller_id: str, order_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller_form),
        expected_status_version: int = Form(...),
    ):
        return seller_transition_screen(seller_id, order_id, "accept", request, conn, expected_status_version)

    @app.post("/seller/{seller_id}/orders/{order_id}/complete")
    def seller_complete_order(
        seller_id: str, order_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller_form),
        expected_status_version: int = Form(...),
    ):
        return seller_transition_screen(seller_id, order_id, "complete", request, conn, expected_status_version)

    @app.post("/seller/{seller_id}/orders/{order_id}/cancel")
    def seller_cancel_order(
        seller_id: str, order_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller_form),
        expected_status_version: int = Form(...),
    ):
        return seller_transition_screen(seller_id, order_id, "cancel", request, conn, expected_status_version)

    # --- Seller model comparison (comparison.md §5, A card "모델 비교 화면") -----

    @app.get("/seller/{seller_id}/compare")
    def seller_compare(
        seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller),
        customer_id_local: Optional[str] = None,
    ):
        context: MerchantContext = request.app.state.merchant
        result = None
        if customer_id_local:
            result = orders_service.get_comparison_for_display(
                conn, seller_id=seller_id, customer_id_local=customer_id_local, runtime=context.runtime,
            )
        titles = _catalog_titles(conn, seller_id)
        return _seller_templates.TemplateResponse(request, "compare.html", {
            "seller_id": seller_id, "active_tab": "compare", "staff": staff,
            "customers": accounts_db.list_customers(conn, seller_id),
            "customer_id_local": customer_id_local, "result": result, "titles": titles,
            "unavailable_labels": _UNAVAILABLE_LABELS,
            "personalization": personalization.last_run(seller_id),
            "personalization_notice": request.query_params.get("p"),
        })

    @app.post("/seller/{seller_id}/personalize")
    def seller_personalize(seller_id: str, request: Request, staff=Depends(require_seller_form),
                           customer_id_local: str = Form("")):
        context: MerchantContext = request.app.state.merchant
        outcome = personalization.start(context)
        back = f"/seller/{seller_id}/compare?p={outcome}"
        return RedirectResponse(back + (f"&customer_id_local={customer_id_local}" if customer_id_local else ""), status_code=303)

    # --- Seller social screens (M25 DM, M26/M27 feed, M29 group-buy, M30 price) --

    @app.get("/seller/{seller_id}/messages")
    def seller_messages(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        threads = social_service.list_threads(conn, seller_id=seller_id)
        return _seller_templates.TemplateResponse(request, "messages.html", {
            "seller_id": seller_id, "active_tab": "messages", "threads": threads, "staff": staff,
        })

    @app.get("/seller/{seller_id}/messages/{customer_id_local}")
    def seller_message_thread(seller_id: str, customer_id_local: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        messages = social_service.list_thread_messages(conn, seller_id=seller_id, customer_id_local=customer_id_local)
        return _seller_templates.TemplateResponse(request, "message_thread.html", {
            "seller_id": seller_id, "active_tab": "messages", "staff": staff,
            "customer_id_local": customer_id_local, "messages": messages,
        })

    @app.post("/seller/{seller_id}/messages/{customer_id_local}")
    def seller_send_message(seller_id: str, customer_id_local: str, conn=Depends(get_conn), staff=Depends(require_seller_form), body: str = Form(...)):
        if accounts_db.fetch_customer(conn, seller_id, customer_id_local) is None:
            raise ContractError("NOT_FOUND", "/customer_id_local")
        social_service.send_message(conn, seller_id=seller_id, customer_id_local=customer_id_local, sender="seller", body=body)
        notifications.notify(conn, seller_id, [customer_id_local], "message", "판매자가 쪽지를 보냈어요: " + body.strip()[:40],
                             f"/buyer/{seller_id}/messages")
        return RedirectResponse(f"/seller/{seller_id}/messages/{customer_id_local}", status_code=303)

    @app.get("/seller/{seller_id}/prices")
    def seller_prices(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        histories = {
            item["item_id_local"]: social_service.get_price_history(conn, seller_id=seller_id, item_id_local=item["item_id_local"])
            for item in catalog
        }
        return _seller_templates.TemplateResponse(request, "prices.html", {
            "seller_id": seller_id, "active_tab": "prices", "catalog": catalog, "histories": histories, "staff": staff,
        })

    @app.post("/seller/{seller_id}/prices")
    def seller_record_price(
        seller_id: str, conn=Depends(get_conn), staff=Depends(require_seller_form),
        item_id_local: str = Form(...), price_minor: int = Form(...),
    ):
        social_service.record_price(conn, seller_id=seller_id, item_id_local=item_id_local, price_minor=price_minor)
        return RedirectResponse(f"/seller/{seller_id}/prices", status_code=303)

    # --- Buyer auth ----------------------------------------------------------

    @app.get("/buyer/{seller_id}/signup")
    def buyer_signup_form(seller_id: str, request: Request):
        return _buyer_templates.TemplateResponse(request, "signup.html", {
            "seller_id": seller_id, "active_tab": "", "error": None, "form": {},
        })

    @app.post("/buyer/{seller_id}/signup")
    def buyer_signup(
        seller_id: str, request: Request, conn=Depends(get_conn),
        customer_id_local: str = Form(...), display_name: str = Form(...), password: str = Form(...),
    ):
        try:
            accounts_service.signup_customer(
                conn, seller_id=seller_id, customer_id_local=customer_id_local,
                display_name=display_name, password=password,
            )
        except AccountError as exc:
            return _buyer_templates.TemplateResponse(request, "signup.html", {
                "seller_id": seller_id, "active_tab": "", "error": str(exc),
                "form": {"customer_id_local": customer_id_local, "display_name": display_name},
            })
        token = session.sign({"role": "customer", "seller_id": seller_id, "customer_id_local": customer_id_local,
                               "display_name": display_name})
        response = RedirectResponse(f"/buyer/{seller_id}/", status_code=303)
        response.set_cookie(session.customer_cookie(seller_id), token, httponly=True, samesite="lax")
        return response

    @app.get("/buyer/{seller_id}/login")
    def buyer_login_form(seller_id: str, request: Request):
        return _buyer_templates.TemplateResponse(request, "login.html", {
            "seller_id": seller_id, "active_tab": "", "error": None,
        })

    @app.post("/buyer/{seller_id}/login")
    def buyer_login(
        seller_id: str, request: Request, conn=Depends(get_conn),
        customer_id_local: str = Form(...), password: str = Form(...),
    ):
        try:
            account = accounts_service.authenticate_customer(
                conn, seller_id=seller_id, customer_id_local=customer_id_local, password=password,
            )
        except AccountError as exc:
            return _buyer_templates.TemplateResponse(request, "login.html", {
                "seller_id": seller_id, "active_tab": "", "error": str(exc),
            })
        token = session.sign({"role": "customer", "seller_id": seller_id, "customer_id_local": customer_id_local,
                               "display_name": account["display_name"]})
        response = RedirectResponse(f"/buyer/{seller_id}/", status_code=303)
        response.set_cookie(session.customer_cookie(seller_id), token, httponly=True, samesite="lax")
        return response

    @app.get("/buyer/{seller_id}/logout")
    def buyer_logout(seller_id: str):
        response = RedirectResponse(f"/buyer/{seller_id}/", status_code=303)
        response.delete_cookie(session.customer_cookie(seller_id))
        return response

    # --- Buyer screens -----------------------------------------------------

    @app.get("/buyer/{seller_id}/")
    def buyer_home(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(current_customer),
                   q: Optional[str] = None, cat: Optional[str] = None):
        context: MerchantContext = request.app.state.merchant
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        recommendation = orders_service.get_recommendations_for_display(
            conn, seller_id=seller_id, customer_id_local=(customer["customer_id_local"] if customer else "guest"),
            runtime=context.runtime,
        )
        catalog_by_id = {item["item_id_local"]: item for item in catalog}
        recommended = [
            catalog_by_id[entry["item_id_local"]] for entry in recommendation["items"]
            if entry["item_id_local"] in catalog_by_id
        ]
        return _buyer_templates.TemplateResponse(request, "home.html", {
            "seller_id": seller_id, "active_tab": "home", "customer": customer,
            "catalog": shop_service.search(catalog, q, cat), "q": q or "", "cat": cat,
            "categories": shop_service.categories(catalog),
            "ratings": shop_db.rating_summary(conn, seller_id),
            "wished": set(shop_db.wishlist(conn, seller_id, customer["customer_id_local"])) if customer else set(),
            "recommended": [i for i in recommended if i["listing_status"] == "active"],
            "recommendation_label": orders_service.recommendation_label(recommendation),
            "photos": shop_db.photos(conn, seller_id), "stock": fulfillment.stock_levels(conn, seller_id),
        })

    @app.get("/buyer/{seller_id}/items/{item_id_local}")
    def buyer_item_detail(seller_id: str, item_id_local: str, request: Request, conn=Depends(get_conn), customer=Depends(current_customer)):
        item = orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local)
        if item is None:
            raise ContractError("NOT_FOUND", "/item_id_local")
        price_history = social_service.get_price_history(conn, seller_id=seller_id, item_id_local=item_id_local)
        customer_id = customer["customer_id_local"] if customer else None
        reviews = shop_db.reviews_for_item(conn, seller_id, item_id_local)
        return _buyer_templates.TemplateResponse(request, "product.html", {
            "seller_id": seller_id, "active_tab": "home", "item": item, "price_history": price_history,
            "price_chart_points": _price_chart_points(price_history), "customer": customer,
            "reviews": reviews, "rating": shop_db.rating_summary(conn, seller_id).get(item_id_local),
            "my_review": next((r for r in reviews if r["customer_id_local"] == customer_id), None),
            "can_review": bool(customer_id) and shop_service.bought(conn, seller_id, customer_id, item_id_local),
            "wished": customer_id is not None and item_id_local in shop_db.wishlist(conn, seller_id, customer_id),
            "posts": sns_service.posts_for_item(conn, seller_id=seller_id, item_id_local=item_id_local),
            "group_price": social_service.group_price(conn, seller_id=seller_id, item_id_local=item_id_local)
            if item["listing_status"] == "active" else None,
            "photos": shop_db.photos(conn, seller_id).get(item_id_local, []),
            "stock_left": fulfillment.stock_levels(conn, seller_id).get(item_id_local),
            "terms": fulfillment.store_terms(conn, seller_id), "slots": fulfillment.PICKUP_SLOTS,
            "profile": fulfillment.profile(conn, seller_id, customer_id) if customer_id else None,
            "checkout_error": _CHECKOUT_ERRORS.get(request.query_params.get("e") or ""),
            "names": accounts_db.display_names(conn, seller_id),
        })

    def order_placed(request: Request, conn, seller_id: str, customer: dict, order: dict, details: dict):
        terms = fulfillment.store_terms(conn, seller_id)
        subtotal = _with_total(order)["total_amount"]
        fulfillment.record(conn, seller_id, order, customer["customer_id_local"], details,
                           fulfillment.shipping_fee(terms, details["method"], subtotal))
        fulfillment.hold_stock(conn, seller_id, order)
        context: MerchantContext = request.app.state.merchant
        installed = model_status.installed_models(context.model_dir)
        return _buyer_templates.TemplateResponse(request, "order_confirmation.html", {
            "seller_id": seller_id, "active_tab": "orders", "order": _with_total(order), "customer": customer,
            "titles": _catalog_titles(conn, seller_id), "terms": terms,
            "model_connected": installed["encoder"] and any(installed["variants"].values()),
            **routes_store._order_view(conn, seller_id, order),
        })

    @app.post("/buyer/{seller_id}/orders")
    def buyer_place_order(
        seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer_form),
        item_id_local: str = Form(...), quantity: int = Form(...),
        method: str = Form("pickup"), recipient: str = Form(""), phone: str = Form(""), address: str = Form(""),
        pickup_slot: str = Form(""), memo: str = Form(""),
    ):
        # The price is the server's catalog price; the form only says which item and how many.
        item = orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local)
        if item is None or item["listing_status"] != "active":
            raise ContractError("NOT_FOUND", "/item_id_local")
        if quantity < 1:
            raise ContractError("SCHEMA_INVALID", "/quantity")
        try:
            details = fulfillment.validate(method, recipient, phone, address, pickup_slot, memo)
            fulfillment.check_stock(conn, seller_id, [(item_id_local, quantity)])
        except ContractError as exc:
            return RedirectResponse(f"/buyer/{seller_id}/items/{item_id_local}?e={_checkout_error(exc)}#order", status_code=303)
        unit_price_minor = item["display_price_minor"]
        order = orders_service.place_order(
            conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
            idempotency_key=uuid.uuid4().hex,
            items=[{"item_id_local": item_id_local, "quantity": quantity, "unit_price_minor": unit_price_minor}],
            currency="KRW",
        )
        return order_placed(request, conn, seller_id, customer, order, details)

    @app.get("/buyer/{seller_id}/cart")
    def buyer_cart(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer)):
        cart = orders_service.get_cart(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"])
        return _buyer_templates.TemplateResponse(request, "cart.html", {
            "seller_id": seller_id, "active_tab": "cart", "customer": customer, "cart": cart,
            "titles": _catalog_titles(conn, seller_id),
            # A fresh key per rendered cart: resubmitting this page cannot order twice.
            "checkout_key": "cart-" + uuid.uuid4().hex,
            "terms": (terms := fulfillment.store_terms(conn, seller_id)), "slots": fulfillment.PICKUP_SLOTS,
            "profile": fulfillment.profile(conn, seller_id, customer["customer_id_local"]),
            "delivery_fee": fulfillment.shipping_fee(terms, "delivery", cart["total"]),
            "checkout_error": _CHECKOUT_ERRORS.get(request.query_params.get("e") or ""),
            "photos": shop_db.photos(conn, seller_id),
        })

    @app.post("/buyer/{seller_id}/cart")
    def buyer_add_to_cart(
        seller_id: str, conn=Depends(get_conn), customer=Depends(require_customer_form),
        item_id_local: str = Form(...), quantity: int = Form(1),
    ):
        orders_service.add_to_cart(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                   item_id_local=item_id_local, quantity=quantity)
        return RedirectResponse(f"/buyer/{seller_id}/cart", status_code=303)

    @app.post("/buyer/{seller_id}/cart/{item_id_local}")
    def buyer_update_cart(
        seller_id: str, item_id_local: str, conn=Depends(get_conn), customer=Depends(require_customer_form),
        quantity: int = Form(...),
    ):
        orders_service.set_cart_quantity(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                         item_id_local=item_id_local, quantity=quantity)
        return RedirectResponse(f"/buyer/{seller_id}/cart", status_code=303)

    @app.post("/buyer/{seller_id}/checkout")
    def buyer_checkout(
        seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer_form),
        checkout_key: str = Form(...),
        method: str = Form("pickup"), recipient: str = Form(""), phone: str = Form(""), address: str = Form(""),
        pickup_slot: str = Form(""), memo: str = Form(""),
    ):
        try:
            details = fulfillment.validate(method, recipient, phone, address, pickup_slot, memo)
            if orders_db.fetch_order_by_idempotency_key(conn, seller_id, checkout_key) is None:
                cart = orders_service.get_cart(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"])
                fulfillment.check_stock(conn, seller_id, [(l["item_id_local"], l["quantity"]) for l in cart["lines"]])
        except ContractError as exc:
            return RedirectResponse(f"/buyer/{seller_id}/cart?e={_checkout_error(exc)}", status_code=303)
        order = orders_service.checkout_cart(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
                                             checkout_key=checkout_key)
        return order_placed(request, conn, seller_id, customer, order, details)

    @app.get("/buyer/{seller_id}/orders")
    def buyer_orders(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer)):
        orders = [
            _with_total(o) for o in
            orders_service.list_orders_by_customer(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"])
        ]
        return _buyer_templates.TemplateResponse(request, "orders.html", {
            "seller_id": seller_id, "active_tab": "orders", "orders": orders,
            "status_labels": _STATUS_LABELS, "customer": customer, "titles": _catalog_titles(conn, seller_id),
            "fulfillment": fulfillment.of_orders(conn, seller_id), "methods": fulfillment.METHODS,
            "photos": shop_db.photos(conn, seller_id),
        })

    @app.get("/buyer/{seller_id}/messages")
    def buyer_messages(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer)):
        messages = social_service.list_thread_messages(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"])
        return _buyer_templates.TemplateResponse(request, "messages.html", {
            "seller_id": seller_id, "active_tab": "messages", "customer": customer, "messages": messages,
        })

    @app.post("/buyer/{seller_id}/messages")
    def buyer_send_message(seller_id: str, conn=Depends(get_conn), customer=Depends(require_customer_form), body: str = Form(...)):
        social_service.send_message(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"], sender="customer", body=body)
        return RedirectResponse(f"/buyer/{seller_id}/messages", status_code=303)

    deps = routes_sns.ScreenDeps(get_conn, current_customer, require_customer, require_customer_form,
                                 require_seller, require_seller_form, _buyer_templates, _seller_templates)
    routes_sns.register(app, deps)
    routes_group.register(app, deps)
    routes_shop.register(app, deps)
    routes_chat.register(app, deps, BRAND_NAME)
    routes_store.register(app, deps)
    return app


app = create_app()
