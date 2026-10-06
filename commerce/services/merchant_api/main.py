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
import os
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
from commerce.services.merchant_api import accounts_db, accounts_service, orders_db, orders_service, session, social_db, social_service
from commerce.services.merchant_api.accounts_service import AccountError
from commerce.services.merchant_api.context import MerchantContext, MerchantSettings, build_context

_APPS_DIR = Path(__file__).resolve().parents[2] / "apps"


def _csrf_context(cookie_name: str):
    def processor(request: Request) -> dict:
        return {"csrf_token": session.csrf_token(request.cookies.get(cookie_name))}
    return processor


_seller_templates = Jinja2Templates(directory=str(_APPS_DIR / "seller" / "templates"),
                                    context_processors=[_csrf_context(session.SELLER_COOKIE)])
_buyer_templates = Jinja2Templates(directory=str(_APPS_DIR / "buyer" / "templates"),
                                   context_processors=[_csrf_context(session.CUSTOMER_COOKIE)])

_STATUS_LABELS = {"requested": "접수", "accepted": "처리중", "completed": "완료", "cancelled": "취소"}


def _with_total(order: dict) -> dict:
    """Screen-only total_amount, derived from order items (not part of commerce_order.v1)."""
    total = sum(item["quantity"] * item["unit_price_minor"] for item in order["items"])
    return {**order, "total_amount": total}


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


def settings_from_env() -> MerchantSettings:
    enabled = os.environ.get("FL_ENABLED", "false").lower()
    if enabled not in ("true", "false"):
        raise ValueError("FL_ENABLED must be true or false")
    feature_db_path = Path(os.environ["FEATURE_DB_PATH"])
    return MerchantSettings(
        seller_id=os.environ["MERCHANT_ID"],
        feature_db_path=feature_db_path,
        model_dir=Path(os.environ["MODEL_DIR"]),
        # Not yet in run_local.py's plan (C-owned); default keeps the launcher
        # working until a PR adds it there (working-agreement.md §3).
        merchant_db_path=Path(os.environ.get("MERCHANT_DB_PATH", str(feature_db_path.parent / "orders.sqlite"))),
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
        try:
            await context.fl_client.start()
            yield
        finally:
            try:
                await context.fl_client.stop()
            finally:
                await asyncio.to_thread(context.jobs.close)
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
        conn = orders_db.connect(context.merchant_db_path)
        social_db.ensure_schema(conn)
        accounts_db.ensure_schema(conn)
        try:
            yield conn
        finally:
            conn.close()

    def current_customer(seller_id: str, request: Request) -> Optional[dict]:
        account = session.unsign(request.cookies.get(session.CUSTOMER_COOKIE))
        if account is None or account.get("seller_id") != seller_id or account.get("role") != "customer":
            return None
        return account

    def require_customer(seller_id: str, request: Request) -> dict:
        account = current_customer(seller_id, request)
        if account is None:
            raise LoginRequired(f"/buyer/{seller_id}/login")
        return account

    def current_seller_staff(seller_id: str, request: Request) -> Optional[dict]:
        account = session.unsign(request.cookies.get(session.SELLER_COOKIE))
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
        if not session.check_csrf(request.cookies.get(session.CUSTOMER_COOKIE), csrf_token):
            raise ContractError("FORBIDDEN", "/csrf_token")
        return account

    def require_seller_form(seller_id: str, request: Request, csrf_token: Optional[str] = Form(None)) -> dict:
        account = require_seller(seller_id, request)
        if not session.check_csrf(request.cookies.get(session.SELLER_COOKIE), csrf_token):
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
        response.set_cookie(session.SELLER_COOKIE, token, httponly=True, samesite="lax")
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
        response.set_cookie(session.SELLER_COOKIE, token, httponly=True, samesite="lax")
        return response

    @app.get("/seller/{seller_id}/logout")
    def seller_logout(seller_id: str):
        response = RedirectResponse(f"/seller/{seller_id}/login", status_code=303)
        response.delete_cookie(session.SELLER_COOKIE)
        return response

    # --- Seller screens ---------------------------------------------------

    @app.get("/seller/{seller_id}/overview")
    def seller_overview(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        orders = orders_service.list_orders(conn, seller_id=seller_id)
        status_counts = {"requested": 0, "accepted": 0, "completed": 0, "cancelled": 0}
        for order in orders:
            status_counts[order["status"]] += 1
        return _seller_templates.TemplateResponse(request, "overview.html", {
            "seller_id": seller_id, "active_tab": "overview", "staff": staff,
            "product_count": len(catalog), "status_counts": status_counts,
        })

    @app.get("/seller/{seller_id}/products")
    def seller_products(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        return _seller_templates.TemplateResponse(request, "products.html", {
            "seller_id": seller_id, "active_tab": "products", "catalog": catalog, "staff": staff,
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
            "status_labels": _STATUS_LABELS, "staff": staff,
        })

    def seller_transition_screen(seller_id: str, order_id: str, action: str, request: Request, conn, expected_status_version: int):
        context: MerchantContext = request.app.state.merchant
        orders_service.transition_order(
            conn, seller_id=seller_id, order_id=order_id, action=action,
            expected_status_version=expected_status_version, runtime=context.runtime,
        )
        return RedirectResponse(f"/seller/{seller_id}/orders", status_code=303)

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
        titles = {item["item_id_local"]: item["title_text"]
                  for item in orders_service.list_catalog_for_display(conn, seller_id=seller_id)}
        return _seller_templates.TemplateResponse(request, "compare.html", {
            "seller_id": seller_id, "active_tab": "compare", "staff": staff,
            "customers": accounts_db.list_customers(conn, seller_id),
            "customer_id_local": customer_id_local, "result": result, "titles": titles,
            "unavailable_labels": _UNAVAILABLE_LABELS,
        })

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
        return RedirectResponse(f"/seller/{seller_id}/messages/{customer_id_local}", status_code=303)

    @app.get("/seller/{seller_id}/feed")
    def seller_feed(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        posts = social_service.list_feed(conn, seller_id=seller_id)
        return _seller_templates.TemplateResponse(request, "feed.html", {
            "seller_id": seller_id, "active_tab": "feed", "posts": posts, "staff": staff,
        })

    @app.post("/seller/{seller_id}/feed")
    def seller_create_post(
        seller_id: str, conn=Depends(get_conn), staff=Depends(require_seller_form), kind: str = Form(...),
        title: str = Form(...), body: str = Form(None),
    ):
        social_service.create_post(conn, seller_id=seller_id, kind=kind, title=title, body=body)
        return RedirectResponse(f"/seller/{seller_id}/feed", status_code=303)

    @app.get("/seller/{seller_id}/group-buys")
    def seller_group_buys(seller_id: str, request: Request, conn=Depends(get_conn), staff=Depends(require_seller)):
        group_buys = social_service.list_group_buys(conn, seller_id=seller_id)
        return _seller_templates.TemplateResponse(request, "group_buys.html", {
            "seller_id": seller_id, "active_tab": "group_buys", "group_buys": group_buys, "staff": staff,
        })

    @app.post("/seller/{seller_id}/group-buys")
    def seller_create_group_buy(
        seller_id: str, conn=Depends(get_conn), staff=Depends(require_seller_form), item_id_local: str = Form(...),
        target_quantity: int = Form(...), unit_price_minor: int = Form(...), deadline_at: str = Form(...),
    ):
        social_service.create_group_buy(
            conn, seller_id=seller_id, item_id_local=item_id_local, target_quantity=target_quantity,
            unit_price_minor=unit_price_minor, deadline_at=deadline_at,
        )
        return RedirectResponse(f"/seller/{seller_id}/group-buys", status_code=303)

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
        response.set_cookie(session.CUSTOMER_COOKIE, token, httponly=True, samesite="lax")
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
        response.set_cookie(session.CUSTOMER_COOKIE, token, httponly=True, samesite="lax")
        return response

    @app.get("/buyer/{seller_id}/logout")
    def buyer_logout(seller_id: str):
        response = RedirectResponse(f"/buyer/{seller_id}/", status_code=303)
        response.delete_cookie(session.CUSTOMER_COOKIE)
        return response

    # --- Buyer screens -----------------------------------------------------

    @app.get("/buyer/{seller_id}/")
    def buyer_home(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(current_customer)):
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
            "seller_id": seller_id, "active_tab": "home", "catalog": catalog, "customer": customer,
            "recommended": recommended,
            "recommendation_label": orders_service.recommendation_label(recommendation),
        })

    @app.get("/buyer/{seller_id}/items/{item_id_local}")
    def buyer_item_detail(seller_id: str, item_id_local: str, request: Request, conn=Depends(get_conn), customer=Depends(current_customer)):
        item = orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local)
        if item is None:
            raise ContractError("NOT_FOUND", "/item_id_local")
        price_history = social_service.get_price_history(conn, seller_id=seller_id, item_id_local=item_id_local)
        return _buyer_templates.TemplateResponse(request, "product.html", {
            "seller_id": seller_id, "active_tab": "home", "item": item, "price_history": price_history,
            "price_chart_points": _price_chart_points(price_history), "customer": customer,
        })

    @app.post("/buyer/{seller_id}/orders")
    def buyer_place_order(
        seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer_form),
        item_id_local: str = Form(...), quantity: int = Form(...),
    ):
        # The price is the server's catalog price; the form only says which item and how many.
        item = orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local)
        if item is None or item["listing_status"] != "active":
            raise ContractError("NOT_FOUND", "/item_id_local")
        if quantity < 1:
            raise ContractError("SCHEMA_INVALID", "/quantity")
        unit_price_minor = item["display_price_minor"]
        order = orders_service.place_order(
            conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"],
            idempotency_key=uuid.uuid4().hex,
            items=[{"item_id_local": item_id_local, "quantity": quantity, "unit_price_minor": unit_price_minor}],
            currency="KRW",
        )
        return _buyer_templates.TemplateResponse(request, "order_confirmation.html", {
            "seller_id": seller_id, "active_tab": "orders", "order": _with_total(order), "customer": customer,
        })

    @app.get("/buyer/{seller_id}/orders")
    def buyer_orders(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(require_customer)):
        orders = [
            _with_total(o) for o in
            orders_service.list_orders_by_customer(conn, seller_id=seller_id, customer_id_local=customer["customer_id_local"])
        ]
        return _buyer_templates.TemplateResponse(request, "orders.html", {
            "seller_id": seller_id, "active_tab": "orders", "orders": orders,
            "status_labels": _STATUS_LABELS, "customer": customer,
        })

    @app.get("/buyer/{seller_id}/chat")
    def buyer_chat(seller_id: str, request: Request, customer=Depends(current_customer)):
        return _buyer_templates.TemplateResponse(request, "chat.html", {
            "seller_id": seller_id, "active_tab": "chat", "customer": customer,
        })

    # --- Buyer social screens (M25 DM, M26/M27 feed, M29 group-buy) ------------

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

    @app.get("/buyer/{seller_id}/feed")
    def buyer_feed(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(current_customer)):
        posts = social_service.list_feed(conn, seller_id=seller_id)
        return _buyer_templates.TemplateResponse(request, "feed.html", {
            "seller_id": seller_id, "active_tab": "feed", "posts": posts, "customer": customer,
        })

    @app.get("/buyer/{seller_id}/group-buys")
    def buyer_group_buys(seller_id: str, request: Request, conn=Depends(get_conn), customer=Depends(current_customer)):
        group_buys = social_service.list_group_buys(conn, seller_id=seller_id)
        return _buyer_templates.TemplateResponse(request, "group_buys.html", {
            "seller_id": seller_id, "active_tab": "group_buys", "group_buys": group_buys, "customer": customer,
        })

    @app.post("/buyer/{seller_id}/group-buys/{group_buy_id}/join")
    def buyer_join_group_buy(
        seller_id: str, group_buy_id: str, conn=Depends(get_conn), customer=Depends(require_customer_form),
        quantity: int = Form(...),
    ):
        social_service.join_group_buy(
            conn, seller_id=seller_id, group_buy_id=group_buy_id,
            customer_id_local=customer["customer_id_local"], quantity=quantity,
        )
        return RedirectResponse(f"/buyer/{seller_id}/group-buys", status_code=303)

    return app


app = create_app()
