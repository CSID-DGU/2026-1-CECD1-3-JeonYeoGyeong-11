"""Merchant app: health, lifecycle, and the order/catalog domain routes.

Caller authentication is intentionally absent from every route below. Who is
allowed to act as a given seller_id is exactly OQ13 (auth error code) and OQ15
(order_id in seller-origin URLs), both open human decisions
(docs/design/open-questions.md). A card: "그 결정 전에는 주문 상태 전이·
transaction·outbox 같은 도메인 계층부터 만든다" -- so this module exposes the
domain layer over HTTP for local/integration testing only. Do not point a real
buyer or seller browser at these routes before OQ13/OQ15 land and a caller
identity check is added here.
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
from commerce.services.merchant_api import orders_db, orders_service
from commerce.services.merchant_api.context import MerchantContext, MerchantSettings, build_context

_APPS_DIR = Path(__file__).resolve().parents[2] / "apps"
_seller_templates = Jinja2Templates(directory=str(_APPS_DIR / "seller" / "templates"))
_buyer_templates = Jinja2Templates(directory=str(_APPS_DIR / "buyer" / "templates"))

# Screens below render the same domain layer as the JSON routes but are meant
# for local/manual browsing only -- they carry the same "no caller auth yet"
# caveat as the module docstring (OQ13/OQ15).
_STATUS_LABELS = {"requested": "접수", "accepted": "처리중", "completed": "완료", "cancelled": "취소"}


def _with_total(order: dict) -> dict:
    """Screen-only total_amount, derived from order items (not part of commerce_order.v1)."""
    total = sum(item["quantity"] * item["unit_price_minor"] for item in order["items"])
    return {**order, "total_amount": total}

# interfaces.md §8 HTTP mapping. Codes this service cannot yet produce (manifest/
# tensor/round codes are B<->C only) are omitted rather than guessed.
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

    def get_conn(request: Request):
        context: MerchantContext = request.app.state.merchant
        conn = orders_db.connect(context.merchant_db_path)
        try:
            yield conn
        finally:
            conn.close()

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

    # --- Seller screens ---------------------------------------------------

    @app.get("/seller/{seller_id}/overview")
    def seller_overview(seller_id: str, request: Request, conn=Depends(get_conn)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        orders = orders_service.list_orders(conn, seller_id=seller_id)
        status_counts = {"requested": 0, "accepted": 0, "completed": 0, "cancelled": 0}
        for order in orders:
            status_counts[order["status"]] += 1
        return _seller_templates.TemplateResponse(request, "overview.html", {
            "seller_id": seller_id, "active_tab": "overview",
            "product_count": len(catalog), "status_counts": status_counts,
        })

    @app.get("/seller/{seller_id}/products")
    def seller_products(seller_id: str, request: Request, conn=Depends(get_conn)):
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        return _seller_templates.TemplateResponse(request, "products.html", {
            "seller_id": seller_id, "active_tab": "products", "catalog": catalog,
        })

    @app.post("/seller/{seller_id}/products")
    def seller_create_product(
        seller_id: str, request: Request, conn=Depends(get_conn),
        title_text: str = Form(...), display_price_minor: int = Form(...),
        item_id_local: str = Form(...),
    ):
        context: MerchantContext = request.app.state.merchant
        orders_service.register_catalog_item(
            conn, seller_id=seller_id, item_id_local=item_id_local, title_text=title_text,
            display_price_minor=display_price_minor, runtime=context.runtime,
        )
        return RedirectResponse(f"/seller/{seller_id}/products", status_code=303)

    @app.get("/seller/{seller_id}/orders")
    def seller_orders(seller_id: str, request: Request, conn=Depends(get_conn)):
        orders = [_with_total(o) for o in orders_service.list_orders(conn, seller_id=seller_id)]
        return _seller_templates.TemplateResponse(request, "orders.html", {
            "seller_id": seller_id, "active_tab": "orders", "orders": orders,
            "status_labels": _STATUS_LABELS,
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
        seller_id: str, order_id: str, request: Request, conn=Depends(get_conn),
        expected_status_version: int = Form(...),
    ):
        return seller_transition_screen(seller_id, order_id, "accept", request, conn, expected_status_version)

    @app.post("/seller/{seller_id}/orders/{order_id}/complete")
    def seller_complete_order(
        seller_id: str, order_id: str, request: Request, conn=Depends(get_conn),
        expected_status_version: int = Form(...),
    ):
        return seller_transition_screen(seller_id, order_id, "complete", request, conn, expected_status_version)

    # --- Buyer screens -----------------------------------------------------

    @app.get("/buyer/{seller_id}/")
    def buyer_home(seller_id: str, request: Request, conn=Depends(get_conn), customer_id_local: str = "guest-1"):
        context: MerchantContext = request.app.state.merchant
        catalog = orders_service.list_catalog_for_display(conn, seller_id=seller_id)
        recommendation = orders_service.get_recommendations_for_display(
            conn, seller_id=seller_id, customer_id_local=customer_id_local, runtime=context.runtime,
        )
        catalog_by_id = {item["item_id_local"]: item for item in catalog}
        recommended = [
            catalog_by_id[entry["item_id_local"]] for entry in recommendation["items"]
            if entry["item_id_local"] in catalog_by_id
        ]
        return _buyer_templates.TemplateResponse(request, "home.html", {
            "seller_id": seller_id, "active_tab": "home", "catalog": catalog,
            "recommended": recommended,
            "recommendation_is_mock": recommendation["model_version"] == orders_service.MOCK_MODEL_VERSION,
        })

    @app.get("/buyer/{seller_id}/items/{item_id_local}")
    def buyer_item_detail(seller_id: str, item_id_local: str, request: Request, conn=Depends(get_conn)):
        item = orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id_local)
        if item is None:
            raise ContractError("NOT_FOUND", "/item_id_local")
        return _buyer_templates.TemplateResponse(request, "product.html", {
            "seller_id": seller_id, "active_tab": "home", "item": item,
        })

    @app.post("/buyer/{seller_id}/orders")
    def buyer_place_order(
        seller_id: str, request: Request, conn=Depends(get_conn),
        item_id_local: str = Form(...), unit_price_minor: int = Form(...),
        quantity: int = Form(...), customer_id_local: str = Form(...),
    ):
        order = orders_service.place_order(
            conn, seller_id=seller_id, customer_id_local=customer_id_local,
            idempotency_key=uuid.uuid4().hex,
            items=[{"item_id_local": item_id_local, "quantity": quantity, "unit_price_minor": unit_price_minor}],
            currency="KRW",
        )
        return _buyer_templates.TemplateResponse(request, "order_confirmation.html", {
            "seller_id": seller_id, "active_tab": "orders", "order": _with_total(order),
        })

    @app.get("/buyer/{seller_id}/orders")
    def buyer_orders(seller_id: str, request: Request, conn=Depends(get_conn), customer_id_local: str = "guest-1"):
        orders = [
            _with_total(o) for o in
            orders_service.list_orders_by_customer(conn, seller_id=seller_id, customer_id_local=customer_id_local)
        ]
        return _buyer_templates.TemplateResponse(request, "orders.html", {
            "seller_id": seller_id, "active_tab": "orders", "orders": orders,
            "status_labels": _STATUS_LABELS, "customer_id_local": customer_id_local,
        })

    @app.get("/buyer/{seller_id}/chat")
    def buyer_chat(seller_id: str, request: Request):
        return _buyer_templates.TemplateResponse(request, "chat.html", {
            "seller_id": seller_id, "active_tab": "chat",
        })

    return app


app = create_app()
