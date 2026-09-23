"""Merchant app shell: health and lifecycle only; no order/recommendation routes."""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI

from commerce.packages.fl_client.lifecycle import FLClientConfig
from commerce.services.merchant_api.context import MerchantContext, MerchantSettings, build_context


def settings_from_env() -> MerchantSettings:
    enabled = os.environ.get("FL_ENABLED", "false").lower()
    if enabled not in ("true", "false"):
        raise ValueError("FL_ENABLED must be true or false")
    return MerchantSettings(
        seller_id=os.environ["MERCHANT_ID"],
        feature_db_path=Path(os.environ["FEATURE_DB_PATH"]),
        model_dir=Path(os.environ["MODEL_DIR"]),
        fl=FLClientConfig(enabled=enabled == "true", mode=os.environ.get("FL_MODE", "protected"),
                          model_variant=os.environ.get("FL_MODEL_VARIANT", "text_relation")),
    )


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

    @app.get("/healthz")
    def healthz():
        return {"service": "merchant", "status": "scaffold", "ready": False}

    return app


app = create_app()
