"""C: coordinator HTTP app for synthetic rounds. Health only until it is configured.

Routes and status codes are documented in this service's README; the invariants are
interfaces.md §6 and §8. Without REGISTRY_DIR/AUTH_FILE/ROUND_STATE_DIR the app keeps
the scaffold behaviour (health only). FL_MODE=protected fails closed at start (g4).

Interim, pending OQ13: contract_error.v1 has no authentication code, so a missing or
invalid token gets 401 with `WWW-Authenticate: Bearer` and an empty body. Oversize
transfers get 413 with a SCHEMA_INVALID body, the closest code in the v1 enum.
Request bodies, tokens and tensors are never logged or echoed in an error.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from commerce.packages.contracts.errors import ContractError
from commerce.services.fl_coordinator.auth import SellerAuth
from commerce.services.fl_coordinator.npz_payload import MAX_PAYLOAD_BYTES, PayloadTooLarge
from commerce.services.fl_coordinator.round_core import SAFE_ID, ModelRegistry
from commerce.services.fl_coordinator.service import MODES, Coordinator, RoundLedger

MAX_ENVELOPE_BYTES = 1024 * 1024
_STATUS = {
    "FORBIDDEN": 403, "NOT_FOUND": 404,
    "SCHEMA_INVALID": 422, "UNKNOWN_FIELD": 422, "MISSING_REQUIRED_FIELD": 422, "INVALID_TYPE": 422,
    "INVALID_ENUM_VALUE": 422, "VERSION_MISMATCH": 422,
}  # every other code is a state, duplicate or manifest conflict: 409


@dataclass(frozen=True)
class CoordinatorSettings:
    registry_dir: Path
    auth_file: Path
    round_state_dir: Path
    mode: str = "protected"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("Unknown FL mode")
        if self.registry_dir.exists() and not self.registry_dir.is_dir():
            raise ValueError("REGISTRY_DIR must be a directory")
        if self.round_state_dir.exists() and not self.round_state_dir.is_dir():
            raise ValueError("ROUND_STATE_DIR must be a directory")
        if self.auth_file.is_dir():
            raise ValueError("AUTH_FILE must be a file")


def settings_from_env() -> CoordinatorSettings | None:
    registry_dir = os.environ.get("REGISTRY_DIR")
    auth_file = os.environ.get("AUTH_FILE")
    round_state_dir = os.environ.get("ROUND_STATE_DIR")
    if not (registry_dir or auth_file or round_state_dir):
        return None
    if not (registry_dir and auth_file and round_state_dir):
        raise ValueError("REGISTRY_DIR, AUTH_FILE and ROUND_STATE_DIR are set together")
    return CoordinatorSettings(Path(registry_dir), Path(auth_file), Path(round_state_dir),
                               os.environ.get("FL_MODE", "protected"))


class _TooLarge(Exception):
    pass


def _error(code: str, status: int | None = None) -> JSONResponse:
    return JSONResponse(ContractError(code).to_payload(), status_code=status or _STATUS.get(code, 409))


def _reject_constant(_token: str):
    raise ValueError("NaN and Infinity are not JSON")


async def _read_capped(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _TooLarge()
    chunks, total = [], 0
    async for chunk in request.stream():  # the limit applies while receiving, not only at the end
        total += len(chunk)
        if total > limit:
            raise _TooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ContractError("NOT_FOUND")
    return value


def create_app(settings: CoordinatorSettings | None = None, *, now=None) -> FastAPI:
    app = FastAPI(title="FL coordinator", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def _validation(_request, _exc):  # never echo the input
        return _error("SCHEMA_INVALID")

    @app.exception_handler(StarletteHTTPException)
    async def _no_route(_request, exc: StarletteHTTPException):  # unknown path or method: same error shape
        return _error("NOT_FOUND", exc.status_code)

    @app.exception_handler(ContractError)
    async def _contract(_request, exc: ContractError):
        return _error(exc.code)

    @app.exception_handler(PayloadTooLarge)
    @app.exception_handler(_TooLarge)
    async def _too_large(_request, _exc):
        return _error("SCHEMA_INVALID", 413)

    if settings is None:
        @app.get("/healthz")
        def healthz():
            return {"service": "coordinator", "status": "unconfigured", "ready": False}
        return app

    kwargs = {} if now is None else {"now": now}
    coordinator = Coordinator(ModelRegistry(settings.registry_dir), RoundLedger(settings.round_state_dir),
                              mode=settings.mode, **kwargs)
    sellers = SellerAuth(settings.auth_file)
    app.state.coordinator = coordinator  # the operator side (open_round) is in-process until OQ08

    @app.middleware("http")
    async def _authenticate(request: Request, call_next):
        if request.url.path != "/healthz":
            seller = sellers.seller_for(request.headers.get("authorization"))
            if seller is None:
                return Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})
            request.state.seller = seller
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {"service": "coordinator", "status": settings.mode, "ready": coordinator.registry.latest() is not None}

    @app.get("/rounds/current")
    async def current_round(request: Request):
        config = coordinator.current_config(request.state.seller)
        return Response(status_code=204) if config is None else JSONResponse(config)

    @app.get("/models/latest")
    async def latest_model():
        descriptor = coordinator.registry.latest()
        if descriptor is None:
            raise ContractError("NOT_FOUND")
        return JSONResponse(descriptor)

    @app.get("/models/{model_version}/manifest")
    async def model_manifest(model_version: str):
        return JSONResponse(coordinator.registry.manifest(_safe_id(model_version)))

    @app.get("/models/{model_version}/weights")
    async def model_weights(model_version: str):
        return Response(coordinator.registry.weights(_safe_id(model_version)), media_type="application/octet-stream")

    @app.post("/rounds/{round_id}/submissions")
    async def declare(round_id: str, request: Request):
        body = await _read_capped(request, MAX_ENVELOPE_BYTES)
        try:
            envelope = json.loads(body, parse_constant=_reject_constant)
        except ValueError:
            raise ContractError("SCHEMA_INVALID") from None
        coordinator.declare(request.state.seller, _safe_id(round_id), envelope)
        return Response(status_code=201, headers={"Location": "/rounds/%s/submissions/delta" % round_id})

    @app.put("/rounds/{round_id}/submissions/delta")
    async def upload(round_id: str, request: Request):
        payload = await _read_capped(request, MAX_PAYLOAD_BYTES)
        return JSONResponse(coordinator.upload(request.state.seller, _safe_id(round_id), payload), status_code=202)

    @app.get("/rounds/{round_id}/result")
    async def result(round_id: str, request: Request):
        return JSONResponse(coordinator.result(request.state.seller, _safe_id(round_id)))

    return app


app = create_app(settings_from_env())
