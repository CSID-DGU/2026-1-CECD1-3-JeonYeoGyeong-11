"""Seller-started personalization (OQ08 agreement on #31: the A screen starts it).

One button runs B's personalize_local for both variants on the seller's single
background job executor (context.jobs, shared with FL rounds), so it never
blocks a request and never overlaps a training round (JobBusyError). The last
run's outcome is kept in memory per seller for the comparison screen; the
durable result -- an installed personalization revision -- lives on B's side
and shows up in compare_local's T-P/R-P arms.
"""
from __future__ import annotations

import datetime as dt
import threading
from typing import Any

from commerce.packages.contracts.errors import FeatureNotImplemented, JobBusyError

VARIANTS = ("text_only", "text_relation")
_lock = threading.Lock()
_last: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(context) -> None:
    results: dict[str, Any] = {}
    error = None
    try:
        ref = context.runtime.get_local_data_ref()  # one snapshot for both variants
        for variant in VARIANTS:
            r = context.runtime.personalize_local(ref, {}, model_variant=variant)
            results[variant] = {"status": r.status, "reason": r.reason,
                                "base_model_version": r.base_model_version,
                                "revision": r.personalization_revision}
    except FeatureNotImplemented:
        error = "not_connected"
    except Exception as exc:  # shown on the screen; never crashes the job thread
        error = "%s: %s" % (type(exc).__name__, exc)
    with _lock:
        _last[context.seller_id] = {"state": "error" if error else "done", "error": error,
                                    "results": results, "finished_at": _now()}


def start(context) -> str:
    """'started', or 'busy' when a personalization or FL round is already running."""
    with _lock:
        if _last.get(context.seller_id, {}).get("state") == "running":
            return "busy"
        _last[context.seller_id] = {"state": "running", "started_at": _now(), "results": {}, "error": None}
    try:
        context.jobs.submit(lambda: _run(context))
    except JobBusyError:
        with _lock:
            _last.pop(context.seller_id, None)
        return "busy"
    return "started"


def last_run(seller_id: str) -> dict[str, Any] | None:
    with _lock:
        run = _last.get(seller_id)
        return dict(run) if run else None
