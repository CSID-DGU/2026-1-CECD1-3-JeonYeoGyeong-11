"""A composes exactly one B runtime and passes it to C in the same process."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from commerce.packages.contracts.ports import FLClientLifecycle, RecommenderRuntime, SellerJobExecutor
from commerce.packages.fl_client.lifecycle import FLClientConfig, create_client
from commerce.packages.recommender.runtime import open_runtime
from commerce.services.merchant_api.jobs import SellerJobs


def merchant_db_path_from_env() -> Path:
    """Where this seller's orders.sqlite is, by the same rule for the app and its tools.

    MERCHANT_DB_PATH when set; otherwise orders.sqlite next to FEATURE_DB_PATH,
    since run_local.py (C-owned) only passes FEATURE_DB_PATH so far
    (working-agreement.md §3). KeyError when neither is set. Reads os.environ
    directly so the docs gate sees which variables are read.
    """
    configured = os.environ.get("MERCHANT_DB_PATH")
    if configured:
        return Path(configured)
    return Path(os.environ["FEATURE_DB_PATH"]).parent / "orders.sqlite"


@dataclass(frozen=True)
class MerchantSettings:
    seller_id: str
    feature_db_path: Path
    model_dir: Path
    # orders.sqlite is A-only (working-agreement.md §3 storage table). Defaults
    # next to feature_db_path so tests/scaffold that omit it still get a private
    # per-settings path instead of colliding on a shared relative filename.
    merchant_db_path: Path = None  # type: ignore[assignment]
    fl: FLClientConfig = FLClientConfig()

    def __post_init__(self):
        if self.merchant_db_path is None:
            object.__setattr__(self, "merchant_db_path", self.feature_db_path.parent / "orders.sqlite")


@dataclass
class MerchantContext:
    runtime: RecommenderRuntime
    jobs: SellerJobs
    fl_client: FLClientLifecycle
    merchant_db_path: Path = None  # type: ignore[assignment]
    seller_id: str = None  # type: ignore[assignment]
    model_dir: Path = None  # type: ignore[assignment]  # read-only here: the dashboard lists installed models


def build_context(
    settings: MerchantSettings,
    *,
    runtime_factory: Callable[[str, Path, Path], RecommenderRuntime] = open_runtime,
    client_factory: Callable[[RecommenderRuntime, SellerJobExecutor, FLClientConfig], FLClientLifecycle] = create_client,
) -> MerchantContext:
    runtime = runtime_factory(settings.seller_id, settings.feature_db_path, settings.model_dir)
    jobs = SellerJobs()
    try:
        return MerchantContext(
            runtime, jobs, client_factory(runtime, jobs, settings.fl),
            settings.merchant_db_path, settings.seller_id, settings.model_dir,
        )
    except BaseException:
        jobs.close()
        raise
