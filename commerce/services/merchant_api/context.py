"""A composes exactly one B runtime and passes it to C in the same process."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from commerce.packages.contracts.ports import FLClientLifecycle, RecommenderRuntime, SellerJobExecutor
from commerce.packages.fl_client.lifecycle import FLClientConfig, create_client
from commerce.packages.recommender.runtime import open_runtime
from commerce.services.merchant_api.jobs import SellerJobs


@dataclass(frozen=True)
class MerchantSettings:
    seller_id: str
    feature_db_path: Path
    model_dir: Path
    fl: FLClientConfig = FLClientConfig()


@dataclass
class MerchantContext:
    runtime: RecommenderRuntime
    jobs: SellerJobs
    fl_client: FLClientLifecycle


def build_context(
    settings: MerchantSettings,
    *,
    runtime_factory: Callable[[str, Path, Path], RecommenderRuntime] = open_runtime,
    client_factory: Callable[[RecommenderRuntime, SellerJobExecutor, FLClientConfig], FLClientLifecycle] = create_client,
) -> MerchantContext:
    runtime = runtime_factory(settings.seller_id, settings.feature_db_path, settings.model_dir)
    jobs = SellerJobs()
    try:
        return MerchantContext(runtime, jobs, client_factory(runtime, jobs, settings.fl))
    except BaseException:
        jobs.close()
        raise
