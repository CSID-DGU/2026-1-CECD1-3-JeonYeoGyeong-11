"""Disabled lifecycle skeleton. There is no network submission implementation."""
from dataclasses import dataclass

from commerce.packages.contracts.errors import FeatureNotImplemented
from commerce.packages.contracts.ports import RecommenderRuntime, SellerJobExecutor
from commerce.packages.contracts.types import ModelVariant


@dataclass(frozen=True)
class FLClientConfig:
    enabled: bool = False
    mode: str = "protected"
    model_variant: ModelVariant = "text_relation"

    def __post_init__(self):
        if self.mode not in ("protected", "synthetic_plaintext"):
            raise ValueError("Unknown FL mode")
        if self.model_variant not in ("text_only", "text_relation"):
            raise ValueError("Unknown model variant")


class FLClient:
    def __init__(self, runtime: RecommenderRuntime, jobs: SellerJobExecutor, config: FLClientConfig):
        self.runtime = runtime
        self.jobs = jobs
        self.config = config

    async def start(self) -> None:
        if self.config.enabled:
            # Also refuse plaintext: input provenance checks are not implemented.
            raise FeatureNotImplemented("C: FL transport and protection are not implemented")

    async def stop(self) -> None:
        pass  # Disabled client owns no connection or background task.


def create_client(runtime: RecommenderRuntime, jobs: SellerJobExecutor, config: FLClientConfig) -> FLClient:
    return FLClient(runtime, jobs, config)
