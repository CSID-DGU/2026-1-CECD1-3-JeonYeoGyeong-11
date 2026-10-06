"""Seller FL client lifecycle. protected fails closed (g4); synthetic_plaintext runs a round loop.

synthetic_plaintext starts only with a coordinator URL, a seller token and a synthetic
input attestation (D0025). install_only (a seller outside every cohort, such as a new
seller) needs no attestation: it only installs released models and never submits. The loop polls the coordinator: it installs a newer release
first, then joins an open round of this seller's cohort after checking that the pinned
snapshot is the attested synthetic input. A failed check submits nothing. Every B call
goes through the injected seller jobs executor. Default config: disabled, protected.
"""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented, JobBusyError
from commerce.packages.contracts.ports import RecommenderRuntime, SellerJobExecutor
from commerce.packages.contracts.types import ModelVariant
from commerce.packages.fl_client import attestation, rounds
from commerce.packages.fl_client.transport import CoordinatorRefused, CoordinatorTransport


@dataclass(frozen=True)
class FLClientConfig:
    enabled: bool = False
    mode: str = "protected"
    model_variant: ModelVariant = "text_relation"
    coordinator_url: str | None = None
    token: str | None = None
    synthetic_attestation: Path | None = None
    install_only: bool = False  # receive releases, never join a round
    poll_seconds: float = 2.0

    def __post_init__(self):
        if self.mode not in ("protected", "synthetic_plaintext"):
            raise ValueError("Unknown FL mode")
        if self.model_variant not in ("text_only", "text_relation"):
            raise ValueError("Unknown model variant")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

    def __repr__(self):  # never print the token
        return "FLClientConfig(enabled=%r, mode=%r, model_variant=%r, install_only=%r)" % (
            self.enabled, self.mode, self.model_variant, self.install_only)


class FLClient:
    def __init__(self, runtime: RecommenderRuntime, jobs: SellerJobExecutor, config: FLClientConfig):
        self.runtime = runtime
        self.jobs = jobs
        self.config = config
        self.installed: str | None = None  # model_version this client last installed
        self.joined: set[str] = set()  # round_ids already handled, whatever the outcome
        self.last_outcome: str | None = None  # short, non-sensitive status for logs and tests
        self._task: asyncio.Task | None = None
        self._http: httpx.Client | None = None
        self._transport: CoordinatorTransport | None = None
        self._attestation: attestation.Attestation | None = None

    async def start(self) -> None:
        if not self.config.enabled:
            return
        if self.config.mode == "protected":
            raise FeatureNotImplemented("C: protected aggregation is not implemented (g4)")
        cfg = self.config
        if not (cfg.coordinator_url and cfg.token and (cfg.install_only or cfg.synthetic_attestation)):
            # OQ17/D0025: plaintext rounds run only on attested synthetic input.
            raise FeatureNotImplemented("C: synthetic_plaintext needs a coordinator, a token and an attestation")
        if not cfg.install_only:
            self._attestation = attestation.load_attestation(cfg.synthetic_attestation)
            if self._attestation.seller_id != self.runtime.seller_id:
                raise ValueError("the synthetic attestation is for another seller")
        self._http = httpx.Client(base_url=cfg.coordinator_url, timeout=30.0)
        self._transport = CoordinatorTransport(self._http, cfg.token)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._http is not None:
            self._http.close()
            self._http = None

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
            except (CoordinatorRefused, ContractError, JobBusyError, httpx.HTTPError, OSError) as exc:
                self.last_outcome = "error:%s" % type(exc).__name__  # retried on the next poll
            await asyncio.sleep(self.config.poll_seconds)

    def tick(self) -> None:
        """One poll: install a newer release, then join this seller's open round once."""
        transport, variant = self._transport, self.config.model_variant
        latest = transport.latest()
        if latest is not None and latest["model_version"] != self.installed:
            rounds.install_latest(self.runtime, self.jobs, transport, variant)
            self.installed = latest["model_version"]
            self.last_outcome = "installed:%s" % self.installed
        if self.config.install_only:
            return  # never trains, never submits
        config = transport.current_round()
        if config is None or config["round_id"] in self.joined:
            return
        self.joined.add(config["round_id"])  # one attempt per round, never a second submission
        try:
            ack = rounds.participate(self.runtime, self.jobs, transport, variant, config=config,
                                     check_input=self._check_input)
        except attestation.SyntheticInputRefused:
            self.last_outcome = "refused:not_attested_synthetic_input"
            return
        self.last_outcome = "round:%s" % (ack["disposition"] if ack else "none")

    def _check_input(self, local_data_ref: str) -> None:
        attestation.check_snapshot(self.runtime, local_data_ref, self._attestation)


def create_client(runtime: RecommenderRuntime, jobs: SellerJobExecutor, config: FLClientConfig) -> FLClient:
    return FLClient(runtime, jobs, config)
