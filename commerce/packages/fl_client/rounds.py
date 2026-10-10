"""One synthetic round and one release install, driven over B's runtime and the seller job executor.

Every B call runs on the injected jobs executor (one worker per seller), so training
never overlaps another seller task and never blocks a request loop. FLClient.start
runs these in its synthetic_plaintext loop, with the attested-input check of D0025
passed as `check_input`; c1 also drives them directly with a generated-tensor runtime.
"""
from __future__ import annotations

from typing import Callable

from commerce.packages.contracts.ports import RecommenderRuntime, SellerJobExecutor
from commerce.packages.contracts.types import ModelVariant, Payload
from commerce.packages.fl_client.submission import build_submission, check_round_matches, verify_release
from commerce.packages.fl_client.transport import CoordinatorTransport


def participate(runtime: RecommenderRuntime, jobs: SellerJobExecutor, transport: CoordinatorTransport,
                model_variant: ModelVariant, *, config: Payload | None = None,
                check_input: Callable[[str], None] | None = None) -> Payload | None:
    """Train on the open round if this seller is in its cohort; return the ack, or None if there is none.

    `config` is the round already fetched by the caller (else it is fetched here).
    `check_input(local_data_ref)` runs before training and raises to stop the round for
    this seller; the same pinned snapshot is then the one trained on.
    """
    if config is None:
        config = transport.current_round()
    if config is None:
        return None
    manifest = jobs.submit(lambda: runtime.get_shared_manifest(model_variant=model_variant)).result()
    check_round_matches(config, manifest)  # a different variant or base: do not train
    local_data_ref = jobs.submit(runtime.get_local_data_ref).result()
    if check_input is not None:
        jobs.submit(lambda: check_input(local_data_ref)).result()
    result = jobs.submit(lambda: runtime.train_round(local_data_ref, config, model_variant=model_variant)).result()
    submission, payload = build_submission(runtime.seller_id, manifest, config, result)
    return transport.submit(config["round_id"], submission, payload)


def install_latest(runtime: RecommenderRuntime, jobs: SellerJobExecutor, transport: CoordinatorTransport,
                   model_variant: ModelVariant) -> Payload | None:
    """latest -> manifest and weights verified -> B install_release. Works without any round history."""
    descriptor = transport.latest()
    if descriptor is None:
        return None
    version = descriptor["model_version"]
    manifest = transport.manifest(version)
    tensors = verify_release(descriptor, manifest, transport.weights(version))
    jobs.submit(lambda: runtime.install_release(descriptor, manifest, tensors, model_variant=model_variant)).result()
    return descriptor
