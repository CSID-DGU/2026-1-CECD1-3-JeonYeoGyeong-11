"""B's seller runtime, the real RecommenderRuntime (model.md §4·§5, interfaces.md §2–§5).

One object per seller. A's request threads call ingest/upsert/predict; the
seller's job thread calls train_round and install_release through C's client.

- Features: feature_store.FeatureStore (features.sqlite) is the only ledger. The
  frozen text vectors z live in the same file (z_cache).
- Models: models/base/{variant}/{model_version}/ holds release.json, manifest.json,
  weights.npz (B's canonical copy) and local.json (that copy's hash);
  models/base/{variant}/CURRENT names the serving base. A release
  is written to a temporary directory, renamed into place, and only then made
  CURRENT and swapped in memory, so a failure leaves the old base serving.
- Serving: a request pins one snapshot and one model handle when it starts.
  Fallback order and ranking follow interfaces.md §4.
- Training: train_round copies the base named by round_config, reads the snapshot
  local_data_ref pinned, and returns the shared delta. One customer in ten is held
  out for the validation loss, chosen by seller and run seed, never by round_id.
  The training draws (batch order, negatives) come from seller, run seed and
  round_id, so a run that keeps one seed still samples anew every round.
- Personalization (D0019, model.md §8): a copy of the serving base learns only
  query_proj and scorer and is kept only if its validation loss beats the base's.
  It lives in models/personal/{variant}/{base_version}/{revision}/ and is used only
  with that base; a new base serves without it until personalized again.
- compare_local pins one snapshot, one candidate set and both variants' handles,
  then fills T-G, R-G, T-P, R-P or says why an arm is unavailable. The handles are
  the comparison pair pinned in models/comparison.json (pin_comparison, comparison.md
  §5) or, for a variant without a pin, the serving base.
- Warm-up (warm=True, which open_runtime sets): a background thread computes the
  catalog's z, the whole-ledger relations and each installed base's e for the
  current epoch, after opening and after every new event, catalog version or
  release (pokes within WARM_DEBOUNCE seconds are coalesced). A request that
  arrives meanwhile waits for that computation instead of repeating it, so the
  first recommendation does not pay for encoding the whole catalog.

Instacart-style relative-time baskets carry no calendar: the whole relative
ledger counts as earlier than any live as_of. Absolute-time baskets count only
before as_of (completed_at < as_of).
"""
from collections import Counter
from contextlib import closing, contextmanager
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Iterator, Mapping, Sequence
import uuid

import numpy as np
import torch

from commerce.packages.contracts.errors import ContractError, FeatureNotImplemented, JobBusyError
from commerce.packages.contracts.ids import canonical_json, manifest_hash
from commerce.packages.contracts.types import (
    ComparisonArm, ComparisonResult, ModelVariant, Payload, PersonalizationResult, ServingMode, TensorMap, TrainingResult,
)
from commerce.packages.data_adapters.baskets import LocalBasket, Visit, customer_visits
from commerce.packages.data_adapters.text import EmptyProductText, catalog_item_text
from commerce.packages.data_adapters.validation import check_payload
from commerce.packages.recommender.examples import customer_examples, query_example
from commerce.packages.recommender.feature_store import FeatureStore, Snapshot
from commerce.packages.recommender.harex import HarexConfig, HarexRecommender
from commerce.packages.recommender.relations import RelationTensors, build_relations
from commerce.packages.recommender.replay import BUCKETS, progress_bucket, visible_prefix
from commerce.packages.recommender.serving import (
    PERSONAL_GROUPS, SERVICE_ARCHITECTURES, VARIANT_OF_ARCHITECTURE, FrozenText, ServiceArchitecture,
    architecture_for,
    build_manifest, build_model, canonical_npz, check_tensors, load_shared, read_npz, read_tensors, shared_tensors,
)
from commerce.packages.recommender.training import SellerData, TrainConfig, train, validation_loss
from commerce.packages.recommender.z_cache import ZCache, preprocessing_version

VARIANTS: tuple[ModelVariant, ...] = ("text_only", "text_relation")
WARM_DEBOUNCE = 0.3  # seconds without a new change before the warm-up runs
DEFAULT_TOP_N = 10
FALLBACK_MODEL_VERSION = "popularity.local"  # recommendation.model_version when no base is installed
VALIDATION_SHARE = 10  # one customer in ten (model.md §6)
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
# Personalization settings, the same for both variants (model.md §8.1). lr and steps were
# chosen on validation loss at held-out sellers (evaluation/personal_sweep.py, model.md §8).
# Keys of personal_config override them; any other key is an error. "seed" drives the training
# randomness only: the validation customers come from PERSONAL_SPLIT_SEED, fixed per
# seller, so no config can pick the customers its own result is judged on.
PERSONAL_SPLIT_SEED = 0
DEFAULT_PERSONAL = {"steps": 40, "batch_size": 64, "lr": 3e-4, "n_neg": 200, "seed": 0,
                    "min_train_examples": 20, "min_val_examples": 5}
ARMS = (("T-G", "text_only", "global"), ("R-G", "text_relation", "global"),
        ("T-P", "text_only", "personalized"), ("R-P", "text_relation", "personalized"))
REASON_OF_ATTEMPT = {"insufficient_data": "insufficient_data", "validation_rejected": "validation_rejected",
                     "base_mismatch": "base_mismatch"}


@dataclass(frozen=True)
class ModelHandle:
    variant: ModelVariant
    model_version: str
    manifest: Payload
    release: Payload
    model: HarexRecommender  # eval mode; never trained in place
    tensors: TensorMap


@dataclass(frozen=True)
class PersonalHandle:
    variant: ModelVariant
    base_model_version: str
    revision: str
    serving_version: str  # recommendation.model_version while this personalization serves
    model: HarexRecommender  # the base with the seller's own query_proj and scorer
    tensors: TensorMap  # personal.* tensors of the two groups only


@dataclass(frozen=True)
class Catalog:
    epoch: int
    items: tuple[str, ...]  # items with a product text, sorted; row i of z
    z: torch.Tensor
    active: frozenset[str]  # listing_status active, with or without a text


@dataclass
class Prepared:
    """Whole-ledger inputs at one epoch: every basket counts (none is at or after as_of)."""
    epoch: int
    catalog: Catalog
    visits: dict[str, list[Visit]]
    relations: RelationTensors | None = None  # built on the first text_relation use


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        raise ContractError("SCHEMA_INVALID", "/as_of") from None


def _before(basket: LocalBasket, as_of: datetime) -> bool:
    return basket.time_kind != "absolute" or basket.time_value < as_of


def _visits(baskets: Sequence[LocalBasket], known: set[str]) -> dict[str, list[Visit]]:
    """Each customer's ordered visits holding only items with a product text."""
    by_customer: dict[str, list[LocalBasket]] = {}
    for basket in baskets:
        kept = tuple(i for i in basket.items if i.item_id_local in known)
        by_customer.setdefault(basket.customer_id_local, []).append(dataclasses.replace(basket, items=kept))
    out = {}
    for customer, own in by_customer.items():
        try:
            out[customer] = customer_visits(own)
        except ValueError:
            continue  # a relative history with a missing order cannot be ordered; leave it out
    return out


def _time_cuts(visits: Mapping[str, Sequence[Visit]]) -> list:
    """Bucket cut times for absolute-time customers: cuts[0] shows nothing, cuts[b] the b-th decile."""
    times = sorted(v.basket.time_value for vs in visits.values() for v in vs if v.basket.time_kind == "absolute")
    if not times:
        return [None] * BUCKETS
    return [None] + [times[b * len(times) // BUCKETS] for b in range(1, BUCKETS)]


def _bucket(visits: Sequence[Visit], position: int, cuts: list) -> int:
    """The latest snapshot that cannot see the target visit (replay.py for relative time)."""
    if visits[0].basket.time_kind != "absolute":
        return progress_bucket(position, len(visits))
    target = visits[position - 1].basket.time_value
    return max([0] + [b for b in range(1, BUCKETS) if cuts[b] is not None and cuts[b] < target])


def _visible(visits: Mapping[str, Sequence[Visit]], bucket: int, cuts: list) -> dict[str, list[Visit]]:
    out = {}
    for customer, vs in visits.items():
        if vs[0].basket.time_kind != "absolute":
            out[customer] = list(vs[:visible_prefix(bucket, len(vs))])
        else:
            out[customer] = [v for v in vs if cuts[bucket] is not None and v.basket.time_value < cuts[bucket]]
    return out


def planned_steps(n_train: int, round_config: Payload) -> int:
    """model.md §6: min(local_steps, ceil(n_train / batch_size) x max_local_epochs); 0 epochs = no cap."""
    local_steps, epochs = round_config["local_steps"], round_config["max_local_epochs"]
    if local_steps == 0 or n_train == 0:
        return 0
    if epochs == 0:
        return local_steps
    return min(local_steps, math.ceil(n_train / round_config["batch_size"]) * epochs)


def personal_names(model: HarexRecommender) -> dict[str, str]:
    """state_dict key -> local name of the two personalized groups, in parameter order."""
    return {key: "personal." + key.replace(".", "_") for key in model.state_dict()
            if key.split(".")[0] in PERSONAL_GROUPS}


def is_validation_customer(seller_id: str, run_seed: int, customer: str) -> bool:
    digest = _sha256(canonical_json(["val_split", seller_id, run_seed, customer]))
    return int(digest[:8], 16) % VALIDATION_SHARE == 0


def round_train_seed(seller_id: str, run_seed: int, round_id: str) -> int:
    """Seed of one round's training draws. The lab varied them per round and seller
    (fl_lab: seed * 100000 + round * 1000 + seller); a coordinator that keeps one seed
    for the run (it must, for the validation split) would otherwise replay the same
    batches and negatives every round."""
    return int(_sha256(canonical_json(["train", seller_id, run_seed, round_id]))[:8], 16)


class SellerRuntime:
    def __init__(self, seller_id: str, feature_db_path: str | Path, model_dir: str | Path, *,
                 text=None, encoder_dir: str | Path | None = None,
                 architectures: Mapping[int, HarexConfig] = SERVICE_ARCHITECTURES,
                 variants: Mapping[int, str] = VARIANT_OF_ARCHITECTURE, warm: bool = False,
                 clock=lambda: datetime.now(timezone.utc)):
        if not seller_id:
            raise ValueError("seller_id is required")
        self._seller_id = seller_id
        self.store = FeatureStore(feature_db_path, seller_id)
        self.model_dir = Path(model_dir)
        # The frozen encoder is part of the B install (model-lab.md §6.6), one copy per seller.
        self.encoder_dir = Path(encoder_dir) if encoder_dir is not None else self.model_dir / "frozen_text"
        self._text = text
        self._architectures = architectures
        self._variants = variants
        self._state = threading.RLock()  # serving handles and caches
        self._job = threading.Lock()  # one training or personalization at a time
        self._handles: dict[str, ModelHandle | None] = {}
        self._personal: dict[str, PersonalHandle | None] = {}
        self.load_errors: dict[str, str] = {}  # variant -> why its CURRENT base did not load
        self._manifests: dict[str, Payload] = {}
        self._catalog_cache: Catalog | None = None
        self._prepared: Prepared | None = None
        self._e_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._pinned: dict[tuple[str, str], ModelHandle] = {}  # comparison bases other than the serving one
        self._compute = threading.RLock()  # one z / relation / e computation at a time
        self._clock = clock  # the warm-up's "now": live requests use as_of = now
        for variant in VARIANTS:
            self._handles[variant] = self._load_current(variant)
            self._personal[variant] = self._load_personal(variant, self._handles[variant])
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._closing = False
        self._warmer = None
        if warm:
            self._warmer = threading.Thread(target=self._warm_loop, name="b-warm-%s" % seller_id, daemon=True)
            self._warmer.start()
            self._poke()

    @property
    def seller_id(self) -> str:
        return self._seller_id

    # ------------------------------------------------------------------ shared pieces

    def _arch(self, variant: ModelVariant) -> ServiceArchitecture:
        return architecture_for(variant, self._architectures, self._variants)

    def _frozen_text(self):
        if self._text is None:
            self._text = FrozenText(self.encoder_dir)
        return self._text

    def _base_dir(self, variant: str) -> Path:
        return self.model_dir / "base" / variant

    def _catalog(self, snap: Snapshot) -> Catalog:
        with self._state:
            if self._catalog_cache is not None and self._catalog_cache.epoch == snap.epoch:
                return self._catalog_cache
        with self._compute:
            return self._encode_catalog(snap)

    def _encode_catalog(self, snap: Snapshot) -> Catalog:
        with self._state:
            if self._catalog_cache is not None and self._catalog_cache.epoch == snap.epoch:
                return self._catalog_cache  # another thread finished it while this one waited
        texts = {}
        for item_id, item in sorted(snap.catalog.items()):
            try:
                texts[item_id] = catalog_item_text(item)
            except (EmptyProductText, FeatureNotImplemented):
                continue  # no text to encode: the item cannot be scored, only listed by fallback
        text = self._frozen_text()
        items = tuple(texts)
        with closing(ZCache(self.store.path, text, None, text_artifact_hash=text.text_artifact_hash)) as cache:
            z = torch.from_numpy(np.array(cache.vectors([texts[i] for i in items]), dtype=np.float32))
        active = frozenset(i for i, item in snap.catalog.items() if item["listing_status"] == "active")
        catalog = Catalog(snap.epoch, items, z, active)
        with self._state:
            if self._catalog_cache is None or self._catalog_cache.epoch <= snap.epoch:
                self._catalog_cache = catalog
        return catalog

    def _ref_tag(self, epoch: int) -> str:
        return _sha256(canonical_json(["local_data_ref", self.seller_id, epoch]))[:16]

    def _parse_ref(self, local_data_ref: str) -> int:
        parts = local_data_ref.split(".") if isinstance(local_data_ref, str) else []
        if len(parts) != 3 or parts[0] != "fs" or not parts[1].isdigit() or parts[2] != self._ref_tag(int(parts[1])):
            raise ContractError("FORBIDDEN")
        epoch = int(parts[1])
        if epoch > self.store.feature_epoch:
            raise ContractError("NOT_FOUND")
        return epoch

    # ------------------------------------------------------------------ A: ledger

    def ingest_purchase_event(self, event: Payload) -> None:
        if self.store.ingest_event(event):
            self._poke()

    def upsert_catalog_item(self, item: Payload, source_seq: int) -> None:
        if self.store.upsert_item(item, source_seq):
            self._poke()

    # ------------------------------------------------------------------ models on disk

    def get_shared_manifest(self, *, model_variant: ModelVariant = "text_relation") -> Payload:
        arch = self._arch(model_variant)
        with self._state:
            if model_variant not in self._manifests:
                self._manifests[model_variant] = build_manifest(
                    arch, self._frozen_text().text_artifact_hash, preprocessing_version())
            return json.loads(json.dumps(self._manifests[model_variant]))

    def _check_manifest(self, variant: ModelVariant, manifest: Payload, release: Payload) -> None:
        if manifest_hash(manifest) != manifest["manifest_hash"] or release["manifest_hash"] != manifest["manifest_hash"]:
            raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
        # Equal hashes mean the same architecture, tensor names and shapes, text artifact and
        # preprocessing as this B package expects for the variant.
        if manifest["manifest_hash"] != self.get_shared_manifest(model_variant=variant)["manifest_hash"]:
            raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")

    def _read_base(self, variant: ModelVariant, version: str) -> ModelHandle:
        folder = self._base_dir(variant) / version
        release = json.loads((folder / "release.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        local = json.loads((folder / "local.json").read_text(encoding="utf-8"))
        data = (folder / "weights.npz").read_bytes()
        # The file is B's own canonical copy; local.json holds its hash (release.weights_sha256
        # names the bytes C served, which B never sees).
        if release.get("model_version") != version or _sha256(data) != local["weights_sha256"]:
            raise ContractError("MANIFEST_MISMATCH", "/weights_sha256")
        self._check_manifest(variant, manifest, release)
        tensors = read_npz(data, manifest)
        model = build_model(self._arch(variant).config)
        load_shared(model, tensors)
        return ModelHandle(variant, version, manifest, release, model, tensors)

    def _load_current(self, variant: ModelVariant) -> ModelHandle | None:
        pointer = self._base_dir(variant) / "CURRENT"
        if not pointer.is_file():
            return None
        try:
            return self._read_base(variant, pointer.read_text(encoding="utf-8").strip())
        except Exception as error:  # keep serving with fallback; the reason stays local
            self.load_errors[variant] = "%s: %s" % (type(error).__name__, error)
            return None

    def install_release(self, release: Payload, manifest: Payload, tensors: TensorMap, *,
                        model_variant: ModelVariant = "text_relation") -> None:
        """Install decoded release tensors. C's fl_client (verify_release) has already checked the
        downloaded bytes against release.weights_sha256; B checks the descriptor and manifest
        against its own package and every tensor's name, shape, dtype and finiteness."""
        check_payload("model_release.v1", release)
        check_payload("shared_model_manifest.v1", manifest)
        arch = self._arch(model_variant)
        self._check_manifest(model_variant, manifest, release)
        tensors = {name: np.asarray(array) for name, array in tensors.items()}
        check_tensors(manifest, tensors)
        data = canonical_npz(tensors, [t["name"] for t in manifest["tensors"]])
        version = release["model_version"]
        base_dir = self._base_dir(model_variant)
        final = base_dir / version
        with self._state:
            current = self._handles.get(model_variant)
            if final.exists():
                stored = json.loads((final / "release.json").read_text(encoding="utf-8"))
                if canonical_json(stored) != canonical_json(release):
                    raise ContractError("MANIFEST_MISMATCH", "/model_version")  # same version, other bytes
                if current is not None and current.model_version == version:
                    return  # a repeated delivery: nothing changes, personalization stays valid
                handle = self._read_base(model_variant, version)
            else:
                model = build_model(arch.config)
                load_shared(model, tensors)
                staging = base_dir / (".incoming-%s" % uuid.uuid4().hex)
                staging.mkdir(parents=True)
                try:
                    (staging / "weights.npz").write_bytes(data)
                    (staging / "manifest.json").write_bytes(canonical_json(manifest))
                    (staging / "release.json").write_bytes(canonical_json(release))
                    (staging / "local.json").write_bytes(canonical_json({"weights_sha256": _sha256(data)}))
                    os.replace(staging, final)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
                handle = ModelHandle(model_variant, version, manifest, release, model, tensors)
            pointer = base_dir / ("CURRENT.%s" % uuid.uuid4().hex)
            pointer.write_text(version, encoding="utf-8")
            os.replace(pointer, base_dir / "CURRENT")
            self._handles[model_variant] = handle
            self._personal[model_variant] = self._load_personal(model_variant, handle)
            self.load_errors.pop(model_variant, None)
            self._e_cache = {k: v for k, v in self._e_cache.items() if k[0] != model_variant}
        self._poke()

    def export_shared_state(self, *, model_variant: ModelVariant = "text_relation") -> TensorMap:
        self._arch(model_variant)
        with self._state:
            handle = self._handles.get(model_variant)
        if handle is None:
            raise ContractError("NOT_FOUND")
        return {name: array.copy() for name, array in handle.tensors.items()}

    # ------------------------------------------------------------------ A: recommendation

    def _candidates(self, request: Payload, snap: Snapshot) -> list[str]:
        active = {i for i, item in snap.catalog.items() if item["listing_status"] == "active"}
        ids = request["candidate_item_ids"]
        if ids is None:
            return sorted(active)
        for index, item_id in enumerate(ids):
            if item_id not in snap.catalog:
                raise ContractError("NOT_FOUND", "/candidate_item_ids/%d" % index)
        return [i for i in ids if i in active]

    def _prepare(self, snap: Snapshot, need_relations: bool) -> Prepared:
        """The whole-ledger inputs of snap's epoch, computed once and shared by requests and warm-up."""
        with self._state:
            prepared = self._prepared
        if prepared is None or prepared.epoch != snap.epoch:
            with self._compute:
                with self._state:
                    prepared = self._prepared
                if prepared is None or prepared.epoch != snap.epoch:
                    catalog = self._catalog(snap)
                    prepared = Prepared(snap.epoch, catalog, _visits(snap.baskets, set(catalog.items)))
                    with self._state:
                        if self._prepared is None or self._prepared.epoch <= snap.epoch:
                            self._prepared = prepared
        if need_relations and prepared.relations is None:
            with self._compute:
                if prepared.relations is None:
                    row_of = {item: row for row, item in enumerate(prepared.catalog.items)}
                    prepared.relations = build_relations(prepared.visits, row_of, len(prepared.catalog.items))
        return prepared

    @torch.no_grad()
    def _item_vectors(self, arch: ServiceArchitecture, handle: ModelHandle, epoch: int, catalog: Catalog,
                      relations: RelationTensors | None, whole: bool) -> torch.Tensor:
        """e of every catalog item under the base; cached per variant, base and epoch for whole ledgers."""
        key = (arch.variant, handle.model_version, epoch)
        data = SellerData(self.seller_id, catalog.items, catalog.z, [], relations)
        if not whole:
            return handle.model.encode_items(data)
        with self._state:
            e = self._e_cache.get(key)
        if e is not None:
            return e
        with self._compute:
            with self._state:
                e = self._e_cache.get(key)
            if e is None:
                e = handle.model.encode_items(data)  # personalization never changes e (model.md §7)
                with self._state:
                    # Per variant: this epoch's e of at most two bases, the serving one and a pinned
                    # comparison base, so compare_local next to predict_local does not recompute both.
                    kept = {k: v for k, v in self._e_cache.items() if k[0] != arch.variant or k[2] == epoch}
                    same = [k for k in kept if k[0] == arch.variant]
                    for k in same[:-1]:
                        del kept[k]
                    kept[key] = e
                    self._e_cache = kept
        return e

    @torch.no_grad()
    def _model_scores(self, arch: ServiceArchitecture, handle: ModelHandle, model: HarexRecommender,
                      snap: Snapshot, before: Sequence[LocalBasket], customer: str) -> dict[str, float] | None:
        """Scores over the catalog for one customer, or None when the customer has nothing to read."""
        whole = len(before) == len(snap.baskets)
        if whole:
            prepared = self._prepare(snap, arch.config.relation)
            catalog, visits, relations = prepared.catalog, prepared.visits, prepared.relations
        else:
            catalog = self._catalog(snap)
            visits = _visits(before, set(catalog.items))
            relations = None
            if arch.config.relation:
                row_of = {item: row for row, item in enumerate(catalog.items)}
                relations = build_relations(visits, row_of, len(catalog.items))
        own = visits.get(customer)
        if not own:
            return None
        example = query_example(own)
        if not any(v.items for v in example.history):
            return None
        e = self._item_vectors(arch, handle, snap.epoch, catalog, relations, whole)
        data = SellerData(self.seller_id, catalog.items, catalog.z, [example], relations)
        q = model.encode_queries(e, [example], data)
        scores = model.score(q, e)[0]
        if not torch.isfinite(scores).all():
            raise RuntimeError("the model produced a non-finite score")
        return dict(zip(catalog.items, scores.tolist()))

    def _serving(self, variant: ModelVariant, mode: ServingMode) -> tuple[ModelHandle | None, HarexRecommender | None, str | None]:
        """(base handle, model to query with, serving model_version) pinned for one request."""
        if mode not in ("global", "personalized", "auto"):
            raise ValueError("mode must be global, personalized or auto")
        with self._state:
            handle = self._handles.get(variant)
            personal = self._personal.get(variant)
        if personal is not None and (handle is None or personal.base_model_version != handle.model_version):
            personal = None  # never attached to another base (model.md §8.4)
        if mode == "personalized":
            if personal is None:
                raise ContractError("NOT_FOUND")  # model.md §4: no approved personalization for this base
            return handle, personal.model, personal.serving_version
        if handle is None:
            return None, None, None
        if mode == "auto" and personal is not None:
            return handle, personal.model, personal.serving_version
        return handle, handle.model, handle.model_version

    def _recommend(self, request: Payload, snap: Snapshot, arch: ServiceArchitecture, handle: ModelHandle | None,
                   model: HarexRecommender | None, serving_version: str | None, candidates: Sequence[str],
                   top_n: int) -> Payload:
        as_of = _parse_time(request["as_of"])
        before = [b for b in snap.baskets if _before(b, as_of)]
        customer = request["customer_id_local"]
        fallback, scores = None, None
        if handle is None:
            fallback = "no_shared_model"
        elif not before:
            fallback = "no_seller_history"
        else:
            scores = self._model_scores(arch, handle, model, snap, before, customer)
            if scores is None:
                fallback = "no_customer_history"
        if fallback is not None:
            # Local popularity over visits before as_of, ties by ID (interfaces.md §4).
            counts = Counter(item for b in before for item in b.item_ids)
            ranked = sorted(((float(counts[i]), i) for i in candidates), key=lambda p: (-p[0], p[1]))
        else:
            ranked = sorted(((scores[i], i) for i in candidates if i in scores), key=lambda p: (-p[0], p[1]))
        response = {
            "schema_version": "recommendation.v1", "seller_id": self.seller_id, "customer_id_local": customer,
            "as_of": request["as_of"], "model_version": serving_version or FALLBACK_MODEL_VERSION,
            "score_semantics": "next_purchase", "horizon_days": None,
            "is_cold_start": fallback is not None, "fallback_reason": fallback,
            "items": [{"item_id_local": i, "score": s} for s, i in ranked[:top_n]],
        }
        check_payload("recommendation.v1", response)
        return response

    def predict_local(self, request: Payload, *, model_variant: ModelVariant = "text_relation",
                      mode: ServingMode = "auto") -> Payload:
        check_payload("recommendation_request.v1", request)
        if request["seller_id"] != self.seller_id:
            raise ContractError("FORBIDDEN", "/seller_id")
        arch = self._arch(model_variant)
        handle, model, version = self._serving(model_variant, mode)
        snap = self.store.snapshot()
        candidates = self._candidates(request, snap)
        top_n = request["top_n"] if request["top_n"] is not None else DEFAULT_TOP_N
        return self._recommend(request, snap, arch, handle, model, version, candidates, top_n)

    # ------------------------------------------------------------------ warm-up

    def _poke(self) -> None:
        if self._warmer is not None:
            self._idle.clear()
            self._wake.set()

    def _warm_loop(self) -> None:
        while True:
            self._wake.wait()
            while True:  # let a burst of changes settle first
                self._wake.clear()
                if self._closing:
                    return
                if not self._wake.wait(WARM_DEBOUNCE):
                    break
            try:
                self.warm_up()
            except Exception as error:  # a request then computes what it needs itself
                self.load_errors["warm"] = "%s: %s" % (type(error).__name__, error)
            if not self._wake.is_set():
                self._idle.set()

    def warm_up(self) -> None:
        """Compute z, relations and every installed base's e for the current epoch now."""
        snap = self.store.snapshot()
        if not snap.catalog:
            return
        try:
            self._frozen_text()
        except FileNotFoundError:
            return  # no frozen encoder installed: nothing can be scored yet
        with self._state:
            handles = [(variant, handle) for variant, handle in self._handles.items() if handle is not None]
        now = self._clock()
        if not handles or not all(_before(b, now) for b in snap.baskets):
            self._catalog(snap)  # z at least; requests with a later cut compute the rest themselves
            return
        prepared = self._prepare(snap, any(self._arch(v).config.relation for v, _ in handles))
        for variant, handle in handles:
            arch = self._arch(variant)
            self._item_vectors(arch, handle, snap.epoch, prepared.catalog,
                               prepared.relations if arch.config.relation else None, True)

    def wait_warm(self, timeout: float | None = None) -> bool:
        """True once the warm-up has caught up with every change so far (always True without warm=True)."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        """Stop the warm-up thread; the runtime still answers, computing on demand."""
        if self._warmer is not None:
            self._closing = True
            self._wake.set()
            self._warmer.join(timeout=30)
            self._warmer = None
            self._idle.set()

    # ------------------------------------------------------------------ C: training

    def get_local_data_ref(self) -> str:
        epoch = self.store.feature_epoch
        return "fs.%d.%s" % (epoch, self._ref_tag(epoch))

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        if not self._job.acquire(blocking=False):
            raise JobBusyError("a training or personalization job is already running for this seller")
        try:
            yield
        finally:
            self._job.release()

    def training_parts(self, epoch: int, variant: ModelVariant, run_seed: int
                       ) -> tuple[list[SellerData], list[SellerData]]:
        """(train, validation) parts of the snapshot at epoch, one part per relation snapshot."""
        arch = self._arch(variant)
        snap = self.store.snapshot(epoch)
        catalog = self._catalog(snap)
        visits = _visits(snap.baskets, set(catalog.items))
        cuts = _time_cuts(visits)
        groups: dict[bool, dict[int, list]] = {False: {}, True: {}}
        for customer, own in sorted(visits.items()):
            held = is_validation_customer(self.seller_id, run_seed, customer)
            for example in customer_examples(own, range(1, len(own) + 1)):
                if not example.target_items:
                    continue  # every item of the target has no text
                bucket = _bucket(own, example.target_position, cuts) if arch.config.relation else 0
                groups[held].setdefault(bucket, []).append(example)
        row_of = {item: row for row, item in enumerate(catalog.items)}
        relations: dict[int, RelationTensors] = {}

        def parts(grouped: dict[int, list]) -> list[SellerData]:
            out = []
            for bucket, examples in sorted(grouped.items()):
                rel = None
                if arch.config.relation:
                    if bucket not in relations:
                        relations[bucket] = build_relations(_visible(visits, bucket, cuts), row_of, len(catalog.items))
                    rel = relations[bucket]
                out.append(SellerData(self.seller_id, catalog.items, catalog.z, examples, rel))
            return out
        return parts(groups[False]), parts(groups[True])

    def train_round(self, local_data_ref: str, round_config: Payload, *,
                    model_variant: ModelVariant = "text_relation") -> TrainingResult:
        check_payload("round_config.v1", round_config)
        arch = self._arch(model_variant)
        with self._exclusive():
            epoch = self._parse_ref(local_data_ref)
            with self._state:
                handle = self._handles.get(model_variant)
            # The round starts from the base it names, or not at all (model.md §5).
            if handle is None or handle.model_version != round_config["model_version"]:
                raise ContractError("NOT_FOUND", "/model_version")
            if round_config["architecture_version"] != handle.manifest["architecture_version"]:
                raise ContractError("MANIFEST_MISMATCH", "/architecture_version")
            if round_config["manifest_hash"] != handle.manifest["manifest_hash"]:
                raise ContractError("MANIFEST_MISMATCH", "/manifest_hash")
            zero = {name: np.zeros_like(array) for name, array in handle.tensors.items()}
            train_parts, val_parts = self.training_parts(epoch, model_variant, round_config["seed"])
            n_train = sum(len(p.examples) for p in train_parts)
            steps = planned_steps(n_train, round_config)
            if steps == 0:
                return TrainingResult(zero, {"loss_mean": None, "grad_norm_mean": None}, False)
            model = build_model(arch.config)
            load_shared(model, handle.tensors)  # a training copy; the serving model stays as it is
            config = TrainConfig(steps=steps, batch_size=round_config["batch_size"],
                                 n_negatives=round_config["n_neg"], lr=float(round_config["learning_rate"]),
                                 weight_decay=WEIGHT_DECAY, clip_norm=CLIP_NORM,
                                 seed=round_train_seed(self.seller_id, round_config["seed"], round_config["round_id"]))
            log = train(model, train_parts, config)
            after = shared_tensors(model)
            delta = {name: (after[name] - handle.tensors[name]).astype(np.float32) for name in handle.tensors}
            val = validation_loss(model, val_parts, n_negatives=round_config["n_neg"], seed=round_config["seed"]) \
                if val_parts else None
            grad = log["grad_norm_mean"]
            metrics = {"loss_mean": None if val is None or not math.isfinite(val) else float(val),
                       "grad_norm_mean": None if grad is None or not math.isfinite(grad) else float(grad)}
            completed = log["steps"] > 0
            return TrainingResult(delta if completed else zero, metrics, completed)

    # ------------------------------------------------------------------ personalization

    def _personal_dir(self, variant: str, base_version: str) -> Path:
        return self.model_dir / "personal" / variant / base_version

    def _personal_model(self, variant: ModelVariant, base: ModelHandle, heads: Mapping[str, np.ndarray]
                        ) -> HarexRecommender:
        model = build_model(self._arch(variant).config)
        load_shared(model, base.tensors)
        names = personal_names(model)
        if set(heads) != set(names.values()):
            raise ContractError("TENSOR_SET_MISMATCH", "/tensors")
        state = model.state_dict()
        for key, name in names.items():
            if tuple(heads[name].shape) != tuple(state[key].shape) or not np.isfinite(heads[name]).all():
                raise ContractError("MANIFEST_MISMATCH", "/tensors")
            state[key] = torch.from_numpy(np.array(heads[name], dtype=np.float32))
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def _load_personal(self, variant: ModelVariant, base: ModelHandle | None) -> PersonalHandle | None:
        if base is None:
            return None
        folder = self._personal_dir(variant, base.model_version)
        pointer = folder / "CURRENT"
        if not pointer.is_file():
            return None
        try:
            revision = pointer.read_text(encoding="utf-8").strip()
            meta = json.loads((folder / revision / "meta.json").read_text(encoding="utf-8"))
            data = (folder / revision / "weights.npz").read_bytes()
            # Made from exactly this base, or not used at all (model.md §8.4).
            if meta["base_model_version"] != base.model_version or \
                    meta["base_weights_sha256"] != base.release["weights_sha256"] or \
                    meta["weights_sha256"] != _sha256(data) or meta["revision"] != revision:
                raise ContractError("MANIFEST_MISMATCH")
            heads = read_tensors(data, list(personal_names(build_model(self._arch(variant).config)).values()))
            model = self._personal_model(variant, base, heads)
            return PersonalHandle(variant, base.model_version, revision, meta["serving_version"], model, heads)
        except Exception as error:
            self.load_errors["%s.personal" % variant] = "%s: %s" % (type(error).__name__, error)
            return None

    def _record_attempt(self, variant: str, base_version: str, status: str, reason: str | None) -> None:
        folder = self._personal_dir(variant, base_version)
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / ("LAST.%s" % uuid.uuid4().hex)
        temp.write_bytes(canonical_json({"status": status, "reason": reason}))
        os.replace(temp, folder / "LAST.json")

    def _last_reason(self, variant: str, base_version: str) -> str | None:
        last = self._personal_dir(variant, base_version) / "LAST.json"
        if not last.is_file():
            return None
        return REASON_OF_ATTEMPT.get(json.loads(last.read_text(encoding="utf-8")).get("reason"))

    def personalize_local(self, local_data_ref: str, personal_config: Payload, *,
                          model_variant: ModelVariant = "text_relation") -> PersonalizationResult:
        arch = self._arch(model_variant)
        unknown = set(personal_config) - set(DEFAULT_PERSONAL)
        if unknown:
            raise ValueError("unknown personal_config keys: %s" % ", ".join(sorted(unknown)))
        config = dict(DEFAULT_PERSONAL, **personal_config)
        with self._exclusive():
            epoch = self._parse_ref(local_data_ref)
            with self._state:
                base = self._handles.get(model_variant)
            if base is None:
                raise ContractError("NOT_FOUND")
            version = base.model_version
            train_parts, val_parts = self.training_parts(epoch, model_variant, PERSONAL_SPLIT_SEED)
            n_train = sum(len(p.examples) for p in train_parts)
            n_val = sum(len(p.examples) for p in val_parts)
            if n_train < config["min_train_examples"] or n_val < config["min_val_examples"] or config["steps"] == 0:
                self._record_attempt(model_variant, version, "skipped", "insufficient_data")
                return PersonalizationResult("skipped", "insufficient_data", version, None)
            model = build_model(arch.config)
            load_shared(model, base.tensors)
            base_loss = validation_loss(model, val_parts, n_negatives=config["n_neg"], seed=PERSONAL_SPLIT_SEED)
            train(model, train_parts, TrainConfig(steps=config["steps"], batch_size=config["batch_size"],
                                                  n_negatives=config["n_neg"], lr=float(config["lr"]),
                                                  weight_decay=WEIGHT_DECAY, clip_norm=CLIP_NORM,
                                                  seed=config["seed"]), only=PERSONAL_GROUPS)
            model.eval()
            personal_loss = validation_loss(model, val_parts, n_negatives=config["n_neg"], seed=PERSONAL_SPLIT_SEED)
            after = shared_tensors(model)
            moved = tuple("shared.%s_" % group for group in PERSONAL_GROUPS)
            for name, array in base.tensors.items():  # only the two groups may have moved
                if not name.startswith(moved) and not np.array_equal(array, after[name]):
                    raise RuntimeError("personalization changed a shared tensor outside query_proj and scorer")
            if base_loss is None or personal_loss is None or not personal_loss < base_loss:
                self._record_attempt(model_variant, version, "rejected", "validation_rejected")
                return PersonalizationResult("rejected", "validation_rejected", version, None)
            state = model.state_dict()
            heads = {name: state[key].detach().cpu().numpy().astype(np.float32, copy=True)
                     for key, name in personal_names(model).items()}
            data = canonical_npz(heads, list(heads))
            weights_sha = _sha256(data)
            revision = _sha256(canonical_json([self.seller_id, model_variant, version, weights_sha]))[:16]
            serving_version = "personal.%s" % revision
            meta = {"seller_id": self.seller_id, "variant": model_variant, "base_model_version": version,
                    "base_weights_sha256": base.release["weights_sha256"], "revision": revision,
                    "serving_version": serving_version, "weights_sha256": weights_sha, "config": config,
                    "feature_epoch": epoch, "n_train": n_train, "n_val": n_val,
                    "val_loss_base": base_loss, "val_loss_personal": personal_loss}
            folder = self._personal_dir(model_variant, version)
            final = folder / revision
            with self._state:
                current = self._handles.get(model_variant)
                if current is None or current.model_version != version:
                    # The base changed while this ran: the result belongs to no serving base.
                    self._record_attempt(model_variant, version, "rejected", "base_mismatch")
                    return PersonalizationResult("rejected", "base_mismatch", version, None)
                if not final.exists():
                    staging = folder / (".incoming-%s" % uuid.uuid4().hex)
                    staging.mkdir(parents=True)
                    try:
                        (staging / "weights.npz").write_bytes(data)
                        (staging / "meta.json").write_bytes(canonical_json(meta))
                        os.replace(staging, final)
                    except BaseException:
                        shutil.rmtree(staging, ignore_errors=True)
                        raise
                pointer = folder / ("CURRENT.%s" % uuid.uuid4().hex)
                pointer.write_text(revision, encoding="utf-8")
                os.replace(pointer, folder / "CURRENT")
                self._record_attempt(model_variant, version, "installed", None)
                served = self._personal_model(model_variant, current, heads)
                self._personal[model_variant] = PersonalHandle(model_variant, version, revision, serving_version,
                                                               served, heads)
            return PersonalizationResult("installed", None, version, revision)

    # ------------------------------------------------------------------ comparison

    # ------------------------------------------------------------------ comparison pair

    def _pin_path(self) -> Path:
        return self.model_dir / "comparison.json"

    def comparison_pin(self) -> dict[str, str]:
        """variant -> base version compare_local uses; a variant missing here uses its serving base."""
        path = self._pin_path()
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def pin_comparison(self, *, text_only: str | None = None, text_relation: str | None = None) -> None:
        """Fix the base pair compare_local uses (comparison.md §5: a pair chosen beforehand, not the
        latest). Each version must be a base this seller installed; None leaves that variant on its
        serving base, and both None removes the pin. Local, survives a restart."""
        pair = {v: version for v, version in (("text_only", text_only), ("text_relation", text_relation))
                if version is not None}
        for variant, version in pair.items():
            if not (self._base_dir(variant) / version / "release.json").is_file():
                raise ContractError("NOT_FOUND", "/%s" % variant)
            self._pinned_base(variant, version, strict=True)
        path = self._pin_path()
        with self._state:
            if not pair:
                path.unlink(missing_ok=True)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name("comparison.%s" % uuid.uuid4().hex)
            temp.write_bytes(canonical_json(pair))
            os.replace(temp, path)

    def _pinned_base(self, variant: ModelVariant, version: str, *, strict: bool = False) -> ModelHandle | None:
        with self._state:
            handle = self._pinned.get((variant, version))
        if handle is not None:
            return handle
        try:
            handle = self._read_base(variant, version)
        except Exception as error:
            if strict:
                raise ContractError("MANIFEST_MISMATCH", "/%s" % variant) from error
            self.load_errors["%s.comparison" % variant] = "%s: %s" % (type(error).__name__, error)
            return None
        with self._state:
            self._pinned = {k: v for k, v in self._pinned.items() if k[0] != variant}
            self._pinned[(variant, version)] = handle
        return handle

    def compare_local(self, request: Payload) -> ComparisonResult:
        """T-G, R-G, T-P, R-P on one pinned snapshot, candidate set and handle set (interfaces.md §4.1)."""
        check_payload("recommendation_request.v1", request)
        if request["seller_id"] != self.seller_id:
            raise ContractError("FORBIDDEN", "/seller_id")
        pin = self.comparison_pin()
        with self._state:
            pinned = {v: (self._handles.get(v), self._personal.get(v)) for v in VARIANTS}
        for variant, version in pin.items():
            base, _ = pinned[variant]
            if base is None or base.model_version != version:
                base = self._pinned_base(variant, version)  # None: the arm says model_not_ready
                pinned[variant] = (base, self._load_personal(variant, base))
        snap = self.store.snapshot()
        candidates = self._candidates(request, snap)
        top_n = request["top_n"] if request["top_n"] is not None else DEFAULT_TOP_N
        arms = []
        for arm_id, variant, mode in ARMS:
            arch = self._arch(variant)
            base, personal = pinned[variant]
            if personal is not None and (base is None or personal.base_model_version != base.model_version):
                personal = None
            if base is None:
                arms.append(ComparisonArm(arm_id, variant, mode, False, "model_not_ready", None, None, None))
            elif mode == "global":
                rec = self._recommend(request, snap, arch, base, base.model, base.model_version, candidates, top_n)
                arms.append(ComparisonArm(arm_id, variant, mode, True, None, base.model_version, None, rec))
            elif personal is None:
                reason = self._last_reason(variant, base.model_version) or "personalization_not_ready"
                arms.append(ComparisonArm(arm_id, variant, mode, False, reason, base.model_version, None, None))
            else:
                rec = self._recommend(request, snap, arch, base, personal.model, personal.serving_version,
                                      candidates, top_n)
                arms.append(ComparisonArm(arm_id, variant, mode, True, None, base.model_version,
                                          personal.revision, rec))
        unique = sorted(set(candidates))
        return ComparisonResult(comparison_id=uuid.uuid4().hex, as_of=request["as_of"],
                                feature_snapshot_id="fe-%d" % snap.epoch,
                                candidate_set_hash=_sha256(canonical_json(unique)), arms=tuple(arms))
