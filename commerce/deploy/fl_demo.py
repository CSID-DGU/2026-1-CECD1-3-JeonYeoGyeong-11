"""Synthetic FL demo: the one place that turns FL on (D0025, OQ16).

    python -m commerce.deploy.fl_demo --rounds 2            # CI-sized model, no encoder needed
    python -m commerce.deploy.fl_demo --rounds 2 --keep     # stay up afterwards (Ctrl+C to stop)
    python -m commerce.deploy.fl_demo --rounds 2 --rehearse # whole-flow checks over A's screens and routes
    python -m commerce.deploy.fl_demo --encoder-dir <frozen_text> --release-dir <release>  # real model

run_local keeps FL off. This runner builds a fresh demo root (commerce/deploy/var/fl_demo/<time>/,
Git-ignored), fills each seller's feature store from B's deterministic g3 scenario, writes a
synthetic input attestation per seller (ids.snapshot_digest of exactly what was generated),
imports base-0, issues seller tokens, serves the coordinator in this process and starts one
merchant app per seller (A's create_app) with FL_MODE=synthetic_plaintext. It then opens a fixed
number of rounds over the cohort; the fourth seller is not in the cohort and only installs the
new base (install_only). With --rehearse that seller is filled through A's demo seed instead and
the run ends with HTTP checks of the whole flow (see commerce/deploy/README.md). Every seller's FL client refuses to submit if its snapshot no longer matches the
attestation, for instance after a live order. Generated synthetic input only; no protection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
COORDINATOR_PORT = 8250  # apart from run_local's 8200 so both can run
MERCHANT_BASE_PORT = 8150  # seller i listens on 8150 + i
GENERATOR = "commerce.packages.recommender.scenario.g3"


def _expected_digest(seller) -> str:
    from commerce.packages.contracts import ids
    return ids.snapshot_digest([event["purchase_event_id"] for event in seller.events],
                               {item["item_id_local"]: item for item in seller.catalog})


def _runtime_factory(tiny: bool, encoder_dir: str | None):
    from commerce.packages.recommender import scenario as sc
    from commerce.packages.recommender.seller_runtime import SellerRuntime
    if tiny:
        text = sc.FakeText()
        return lambda seller_id, features, models: SellerRuntime(seller_id, features, models, text=text,
                                                                 architectures=sc.TINY_ARCHITECTURES)
    return lambda seller_id, features, models: SellerRuntime(seller_id, features, models,
                                                             encoder_dir=encoder_dir, warm=True)


def _paths(root: Path, seller_id: str) -> tuple[Path, Path, Path]:
    folder = root / seller_id
    return folder / "features.sqlite", folder / "models", folder / "synthetic_attestation.json"


def _seed_storefront(root: Path, seller_id: str) -> None:
    """The seller outside the cohort is filled through A's own path (orders.sqlite + outbox).

    A's demo seed writes accounts, products and past orders into orders.sqlite; the merchant
    app delivers that outbox to B at start, so the A/B pair stays consistent (interfaces.md §2).
    This seller never trains, so it needs no attestation; it only installs released models.
    """
    features, _models, _ = _paths(root, seller_id)
    features.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MERCHANT_ID=seller_id, FEATURE_DB_PATH=str(features))
    env.pop("MERCHANT_DB_PATH", None)  # orders.sqlite next to features.sqlite, as the merchant app expects
    subprocess.run([sys.executable, "-m", "commerce.services.merchant_api.seed_demo_data", "--bulk", "20"],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)


def prepare(root: Path, *, tiny: bool, encoder_dir: str | None, release_dir: str | None, variant: str,
            storefront_from_a: bool = False) -> dict:
    """Fresh stores, attestations for the cohort, base-0 and tokens. Returns the seller plan.

    Cohort sellers get B's deterministic g3 scenario and an attestation. The seller outside
    the cohort is install-only; with storefront_from_a it is filled through A's demo seed
    so A's screens have products, accounts and orders (rehearsal).
    """
    from commerce.packages.fl_client.attestation import write_attestation
    from commerce.packages.recommender import scenario as sc
    from commerce.packages.recommender.serving import load_bundle
    from commerce.services.fl_coordinator import auth
    from commerce.services.fl_coordinator.round_core import ModelRegistry

    if root.exists() and any(root.iterdir()):
        raise SystemExit("demo root must be new or empty: %s" % root)  # never reuse an existing store
    make = _runtime_factory(tiny, encoder_dir)
    data = sc.scenario()
    plan, manifest, tensors = {}, None, None
    for seller_id, seller in data.sellers.items():
        in_cohort = seller_id in sc.COHORT
        plan[seller_id] = {"in_cohort": in_cohort}
        if not in_cohort and storefront_from_a:
            _seed_storefront(root, seller_id)
            continue
        features, models, attest = _paths(root, seller_id)
        runtime = make(seller_id, features, models)
        try:
            sc.load_seller(runtime, seller)
            if in_cohort:
                expected = _expected_digest(seller)
                if not hasattr(runtime, "snapshot_digest"):
                    raise SystemExit("B's runtime has no snapshot_digest yet (D0025); the demo cannot attest its input")
                if runtime.snapshot_digest(runtime.get_local_data_ref()) != expected:
                    raise SystemExit("seller %s: the store does not hold exactly the generated input" % seller_id)
                write_attestation(attest, seller_id, expected, GENERATOR)
            if manifest is None:
                if release_dir:
                    _release, manifest, tensors = load_bundle(release_dir)
                else:
                    manifest, tensors = sc.random_init_release(runtime, variant)
        finally:
            closer = getattr(runtime, "close", None)
            if closer is not None:
                closer()
    registry = root / "fl" / variant / "registry"
    ModelRegistry(registry).register("base-0", manifest, tensors, provenance={
        "source": "import", "kind": "trained" if release_dir else "random_init", "note": "fl_demo base-0"})
    for seller_id in plan:
        plan[seller_id]["token"] = auth.issue(root / "fl" / variant / "auth.json", seller_id)
    return plan


def _tamper(root: Path, seller_id: str, tiny: bool, encoder_dir: str | None) -> None:
    """Show the OQ17 guard: one purchase the generator did not write, as a live order would be."""
    from commerce.packages.recommender import scenario as sc
    seller = sc.scenario().sellers[seller_id]
    features, models, _ = _paths(root, seller_id)
    runtime = _runtime_factory(tiny, encoder_dir)(seller_id, features, models)
    try:
        runtime.ingest_purchase_event(sc.new_item_purchase(seller_id, seller.customers[0],
                                                           seller.catalog[0]["item_id_local"]))
    finally:
        closer = getattr(runtime, "close", None)
        if closer is not None:
            closer()
    print("tampered: %s got one purchase outside its attestation" % seller_id, flush=True)


def _serve_coordinator(root: Path, variant: str):
    import uvicorn
    from commerce.services.fl_coordinator.main import CoordinatorSettings, create_app
    base = root / "fl" / variant
    app = create_app(CoordinatorSettings(base / "registry", base / "auth.json", base / "rounds",
                                         mode="synthetic_plaintext"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=COORDINATOR_PORT,
                                           log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="fl-demo-coordinator", daemon=True)
    thread.start()
    return app.state.coordinator, server, thread


def _wait_health(url: str, service: str, seconds: float = 120.0) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + seconds
    while True:
        try:
            with opener.open(url + "/healthz", timeout=1.0) as response:
                if json.load(response).get("service") == service:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("%s did not become healthy" % url)
        time.sleep(0.3)


def _current_base(root: Path, seller_id: str, variant: str) -> str | None:
    pointer = _paths(root, seller_id)[1] / "base" / variant / "CURRENT"  # working-agreement §3 layout
    return pointer.read_text(encoding="utf-8").strip() if pointer.is_file() else None


def run(args) -> int:
    from commerce.packages.recommender import scenario as sc
    from commerce.services.fl_coordinator.service import RoundSettings

    root = Path(args.root) if args.root else ROOT / "commerce/deploy/var/fl_demo" / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tiny = not args.encoder_dir
    print("demo root: %s (%s model)" % (root, "CI-sized" if tiny else "real"), flush=True)
    plan = prepare(root, tiny=tiny, encoder_dir=args.encoder_dir, release_dir=args.release_dir, variant=args.variant,
                   storefront_from_a=args.rehearse)
    if args.tamper:
        _tamper(root, args.tamper, tiny, args.encoder_dir)
    coordinator, server, thread = _serve_coordinator(root, args.variant)
    coordinator_url = "http://127.0.0.1:%d" % COORDINATOR_PORT
    processes: list[subprocess.Popen] = []
    logs: list = []
    try:
        _wait_health(coordinator_url, "coordinator")
        for i, (seller_id, info) in enumerate(plan.items(), start=1):
            port = MERCHANT_BASE_PORT + i
            command = [sys.executable, "-m", "commerce.deploy.fl_demo", "merchant", "--root", str(root),
                       "--seller", seller_id, "--port", str(port), "--coordinator", coordinator_url,
                       "--variant", args.variant]
            if args.encoder_dir:
                command += ["--encoder-dir", args.encoder_dir]
            if not info["in_cohort"]:
                command.append("--install-only")
            env = dict(os.environ, FL_DEMO_TOKEN=info["token"])  # not on the command line
            log_path = root / seller_id / "merchant.log"  # errors of this seller's app, for diagnosis
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "ab")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            info["url"] = "http://127.0.0.1:%d" % port
        for info in plan.values():
            _wait_health(info["url"], "merchant")
            print("merchant up: %s" % info["url"], flush=True)
        settings = RoundSettings(local_steps=6, max_local_epochs=2, n_neg=8, batch_size=8, learning_rate=0.01,
                                 seed=0, deadline_seconds=args.deadline) if tiny else RoundSettings(
            local_steps=20, max_local_epochs=2, n_neg=20, batch_size=16, learning_rate=1e-3, seed=0,
            deadline_seconds=args.deadline)
        results = []
        for number in range(1, args.rounds + 1):
            config = coordinator.open_round(sc.COHORT, settings)
            print("round %d open: %s on %s" % (number, config["round_id"], config["model_version"]), flush=True)
            while coordinator.ledger.get(config["round_id"])["state"] == "open":
                time.sleep(0.5)
            entry = coordinator.ledger.get(config["round_id"])
            results.append(entry["state"])
            print("round %d %s%s" % (number, entry["state"], " -> " + entry["result_model_version"]
                                     if entry.get("result_model_version") else ""), flush=True)
            latest = coordinator.registry.latest()["model_version"]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and any(
                    _current_base(root, s, args.variant) != latest for s in plan):
                time.sleep(0.5)
            for seller_id in plan:
                print("  %s serves %s" % (seller_id, _current_base(root, seller_id, args.variant)), flush=True)
        ok = all(state == "aggregated" for state in results)
        print("FL DEMO %s: rounds=%s" % ("OK" if ok else "INCOMPLETE", ",".join(results)), flush=True)
        if args.rehearse:
            checks = rehearse(plan, root, coordinator, args.variant, settings)
            failed = [name for name, passed, _ in checks if not passed]
            print("REHEARSAL %s: checks=%d failed=%s" % ("OK" if not failed else "FAILED", len(checks),
                                                          ",".join(failed) or "none"), flush=True)
            ok = ok and not failed
        if args.keep:
            print("merchant screens: %s (Ctrl+C to stop)" % ", ".join(i["url"] for i in plan.values()), flush=True)
            while all(p.poll() is None for p in processes):
                time.sleep(0.5)
        return 0 if ok else 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        server.should_exit = True
        thread.join(timeout=10)
        for log in logs:
            log.close()


def rehearse(plan: dict, root: Path, coordinator, variant: str, settings) -> list[tuple[str, bool, str]]:
    """Whole-flow checks over HTTP after the rounds (A screens + B runtime + C coordinator).

    The storefront is the seller outside the cohort, filled through A's demo seed. Every
    check prints PASS or FAIL; the last one shows the OQ17 guard by sending one catalog
    change through A's route to a cohort seller and opening another round.
    """
    import re
    import uuid
    import httpx
    from dataclasses import replace
    from commerce.packages.recommender import scenario as sc
    from commerce.services.merchant_api import seed_demo_data as seed

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))
        print("%s %s %s" % ("PASS" if passed else "FAIL", name, detail), flush=True)

    store = sc.NEW_SELLER
    url = plan[store]["url"]
    customer, _ = seed._CUSTOMERS[0]
    latest = coordinator.registry.latest()["model_version"]
    check("R1 storefront (outside the cohort) serves the latest FL base", _current_base(root, store, variant) == latest,
          str(_current_base(root, store, variant)))

    def home_state(client) -> tuple[int, bool, str]:
        page = client.get("/buyer/%s/" % store)
        badge = re.search(r'class="badge muted">([^<]+)<', page.text)
        ranked = page.status_code == 200 and "추천 상품" in page.text and "아직 모델 추천을 낼 수 없어" not in page.text
        return page.status_code, ranked, badge.group(1) if badge else "model ranking"

    with httpx.Client(base_url=url, timeout=60.0) as buyer:
        login = buyer.post("/buyer/%s/login" % store, data={"customer_id_local": customer, "password": seed._DEMO_PASSWORD})
        check("R2 buyer login", login.status_code == 303, str(login.status_code))
        status, _ranked, shown = home_state(buyer)
        check("R3 buyer home answers before the order", status == 200, "%s, %s" % (status, shown))
        order = buyer.post("/sellers/%s/orders" % store, json={
            "customer_id_local": customer, "idempotency_key": "rehearsal-%s" % uuid.uuid4().hex,
            "items": [{"item_id_local": "sku-milk", "quantity": 1, "unit_price_minor": 2500}], "currency": "KRW"})
        state, trail = (order.json() if order.status_code == 201 else {}), ["place:%d" % order.status_code]
        for action in ("accept", "complete"):
            if not state:
                break
            step = buyer.post("/sellers/%s/orders/%s/%s" % (store, state["order_id"], action),
                              json={"expected_status_version": state["status_version"]})
            trail.append("%s:%d" % (action, step.status_code))
            state = step.json() if step.status_code == 200 else {}
        check("R4 order placed, accepted and completed through A (B ingests it)", state.get("status") == "completed",
              " ".join(trail))
        started, ranked = time.monotonic(), False
        while not ranked and time.monotonic() - started < 120:  # B folds the new purchase into its features
            status, ranked, shown = home_state(buyer)
            if not ranked:
                time.sleep(2)
        check("R5 after the order the buyer home shows a model ranking (FL base)", ranked,
              "%s, %s after %.0fs" % (status, shown, time.monotonic() - started))

    with httpx.Client(base_url=url, timeout=60.0) as staff:
        login = staff.post("/seller/%s/login" % store, data={"username": seed._SELLER_ACCOUNT["username"],
                                                             "password": seed._SELLER_ACCOUNT["password"]})
        compare = staff.get("/seller/%s/compare" % store, params={"customer_id_local": customer})
        arms = all(arm in compare.text for arm in ("T-G", "R-G", "T-P", "R-P"))
        check("R6 seller compare screen shows the four arms", login.status_code == 303 and compare.status_code == 200
              and arms, "%s/%s" % (login.status_code, compare.status_code))

    cohort_seller = sc.COHORT[0]
    with httpx.Client(base_url=plan[cohort_seller]["url"], timeout=60.0) as api:
        changed = api.post("/sellers/%s/catalog-items" % cohort_seller, json={
            "item_id_local": "rehearsal-live-item", "title_text": "리허설 상품", "display_price_minor": 1000})
    guard = coordinator.open_round(sc.COHORT, replace(settings, deadline_seconds=20))
    while coordinator.ledger.get(guard["round_id"])["state"] == "open":
        time.sleep(0.5)
    state = coordinator.ledger.get(guard["round_id"])["state"]
    check("R7 a live catalog change on a cohort seller stops its FL: the round is discarded",
          changed.status_code == 201 and state == "discarded" and coordinator.registry.latest()["model_version"] == latest,
          "%s/%s" % (changed.status_code, state))
    return checks


def merchant(args) -> int:
    """One seller: A's merchant app with the synthetic FL client turned on (child of run)."""
    import uvicorn
    from commerce.packages.fl_client.lifecycle import FLClientConfig
    from commerce.services.merchant_api.context import MerchantSettings, build_context
    from commerce.services.merchant_api.main import create_app

    features, models, attest = _paths(Path(args.root), args.seller)
    settings = MerchantSettings(
        seller_id=args.seller, feature_db_path=features, model_dir=models,
        fl=FLClientConfig(enabled=True, mode="synthetic_plaintext", model_variant=args.variant,
                          coordinator_url=args.coordinator, token=os.environ["FL_DEMO_TOKEN"],
                          synthetic_attestation=None if args.install_only else attest,
                          install_only=args.install_only, poll_seconds=1.0))
    factory = _runtime_factory(not args.encoder_dir, args.encoder_dir)
    app = create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=factory))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    child = sub.add_parser("merchant", help=argparse.SUPPRESS)
    for p in (parser, child):
        p.add_argument("--variant", default="text_relation", choices=("text_only", "text_relation"))
        p.add_argument("--encoder-dir", help="frozen encoder folder for the real model (omit: CI-sized model)")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--root", help="new or empty demo folder (default: commerce/deploy/var/fl_demo/<time>)")
    parser.add_argument("--release-dir", help="first release folder (release.json, manifest.json, weights.npz)")
    parser.add_argument("--deadline", type=int, default=300, help="round deadline in seconds")
    parser.add_argument("--keep", action="store_true", help="keep the merchants up after the rounds")
    parser.add_argument("--tamper", metavar="SELLER",
                        help="add one non-generated purchase to SELLER after attesting, to show the refusal")
    child.add_argument("--root", required=True)
    child.add_argument("--seller", required=True)
    child.add_argument("--port", type=int, required=True)
    child.add_argument("--coordinator", required=True)
    child.add_argument("--install-only", action="store_true")
    parser.add_argument("--rehearse", action="store_true",
                        help="fill the seller outside the cohort through A's demo seed and check the whole flow")
    args = parser.parse_args(argv)
    if args.release_dir and not args.encoder_dir:
        parser.error("--release-dir needs --encoder-dir (a real release needs the real encoder)")
    return merchant(args) if args.command == "merchant" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
