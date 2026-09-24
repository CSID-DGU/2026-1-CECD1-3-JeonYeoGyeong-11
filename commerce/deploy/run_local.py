"""Start the local services and check each /healthz. FL stays disabled."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[2]


def service_plan(merchants: int) -> list[tuple[str, str, int, dict[str, str]]]:
    plan = [
        ("central", "commerce.services.central_api.main:app", 8000, {}),
        ("coordinator", "commerce.services.fl_coordinator.main:app", 8200, {}),
    ]
    for i in range(1, merchants + 1):
        base = ROOT / "commerce/deploy/var" / f"merchant_{i}"
        plan.append(("merchant", "commerce.services.merchant_api.main:app", 8100 + i, {
            "MERCHANT_ID": f"merchant-{i}", "FEATURE_DB_PATH": str(base / "features.sqlite"),
            "MODEL_DIR": str(base / "models"), "FL_ENABLED": "false", "FL_MODE": "protected",
            "FL_MODEL_VARIANT": "text_relation",
        }))
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merchants", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--check", action="store_true", help="Show process plan without starting it")
    parser.add_argument("--smoke", action="store_true", help="Check health then stop all started processes")
    args = parser.parse_args(argv)
    plan = service_plan(args.merchants)
    for role, module, port, _ in plan:
        print(f"{role}: {module} -> http://127.0.0.1:{port}/healthz", flush=True)
    if args.check:
        return 0
    processes: list[subprocess.Popen] = []
    ready: dict[str, bool] = {}
    opener = build_opener(ProxyHandler({}))
    try:
        # Detect occupied ports before launching; never stop an unrelated server.
        for _, _, port, _ in plan:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        for role, module, port, overrides in plan:
            # Do not forward merchant secrets/paths into central child environments.
            env = {k: v for k, v in os.environ.items() if not k.startswith(
                ("MERCHANT_", "FEATURE_", "MODEL_", "FL_", "COORDINATOR_", "REGISTRY_", "AUTH_", "ROUND_", "CENTRAL_"))}
            env.update(overrides)
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1",
                 "--port", str(port), "--workers", "1", "--no-access-log", "--log-level", "warning"],
                cwd=ROOT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            processes.append(process)
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{role} startup failed")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=0.5) as response:
                        payload = json.load(response)
                    # Check the shape only. Pinning today's "scaffold"/ready=false would fail the
                    # required CI job the moment A or C implements a service.
                    if payload.get("service") != role or not isinstance(payload.get("ready"), bool):
                        raise RuntimeError(f"{role}: unexpected health response {payload!r}")
                    ready[f"{role}:{port}"] = payload["ready"]
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{role} health timeout")
                    time.sleep(0.1)
        states = " ".join(f"{name}={'ready' if ok else 'not-ready'}" for name, ok in ready.items())
        print(f"SERVICES UP: {states}; FL disabled", flush=True)
        if not args.smoke:
            while all(p.poll() is None for p in processes):
                time.sleep(0.5)
            raise RuntimeError("A scaffold service exited")
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == "__main__":
    raise SystemExit(main())
