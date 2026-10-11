"""Start only the seller apps for a screen demo, with run_local's layout but a longer start-up wait.

run_local (C) waits 20 s per service. With trained models installed each seller loads torch and
the frozen encoder first (about 10 s alone, more when several start together), so a six-store
demo can stop at "merchant health timeout". This helper starts merchant-1..N the same way
(MERCHANT_ID, FEATURE_DB_PATH and MODEL_DIR under commerce/deploy/var/merchant_i, port 8100+i,
FL off) and waits up to --wait seconds for each. It does not start central or the coordinator.
Use run_local for anything FL; this is a stopgap until its wait is raised.

    python -m commerce.services.merchant_api.serve_demo --merchants 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[3]
VAR = ROOT / "commerce" / "deploy" / "var"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merchants", type=int, default=6)
    parser.add_argument("--wait", type=int, default=180, help="seconds to wait for each seller's /healthz")
    args = parser.parse_args(argv)
    opener = build_opener(ProxyHandler({}))
    processes = []
    try:
        for i in range(1, args.merchants + 1):
            base = VAR / f"merchant_{i}"
            base.mkdir(parents=True, exist_ok=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("MERCHANT_ID", "FEATURE_", "MODEL_", "FL_"))}
            env.update({"MERCHANT_ID": f"merchant-{i}", "FEATURE_DB_PATH": str(base / "features.sqlite"),
                        "MODEL_DIR": str(base / "models"), "FL_ENABLED": "false", "FL_MODE": "protected"})
            log = open(base / "server.log", "ab")
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "commerce.services.merchant_api.main:app", "--host", "127.0.0.1",
                 "--port", str(8100 + i), "--workers", "1", "--no-access-log", "--log-level", "warning"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        for i, process in enumerate(processes, start=1):
            deadline = time.monotonic() + args.wait
            while True:
                try:
                    with opener.open(f"http://127.0.0.1:{8100 + i}/healthz", timeout=1) as response:
                        json.load(response)
                    print(f"merchant-{i}: http://127.0.0.1:{8100 + i}/buyer/merchant-{i}/", flush=True)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        print(f"merchant-{i} did not start; see {VAR / f'merchant_{i}' / 'server.log'}", flush=True)
                        return 1
                    time.sleep(0.5)
        print("ALL SELLERS UP (Ctrl+C to stop)", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
