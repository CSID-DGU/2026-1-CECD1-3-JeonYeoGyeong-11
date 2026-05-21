"""Execution helpers shared by suite orchestrators."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Sequence


def run_cmd(cmd: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def execute_or_reuse_result(
    cmd: Sequence[str],
    result_path: Path,
    reuse_existing: bool,
    *,
    cwd: Path,
) -> tuple[bool, float | None]:
    """Return (reused_existing_result, observed_subprocess_wall_time_sec)."""
    if reuse_existing and result_path.is_file():
        print(f"Reusing existing result: {result_path}")
        return True, None
    start = time.perf_counter()
    run_cmd(cmd, cwd=cwd)
    return False, float(time.perf_counter() - start)
