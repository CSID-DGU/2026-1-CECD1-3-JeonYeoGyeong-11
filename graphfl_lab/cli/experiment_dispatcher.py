"""Unified experiment dispatcher for single-run tracks."""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence

from graphfl_lab.cli import cora_experiment, vision_experiment


TRACK_MODULES = {
    "cora": cora_experiment,
    "vision": vision_experiment,
}


def _parse_track(argv: Sequence[str]) -> tuple[str | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--track", choices=sorted(TRACK_MODULES))
    parsed, remaining = parser.parse_known_args(list(argv))
    return parsed.track, remaining


def main(argv: Sequence[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    track, remaining = _parse_track(args)
    if track is None:
        track = "cora"
        warnings.warn(
            "run_experiment.py without --track is deprecated and currently "
            "defaults to the Cora runner. Use --track cora or --track vision.",
            DeprecationWarning,
            stacklevel=2,
        )
    module = TRACK_MODULES[track]
    old_argv = sys.argv
    sys.argv = [old_argv[0], *remaining]
    try:
        return module.main()
    finally:
        sys.argv = old_argv


def main_for_track(track: str, argv: Sequence[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    return main(["--track", track, *args])


def cora_main(argv: Sequence[str] | None = None):
    return main_for_track("cora", argv)


def vision_main(argv: Sequence[str] | None = None):
    return main_for_track("vision", argv)
