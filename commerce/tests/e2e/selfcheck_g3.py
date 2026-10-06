"""g3 selfcheck: synthetic FL over B's real runtime and C's coordinator, CI-sized.

Exit codes follow commerce/tools/gate.py: 0 all items, 3 implemented items pass but some
remain, 1 a check failed. The remaining items are printed so that the in-process run is
never read as the whole g3.
"""
import os
import unittest

REMAINING = (
    "the same flow with separate processes started by run_local (launcher FL waits for OQ16)",
    "the real model run (MiniLM + first release) is reported on the PR, not run in CI",
)


def run(verbose: bool = False) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    suite = unittest.defaultTestLoader.discover(here, pattern="test_g3_*.py", top_level_dir=repo_root)
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("g3: in-process flow passed (tests=%d); remaining items:" % result.testsRun)
    for item in REMAINING:
        print("  - " + item)
    return 3
