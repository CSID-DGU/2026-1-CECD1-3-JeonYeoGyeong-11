"""c1 selfcheck: the synthetic round core with generated tensors.

Exit codes follow commerce/tools/gate.py: 0 all items, 3 implemented items pass
but some remain, 1 a check failed. The remaining items are printed so that a
passing core is never read as a finished c1.
"""
import os
import unittest

REMAINING = (
    "HTTP routes and Bearer seller auth over this core (auth failure code waits for OQ13)",
    "coordinator-side persistence of releases (REGISTRY_DIR) and the pre-fixed cohort setting",
    "FL client submission path and plaintext activation (waits for OQ17)",
    "run_local integration: coordinator health order and shutdown",
)


def run(verbose: bool = False) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    suite = unittest.defaultTestLoader.discover(
        os.path.join(repo_root, "commerce", "tests", "e2e"), pattern="test_c1_round.py", top_level_dir=repo_root)
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("c1: round core checks passed (tests=%d); remaining items:" % result.testsRun)
    for item in REMAINING:
        print("  - " + item)
    return 3
