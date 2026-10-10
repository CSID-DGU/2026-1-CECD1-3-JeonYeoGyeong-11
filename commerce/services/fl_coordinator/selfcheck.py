"""c1 selfcheck: the synthetic round core with generated tensors.

Exit codes follow commerce/tools/gate.py: 0 all items, 3 implemented items pass
but some remain, 1 a check failed. The remaining items are printed so that a
passing core is never read as a finished c1.
"""
import os
import unittest

REMAINING = (
    "contract_error body of the 401 response (waits for OQ13; today 401 has an empty body)",
    "who opens rounds and when, with a fixed round count (trigger waits for OQ08)",
    "FLClient.start running synthetic_plaintext rounds for a seller (waits for OQ17)",
    "run_local starting a configured coordinator and enabled clients (waits for OQ16)",
)


def run(verbose: bool = False) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    suite = unittest.defaultTestLoader.discover(
        os.path.join(repo_root, "commerce", "tests", "e2e"), pattern="test_c1_*.py", top_level_dir=repo_root)
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("c1: round core checks passed (tests=%d); remaining items:" % result.testsRun)
    for item in REMAINING:
        print("  - " + item)
    return 3
