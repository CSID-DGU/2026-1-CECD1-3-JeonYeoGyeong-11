"""g2 selfcheck: order -> features -> real recommendation through A's app and B's runtime, CI-sized.

Exit codes follow commerce/tools/gate.py: 0 all items, 3 implemented items pass but some
remain, 1 a check failed. A runs this gate (working-agreement §4); C writes it.
"""
import os
import unittest

REMAINING = (
    "redelivery after a B failure reflected exactly once, checked through the real runtime",
    "the same flow as separate processes is commerce.deploy.fl_demo --rehearse (manual)",
)


def run(verbose: bool = False) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    suite = unittest.defaultTestLoader.discover(here, pattern="test_g2_*.py", top_level_dir=repo_root)
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("g2: in-process flow passed (tests=%d); remaining items:" % result.testsRun)
    for item in REMAINING:
        print("  - " + item)
    return 3
