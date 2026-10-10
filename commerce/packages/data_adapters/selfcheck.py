"""b1 selfcheck: adapters, live input, product text, identifiers, missing values.

Runs the tests in tests/ on hand-made synthetic input. Raw-data reproduction is
reported on the PR instead (data.md §6). Exit 3 while items below remain.
"""
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

REMAINING = ()


def run(verbose: bool = False) -> int:
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), top_level_dir=str(REPO_ROOT))
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1).run(suite)
    if not result.wasSuccessful() or result.testsRun == 0:
        return 1
    print("b1 구현 항목 통과: tests=%d" % result.testsRun)
    for item in REMAINING:
        print("남은 b1 항목: %s" % item)
    return 3 if REMAINING else 0
