"""Compatibility wrappers for the former spectral strategy package.

New code should import from ``graphfl_lab.strategies.graphfl``.  This package
stays intentionally thin so older experiments and notebooks keep running.
"""

from graphfl_lab.strategies.graphfl import *  # noqa: F401,F403
