"""Compatibility wrappers for the former spectral strategy package.

New code should import from ``spectral_fl.strategies.graphfl``.  This package
stays intentionally thin so older experiments and notebooks keep running.
"""

from spectral_fl.strategies.graphfl import *  # noqa: F401,F403
