"""Compatibility wrapper for ``merge_vision_fedavg_ours.py``."""

from importlib import util
from pathlib import Path

_IMPL_PATH = Path(__file__).with_name("merge_vision_fedavg_ours.py")
_SPEC = util.spec_from_file_location("merge_vision_fedavg_ours", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load {_IMPL_PATH}")
_IMPL = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

if __name__ == "__main__":
    _IMPL.main()
