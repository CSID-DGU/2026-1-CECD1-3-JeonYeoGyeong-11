"""b1 checks on hand-made synthetic input only; run through the b1 selfcheck."""
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
INSTACART_SMALL = FIXTURES / "instacart_small"
LIVE = FIXTURES / "live"
CONTRACT_FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "fixtures"
