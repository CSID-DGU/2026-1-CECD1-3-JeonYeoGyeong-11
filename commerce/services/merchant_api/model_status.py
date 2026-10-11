"""What the seller dashboard says about the recommender: which shared models are installed in
MODEL_DIR (the working-agreement §3 layout: frozen_text/, base/<variant>/CURRENT), and how B
answers a customer who has bought here. Read-only; B owns the files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

VARIANTS = ("text_relation", "text_only")
FALLBACK_LABELS = {
    "no_shared_model": "공유 모델이 아직 설치되지 않음",
    "no_seller_history": "이 매장의 판매 기록이 아직 없음",
    "no_customer_history": "구매 기록이 없는 고객(첫 방문)",
}


def installed_models(model_dir: Optional[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {"encoder": False, "variants": {v: None for v in VARIANTS}}
    if model_dir is None:
        return out
    model_dir = Path(model_dir)
    out["encoder"] = (model_dir / "frozen_text" / "config.json").is_file()
    for variant in VARIANTS:
        pointer = model_dir / "base" / variant / "CURRENT"
        try:
            out["variants"][variant] = pointer.read_text(encoding="utf-8").strip() or None
        except OSError:
            pass
    return out


def probe_customer(conn, seller_id: str) -> str:
    """The latest buyer with a completed order, so the probe shows the model rather than the
    first-visit fallback every new customer gets; a made-up ID when nobody has bought yet."""
    row = conn.execute("SELECT customer_id_local FROM orders WHERE seller_id = ? AND status = 'completed' "
                       "ORDER BY completed_at DESC LIMIT 1", (seller_id,)).fetchone()
    return row["customer_id_local"] if row else "dashboard-probe"
