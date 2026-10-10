"""국세청 사업자등록정보 진위확인 API (공공데이터포털, data.go.kr) client.

Real endpoint: POST https://api.odcloud.kr/api/nts-businessman/v1/validate?serviceKey=...
Request:  {"businesses": [{"b_no": b_no, "start_dt": start_dt, "p_nm": p_nm}]}
Response: {"status_code": "OK", "data": [{"valid": "01" | "02", ...}]}  -- "01" = matched.

A real serviceKey (from data.go.kr, "사업자등록정보 진위확인 및 상태조회 서비스") is
required for real verification. Without one (NTS_SERVICE_KEY unset), this falls
back to a local mock check -- same "mock now, swap to real later" shape as
orders_service.get_recommendations_for_display. The real-mode code path is
implemented against the published API contract but has not been exercised
against the live endpoint in this environment (no key available here).
"""
from __future__ import annotations

import os
import re
from typing import NamedTuple, Optional

_VALIDATE_URL = "https://api.odcloud.kr/api/nts-businessman/v1/validate"
_B_NO_PATTERN = re.compile(r"^\d{10}$")


class VerificationResult(NamedTuple):
    verified: bool
    mode: str  # "real" | "mock"
    detail: str


def _mock_verify(b_no: str, start_dt: str, p_nm: str) -> VerificationResult:
    """Format-only check: well-formed 10-digit b_no, non-empty name/date.
    Never claims a real NTS match. Clearly labeled mode="mock" so callers can
    show a "실제 국세청 조회 아님" notice."""
    if not _B_NO_PATTERN.match(b_no):
        return VerificationResult(False, "mock", "사업자등록번호는 숫자 10자리여야 합니다.")
    if not start_dt.strip() or not p_nm.strip():
        return VerificationResult(False, "mock", "개업일자·대표자성명을 입력하세요.")
    return VerificationResult(True, "mock", "데모 모드: 형식만 확인했습니다 (실제 국세청 조회 아님).")


def _real_verify(b_no: str, start_dt: str, p_nm: str, service_key: str) -> VerificationResult:
    import httpx  # already a project dependency (fastapi test client)

    try:
        response = httpx.post(
            _VALIDATE_URL, params={"serviceKey": service_key},
            json={"businesses": [{"b_no": b_no, "start_dt": start_dt, "p_nm": p_nm}]},
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return VerificationResult(False, "real", "국세청 API 호출 실패: %s" % exc)

    data = (payload.get("data") or [{}])[0]
    if data.get("valid") == "01":
        return VerificationResult(True, "real", "국세청 진위확인 일치")
    return VerificationResult(False, "real", data.get("valid_msg") or "국세청 진위확인 불일치")


def verify_business_registration(
    b_no: str, start_dt: str, p_nm: str, *, service_key: Optional[str] = None,
) -> VerificationResult:
    key = service_key if service_key is not None else os.environ.get("NTS_SERVICE_KEY")
    b_no_digits = re.sub(r"\D", "", b_no)
    if key:
        return _real_verify(b_no_digits, start_dt, p_nm, key)
    return _mock_verify(b_no_digits, start_dt, p_nm)
