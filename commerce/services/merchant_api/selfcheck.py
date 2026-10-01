"""a1 gate (working-agreement.md §4): order state machine over real HTTP,
other-seller rejection, evaluation metrics hand-checkable example.

Exit codes follow gate.py's convention: 0 all done, 3 partial (remaining items
printed), 1 failure, 2 the check itself is broken.

Deliberately NOT covered here (remaining, pending open human decisions):
- Caller authentication / cookie scope (OQ13, OQ15): no route in main.py checks
  who is allowed to act as a given seller_id yet.
- g2 (real B runtime wired in): B's runtime is still UnimplementedRuntime; this
  selfcheck injects its own FakeRecommenderRuntime test double instead.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import FakeRecommenderRuntime


def _context_factory(settings: MerchantSettings):
    return build_context(settings, runtime_factory=lambda sid, feat, model: FakeRecommenderRuntime(sid))


def _check_order_round_trip_over_http(root: Path) -> None:
    settings = MerchantSettings(seller_id="a1-seller", feature_db_path=root / "features.sqlite",
                                model_dir=root / "models", merchant_db_path=root / "orders.sqlite")
    with TestClient(create_app(settings, context_factory=_context_factory)) as client:
        create = client.post("/sellers/a1-seller/orders", json={
            "customer_id_local": "cust-1", "idempotency_key": "a1-key-1", "currency": "KRW",
            "items": [{"item_id_local": "sku-milk", "quantity": 2, "unit_price_minor": 3200}],
        })
        assert create.status_code == 201, create.text
        order = create.json()
        assert order["status"] == "requested" and order["status_version"] == 1, order

        accept = client.post(f"/sellers/a1-seller/orders/{order['order_id']}/accept",
                              json={"expected_status_version": 1})
        assert accept.status_code == 200 and accept.json()["status"] == "accepted", accept.text

        complete = client.post(f"/sellers/a1-seller/orders/{order['order_id']}/complete",
                                json={"expected_status_version": 2})
        assert complete.status_code == 200 and complete.json()["status"] == "completed", complete.text

        stale = client.post(f"/sellers/a1-seller/orders/{order['order_id']}/cancel",
                             json={"expected_status_version": 1})
        assert stale.status_code == 409 and stale.json()["code"] == "STALE_STATUS_VERSION", stale.text

        illegal = client.post(f"/sellers/a1-seller/orders/{order['order_id']}/cancel",
                               json={"expected_status_version": 3})
        assert illegal.status_code == 409 and illegal.json()["code"] == "ILLEGAL_STATE_TRANSITION", illegal.text

        other_seller = client.get(f"/sellers/some-other-seller/orders/{order['order_id']}")
        assert other_seller.status_code == 404, other_seller.text

        catalog = client.post("/sellers/a1-seller/catalog-items",
                               json={"item_id_local": "sku-milk", "title_text": "우유 1L"})
        assert catalog.status_code == 201 and catalog.json()["schema_version"] == "catalog_item.v1", catalog.text


def _check_metrics_hand_example() -> None:
    from commerce.evaluation.metrics.ranking import ndcg_at_k, recall_at_k
    ranked = [("a", 3.0), ("b", 2.0), ("c", 2.0), ("d", 1.0)]
    relevant = {"b", "d"}
    recall = recall_at_k(ranked, relevant, 2)
    ndcg = ndcg_at_k(ranked, relevant, 2)
    # See commerce/evaluation/metrics/tests/test_ranking.py for the derivation.
    assert abs(recall - 0.25) < 1e-9, recall
    assert abs(ndcg - 0.19343) < 1e-4, ndcg


def _check_unit_tests(verbose: bool) -> unittest.TestResult:
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTests(loader.loadTestsFromName("commerce.services.merchant_api.tests.test_orders"))
    suite.addTests(loader.loadTestsFromName("commerce.evaluation.metrics.tests.test_ranking"))
    return unittest.TextTestRunner(verbosity=2 if verbose else 0).run(suite)


REMAINING = [
    "쿠키/세션 인증 범위(OQ13, OQ15 결정 대기 — 현재 라우트는 호출자 인증이 없음)",
    "B 실제 추천 연결(g2) — FakeRecommenderRuntime 더블로만 검증됨",
]


def run(verbose: bool = False) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _check_order_round_trip_over_http(Path(tmp))
            _check_metrics_hand_example()
        except AssertionError as exc:
            sys.stderr.write("a1 FAIL: %s\n" % exc)
            return 1
        except Exception as exc:  # the check itself is broken
            sys.stderr.write("a1 BROKEN: %r\n" % exc)
            return 2

    result = _check_unit_tests(verbose)
    if not result.wasSuccessful():
        return 1

    print("a1: 주문 상태전이/타 판매자 거부/카탈로그 upsert(HTTP) 통과, 평가지표 손계산 예제 통과, "
          "%d개 단위 테스트 통과." % result.testsRun)
    for item in REMAINING:
        print("a1 남은 항목: %s" % item)
    return 3


if __name__ == "__main__":
    raise SystemExit(run(verbose=True))
