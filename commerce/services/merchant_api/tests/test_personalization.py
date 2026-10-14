"""Seller-started personalization (OQ08): runs both variants on the job
executor, reports the outcome on the comparison screen, refuses to overlap."""
from __future__ import annotations

import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import PersonalizationResult
from commerce.services.merchant_api import personalization
from commerce.services.merchant_api.context import MerchantSettings, build_context
from commerce.services.merchant_api.main import create_app
from commerce.services.merchant_api.tests.fakes import FakeRecommenderRuntime

SELLER = "seller-p"


class PersonalizingRuntime(FakeRecommenderRuntime):
    def __init__(self, seller_id, gate=None):
        super().__init__(seller_id)
        self.gate, self.calls = gate, []

    def get_local_data_ref(self):
        return "fs.3.abc"

    def personalize_local(self, local_data_ref, personal_config, *, model_variant="text_relation"):
        if self.gate:
            self.gate.wait(5)
        self.calls.append((local_data_ref, model_variant))
        status = "installed" if model_variant == "text_relation" else "rejected"
        return PersonalizationResult(status, None if status == "installed" else "validation_rejected",
                                     "r-r500", "rev1" if status == "installed" else None)


class PersonalizationScreenTest(unittest.TestCase):
    def _client(self, runtime):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        settings = MerchantSettings(seller_id=SELLER, feature_db_path=root / "f.sqlite", model_dir=root / "m",
                                    merchant_db_path=root / "orders.sqlite")
        client = TestClient(create_app(settings, context_factory=lambda s: build_context(s, runtime_factory=lambda *a: runtime)),
                            follow_redirects=False)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        personalization._last.pop(SELLER, None)
        client.post(f"/seller/{SELLER}/signup", data={
            "username": "owner", "display_name": "매장", "password": "pw-1234",
            "business_reg_no": "1234567890", "business_open_date": "20200101", "business_rep_name": "홍길동"})
        return client

    def _press(self, client):
        page = client.get(f"/seller/{SELLER}/compare").text
        token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
        return client.post(f"/seller/{SELLER}/personalize", data={"csrf_token": token})

    def _wait_done(self):
        for _ in range(100):
            run = personalization.last_run(SELLER)
            if run and run["state"] != "running":
                return run
            time.sleep(0.05)
        self.fail("personalization did not finish")

    def test_runs_both_variants_on_one_snapshot_and_shows_the_outcome(self):
        runtime = PersonalizingRuntime(SELLER)
        client = self._client(runtime)
        self.assertEqual(self._press(client).status_code, 303)
        run = self._wait_done()
        self.assertEqual(runtime.calls, [("fs.3.abc", "text_only"), ("fs.3.abc", "text_relation")])
        self.assertEqual(run["results"]["text_relation"]["status"], "installed")
        page = client.get(f"/seller/{SELLER}/compare").text
        self.assertIn("적용됨", page)
        self.assertIn("검증 미달로 미적용", page)

    def test_a_variant_without_a_base_shows_no_model_and_the_other_still_runs(self):
        # B's #49 review: only text_relation installed (as C's fl_demo does) -> text_only NOT_FOUND.
        class OneBase(PersonalizingRuntime):
            def personalize_local(self, local_data_ref, personal_config, *, model_variant="text_relation"):
                if model_variant == "text_only":
                    raise ContractError("NOT_FOUND", "/model_variant")
                return super().personalize_local(local_data_ref, personal_config, model_variant=model_variant)
        runtime = OneBase(SELLER)
        client = self._client(runtime)
        self._press(client)
        run = self._wait_done()
        self.assertIsNone(run["error"])
        self.assertEqual(run["results"]["text_only"]["reason"], "model_not_ready")
        self.assertEqual(run["results"]["text_relation"]["status"], "installed")
        page = client.get(f"/seller/{SELLER}/compare").text
        self.assertIn("모델 없음", page)
        self.assertIn("적용됨", page)

    def test_other_contract_errors_still_stop_the_run(self):
        class Forbidden(PersonalizingRuntime):
            def personalize_local(self, *a, **k):
                raise ContractError("FORBIDDEN", "/seller_id")
        client = self._client(Forbidden(SELLER))
        self._press(client)
        self.assertIn("FORBIDDEN", self._wait_done()["error"])

    def test_stub_runtime_says_not_connected(self):
        client = self._client(FakeRecommenderRuntime(SELLER))  # personalize_local raises FeatureNotImplemented
        self._press(client)
        self.assertEqual(self._wait_done()["error"], "not_connected")
        self.assertIn("연결되지 않아 개인화를", client.get(f"/seller/{SELLER}/compare").text)

    def test_a_second_press_while_running_is_refused(self):
        gate = threading.Event()
        client = self._client(PersonalizingRuntime(SELLER, gate))
        self._press(client)
        r = self._press(client)
        self.assertIn("p=busy", r.headers["location"])
        gate.set()
        self._wait_done()

    def test_requires_csrf(self):
        client = self._client(PersonalizingRuntime(SELLER))
        self.assertEqual(client.post(f"/seller/{SELLER}/personalize").status_code, 403)


if __name__ == "__main__":
    unittest.main()
