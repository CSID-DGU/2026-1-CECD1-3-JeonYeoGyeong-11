"""Signup/login coverage: password hashing, customer accounts, NTS-gated seller
accounts (mock mode, since no real NTS_SERVICE_KEY is available here), and the
signed session cookie helpers."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from commerce.services.merchant_api import accounts_db as db
from commerce.services.merchant_api import accounts_service as svc
from commerce.services.merchant_api import orders_db
from commerce.services.merchant_api import session
from commerce.services.merchant_api.accounts_service import AccountError

SELLER = "synthetic-seller-1"


class AccountsServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.conn = orders_db.connect(Path(self.tmp.name) / "orders.sqlite")
        db.ensure_schema(self.conn)
        self.addCleanup(self.conn.close)

    # --- customers -------------------------------------------------------------

    def test_customer_can_sign_up_and_log_in(self):
        svc.signup_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1",
                             display_name="홍길동", password="pw1234")
        account = svc.authenticate_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1", password="pw1234")
        self.assertEqual(account["display_name"], "홍길동")

    def test_customer_wrong_password_is_rejected(self):
        svc.signup_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1",
                             display_name="홍길동", password="pw1234")
        with self.assertRaises(AccountError):
            svc.authenticate_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1", password="wrong")

    def test_duplicate_customer_id_is_rejected(self):
        svc.signup_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1",
                             display_name="a", password="pw1234")
        with self.assertRaises(AccountError):
            svc.signup_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1",
                                 display_name="b", password="other-pw")

    def test_password_is_not_stored_in_plaintext(self):
        svc.signup_customer(self.conn, seller_id=SELLER, customer_id_local="cust-1",
                             display_name="a", password="pw1234")
        row = db.fetch_customer(self.conn, SELLER, "cust-1")
        self.assertNotIn("pw1234", row["password_hash"])

    # --- sellers (NTS-gated) -----------------------------------------------------

    def test_seller_signup_passes_in_mock_mode_with_wellformed_fields(self):
        result = svc.signup_seller(
            self.conn, seller_id=SELLER, username="owner-1", display_name="제주농장",
            password="pw1234", business_reg_no="123-45-67890",
            business_open_date="20200101", business_rep_name="홍길동",
        )
        self.assertEqual(result["business_verification_mode"], "mock")
        account = svc.authenticate_seller(self.conn, seller_id=SELLER, username="owner-1", password="pw1234")
        self.assertTrue(account["business_verified"])

    def test_seller_signup_rejects_malformed_business_reg_no(self):
        with self.assertRaises(AccountError):
            svc.signup_seller(
                self.conn, seller_id=SELLER, username="owner-1", display_name="제주농장",
                password="pw1234", business_reg_no="not-a-number",
                business_open_date="20200101", business_rep_name="홍길동",
            )
        self.assertIsNone(db.fetch_seller_account(self.conn, SELLER, "owner-1"))

    def test_seller_signup_calls_real_nts_api_when_service_key_configured(self):
        with mock.patch("commerce.services.merchant_api.nts_client.os.environ.get", return_value="fake-key"), \
             mock.patch("commerce.services.merchant_api.nts_client._real_verify") as mocked:
            from commerce.services.merchant_api.nts_client import VerificationResult
            mocked.return_value = VerificationResult(True, "real", "국세청 진위확인 일치")
            result = svc.signup_seller(
                self.conn, seller_id=SELLER, username="owner-1", display_name="제주농장",
                password="pw1234", business_reg_no="1234567890",
                business_open_date="20200101", business_rep_name="홍길동",
            )
        mocked.assert_called_once()
        self.assertEqual(result["business_verification_mode"], "real")


class SessionTest(unittest.TestCase):
    def test_sign_and_unsign_round_trips(self):
        token = session.sign({"role": "customer", "seller_id": SELLER, "customer_id_local": "cust-1"})
        payload = session.unsign(token)
        self.assertEqual(payload["customer_id_local"], "cust-1")

    def test_tampered_token_is_rejected(self):
        token = session.sign({"role": "customer", "seller_id": SELLER, "customer_id_local": "cust-1"})
        body_b64, _, sig_b64 = token.partition(".")
        tampered = body_b64 + "x." + sig_b64
        self.assertIsNone(session.unsign(tampered))

    def test_garbage_cookie_value_is_rejected(self):
        self.assertIsNone(session.unsign("not-a-valid-token"))
        self.assertIsNone(session.unsign(None))

    def test_expired_token_is_rejected(self):
        with mock.patch("commerce.services.merchant_api.session.time") as mocked_time:
            mocked_time.time.return_value = 1_000_000.0
            token = session.sign({"role": "customer", "seller_id": SELLER, "customer_id_local": "cust-1"})
            mocked_time.time.return_value = 1_000_000.0 + session._MAX_AGE_SECONDS + 1
            self.assertIsNone(session.unsign(token))


if __name__ == "__main__":
    unittest.main()
