"""c1: seller bearer tokens. The auth file keeps work-factor hashes only."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from commerce.services.fl_coordinator import auth


class SellerTokens(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "fl" / "text_only" / "auth.json"

    def test_token_names_its_seller_and_only_a_hash_is_stored(self):
        token = auth.issue(self.path, "seller-a")
        self.assertTrue(token.startswith("seller-a."))
        secret = token.split(".", 1)[1]
        stored = self.path.read_text(encoding="utf-8")
        self.assertNotIn(secret, stored)
        self.assertIn('"scrypt"', stored)
        self.assertEqual(auth.SellerAuth(self.path).seller_for("Bearer " + token), "seller-a")

    def test_wrong_or_borrowed_or_malformed_tokens_resolve_to_nobody(self):
        token_a = auth.issue(self.path, "seller-a")
        token_b = auth.issue(self.path, "seller-b")
        checker = auth.SellerAuth(self.path)
        secret_b = token_b.split(".", 1)[1]
        for header in (None, "", "Bearer", "Bearer ", "Basic " + token_a, "Bearer seller-a.wrong",
                       "Bearer seller-a." + secret_b, "Bearer seller-z." + secret_b, "Bearer " + token_a + " x",
                       "Bearer ../x.y", "Bearer seller-a"):
            self.assertIsNone(checker.seller_for(header), header)
        self.assertEqual(checker.seller_for("bearer " + token_b), "seller-b")

    def test_rotation_and_revocation(self):
        old = auth.issue(self.path, "seller-a")
        new = auth.issue(self.path, "seller-a")
        checker = auth.SellerAuth(self.path)
        self.assertIsNone(checker.seller_for("Bearer " + old))
        self.assertEqual(checker.seller_for("Bearer " + new), "seller-a")
        self.assertTrue(auth.revoke(self.path, "seller-a"))
        self.assertIsNone(auth.SellerAuth(self.path).seller_for("Bearer " + new))
        self.assertFalse(auth.revoke(self.path, "seller-a"))

    def test_cli_prints_the_token_once_and_refuses_a_directory(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(auth.main(["issue", "--auth-file", str(self.path), "--seller", "seller-a"]), 0)
        token = out.getvalue().strip()
        self.assertEqual(auth.SellerAuth(self.path).seller_for("Bearer " + token), "seller-a")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(auth.main(["issue", "--auth-file", str(self.path.parent), "--seller", "seller-b"]), 1)
            self.assertEqual(auth.main(["issue", "--auth-file", str(self.path), "--seller", "../x"]), 1)


if __name__ == "__main__":
    unittest.main()
