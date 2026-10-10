"""Seller bearer tokens for one variant's coordinator (AUTH_FILE). Only work-factor hashes are stored.

    python -m commerce.services.fl_coordinator.auth issue --auth-file <AUTH_FILE> --seller <seller_id>
    python -m commerce.services.fl_coordinator.auth revoke --auth-file <AUTH_FILE> --seller <seller_id>

A token is `<seller_id>.<secret>`, so the seller is read from the token and matched
against the request body (interfaces.md §6). The file keeps a scrypt hash per seller
and never the token; `issue` prints the token once and replaces any earlier one.
A verified token is remembered in memory by its SHA-256 so that every request does
not pay the scrypt cost; nothing about tokens is written anywhere else.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
from pathlib import Path

from commerce.packages.contracts import ids
from commerce.services.fl_coordinator.round_core import SAFE_ID, atomic_write

KDF = {"name": "scrypt", "n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def _derive(secret: str, salt: bytes, kdf: dict) -> bytes:
    if kdf.get("name") != "scrypt":
        raise ValueError("unsupported kdf")
    return hashlib.scrypt(secret.encode("utf-8"), salt=salt, n=kdf["n"], r=kdf["r"], p=kdf["p"],
                          dklen=kdf["dklen"], maxmem=64 * 1024 * 1024)


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes()) if path.exists() else {"kdf": dict(KDF), "sellers": {}}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, ids.canonical_json(data))
    if os.name == "posix":
        os.chmod(path, 0o600)


def issue(path: Path | str, seller_id: str) -> str:
    """Create or rotate the token of `seller_id`; return it. Only its hash is stored."""
    if not SAFE_ID.fullmatch(seller_id):
        raise ValueError("unsafe seller id")
    path = Path(path)
    data = _read(path)
    secret = secrets.token_urlsafe(32)  # no "." in the alphabet
    salt = secrets.token_bytes(16)
    data["sellers"][seller_id] = {"salt": salt.hex(), "hash": _derive(secret, salt, data["kdf"]).hex()}
    _write(path, data)
    return "%s.%s" % (seller_id, secret)


def revoke(path: Path | str, seller_id: str) -> bool:
    path = Path(path)
    data = _read(path)
    removed = data["sellers"].pop(seller_id, None) is not None
    if removed:
        _write(path, data)
    return removed


class SellerAuth:
    """Resolve an Authorization header to a seller id, or None. Loaded once at coordinator start."""

    def __init__(self, path: Path | str):
        data = _read(Path(path))
        self._kdf = data["kdf"]
        self._sellers = {seller: (bytes.fromhex(entry["salt"]), bytes.fromhex(entry["hash"]))
                         for seller, entry in data["sellers"].items()}
        self._verified: dict[bytes, str] = {}

    def seller_for(self, authorization: str | None) -> str | None:
        if not authorization:
            return None
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token or " " in token.strip():
            return None
        token = token.strip()
        seller_id, dot, secret = token.rpartition(".")
        if not dot or not secret or not SAFE_ID.fullmatch(seller_id) or seller_id not in self._sellers:
            return None
        fingerprint = hashlib.sha256(token.encode("utf-8")).digest()
        if self._verified.get(fingerprint) == seller_id:
            return seller_id
        salt, expected = self._sellers[seller_id]
        if not hmac.compare_digest(_derive(secret, salt, self._kdf), expected):
            return None
        if len(self._verified) < 1024:
            self._verified[fingerprint] = seller_id
        return seller_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Issue or revoke a seller token for one variant's coordinator")
    parser.add_argument("action", choices=("issue", "revoke"))
    parser.add_argument("--auth-file", required=True, help="AUTH_FILE (a file, not a directory)")
    parser.add_argument("--seller", required=True)
    args = parser.parse_args(argv)
    if Path(args.auth_file).is_dir():
        print("AUTH_FILE must be a file", file=sys.stderr)
        return 1
    if args.action == "issue":
        try:
            print(issue(args.auth_file, args.seller))  # shown once; give it to that seller only
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    return 0 if revoke(args.auth_file, args.seller) else 1


if __name__ == "__main__":
    raise SystemExit(main())
