"""HTTP client for the coordinator's synthetic round routes (see the coordinator README).

Takes an httpx.Client whose base_url is the coordinator and the seller's bearer token.
Nothing here decides whether a seller may send plaintext: FLClient.start does, and only
on attested synthetic input (D0025).
"""
from __future__ import annotations

import json

import httpx

from commerce.packages.contracts.types import Payload
from commerce.services.fl_coordinator.npz_payload import MAX_PAYLOAD_BYTES, PayloadTooLarge


class CoordinatorRefused(RuntimeError):
    """A non-success response. `code` is the contract_error code when the body has one."""

    def __init__(self, status: int, code: str | None):
        super().__init__("coordinator refused: %s %s" % (status, code or ""))
        self.status, self.code = status, code


class CoordinatorTransport:
    def __init__(self, client: httpx.Client, token: str):
        self._client = client
        self._headers = {"Authorization": "Bearer " + token}

    def _call(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self._client.request(method, path, headers=dict(self._headers, **kwargs.pop("headers", {})),
                                        **kwargs)
        if response.status_code >= 400:
            try:
                code = response.json().get("code")
            except (ValueError, AttributeError):
                code = None
            raise CoordinatorRefused(response.status_code, code)
        return response

    def current_round(self) -> Payload | None:
        response = self._call("GET", "/rounds/current")
        return None if response.status_code == 204 else response.json()

    def latest(self) -> Payload | None:
        try:
            return self._call("GET", "/models/latest").json()
        except CoordinatorRefused as exc:
            if exc.status == 404:
                return None
            raise

    def manifest(self, model_version: str) -> Payload:
        return self._call("GET", "/models/%s/manifest" % model_version).json()

    def weights(self, model_version: str) -> bytes:
        with self._client.stream("GET", "/models/%s/weights" % model_version, headers=self._headers) as response:
            if response.status_code >= 400:
                raise CoordinatorRefused(response.status_code, None)
            chunks, total = [], 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_PAYLOAD_BYTES:
                    raise PayloadTooLarge("release exceeds the transfer limit")
                chunks.append(chunk)
        return b"".join(chunks)

    def submit(self, round_id: str, submission: Payload, payload: bytes) -> Payload:
        declared = self._call("POST", "/rounds/%s/submissions" % round_id, content=json.dumps(submission),
                              headers={"Content-Type": "application/json"})
        location = declared.headers["Location"]
        if location != "/rounds/%s/submissions/delta" % round_id:  # never follow an arbitrary URL
            raise CoordinatorRefused(declared.status_code, None)
        return self._call("PUT", location, content=payload,
                          headers={"Content-Type": "application/octet-stream"}).json()

    def result(self, round_id: str) -> Payload:
        return self._call("GET", "/rounds/%s/result" % round_id).json()
