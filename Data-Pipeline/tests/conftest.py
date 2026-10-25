"""Shared test fixtures. No test touches the network: HTTP is faked."""

import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import requests

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.common.config import load_params  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, body=b"", json_data=None):
        self.status_code = status
        self._body = body if json_data is None else json.dumps(json_data).encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")

    def json(self):
        return json.loads(self._body)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Maps URL -> FakeResponse (or callable(params) -> FakeResponse)."""

    def __init__(self, routes):
        self.routes = routes
        self.request_timeout = 5
        self.calls = []

    def get(self, url, params=None, **_kwargs):
        self.calls.append((url, params))
        route = self.routes.get(url)
        if route is None:
            return FakeResponse(status=404)
        return route(params) if callable(route) else route


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def params(tmp_path):
    """Real params.yaml, with every output redirected to a temp folder."""
    p = copy.deepcopy(load_params())
    p["paths"].update({k: str(tmp_path / k) for k in ("raw", "interim", "processed", "logs")})
    return p


@pytest.fixture
def fake_session(monkeypatch):
    """Patch build_session in a module so it returns a FakeSession with given routes."""
    def _install(module, routes):
        session = FakeSession(routes)
        monkeypatch.setattr(module, "build_session", lambda *a, **k: session)
        return session
    return _install
