"""Tests for config, file and HTTP helpers."""

import zipfile

import pytest
import requests

from conftest import FakeResponse, FakeSession, make_zip, md5, sha256
from scripts.common import config
from scripts.common.files import (
    ArchiveError, extract_images, file_hash, read_manifest, write_manifest,
)
from scripts.common.http_utils import DownloadError, build_session, download_file


# --- config -----------------------------------------------------------------

def test_load_params_has_every_dataset():
    params = config.load_params()
    assert set(params["ingest"]) == {"isic", "ham10000", "pad_ufes_20", "fitzpatrick17k"}


def test_load_params_missing_file(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load_params(tmp_path / "nope.yaml")


def test_load_params_empty_file(tmp_path):
    empty = tmp_path / "params.yaml"
    empty.write_text("# only a comment\n")
    with pytest.raises(config.ConfigError, match="empty"):
        config.load_params(empty)


def test_resolve_path_relative_and_missing():
    p = {"paths": {"raw": "data/raw"}}
    assert config.resolve_path(p, "raw") == config.PIPELINE_ROOT / "data/raw"
    with pytest.raises(config.ConfigError):
        config.resolve_path(p, "processed")


# --- files ------------------------------------------------------------------

def test_file_hash(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"skin")
    assert file_hash(f) == sha256(b"skin")
    assert file_hash(f, "md5") == md5(b"skin")


def test_manifest_roundtrip_sorted(tmp_path):
    path = tmp_path / "manifest.csv"
    rows = [{"file": "b.jpg", "bytes": 2, "sha256": "x", "source_url": "u2"},
            {"file": "a.jpg", "bytes": 1, "sha256": "y", "source_url": "u1"}]
    assert write_manifest(rows, path) == 2
    assert [r["file"] for r in read_manifest(path)] == ["a.jpg", "b.jpg"]


def test_read_manifest_missing_returns_empty(tmp_path):
    assert read_manifest(tmp_path / "none.csv") == []


def test_extract_images_flattens_and_filters(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(make_zip({
        "imgs_part_1/PAT_1.png": b"img1",
        "nested/deeper/ISIC_1.JPG": b"img2",
        "readme.txt": b"not an image",
        "__MACOSX/._PAT_1.png": b"junk",
    }))
    out = extract_images(archive, tmp_path / "images")
    assert sorted(p.name for p in out) == ["ISIC_1.JPG", "PAT_1.png"]


def test_extract_images_blocks_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(make_zip({"../../escape.jpg": b"x"}))
    extract_images(archive, tmp_path / "images")
    assert (tmp_path / "images" / "escape.jpg").exists()
    assert not (tmp_path.parent / "escape.jpg").exists()


def test_extract_images_rejects_non_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(ArchiveError):
        extract_images(bad, tmp_path / "images")


def test_extract_images_rejects_corrupt_member(tmp_path):
    archive = tmp_path / "c.zip"
    archive.write_bytes(make_zip({"a.jpg": b"A" * 1000}))
    data = bytearray(archive.read_bytes())
    offset = data.find(b"A" * 100)
    data[offset:offset + 10] = b"B" * 10  # corrupt the stored bytes, CRC no longer matches
    archive.write_bytes(bytes(data))
    with pytest.raises((ArchiveError, zipfile.BadZipFile)):
        extract_images(archive, tmp_path / "images")


# --- http -------------------------------------------------------------------

def test_build_session_sets_retry_and_agent():
    session = build_session({"max_retries": 3, "user_agent": "test-agent", "timeout_seconds": 9})
    assert session.headers["User-Agent"] == "test-agent"
    assert session.get_adapter("https://x").max_retries.total == 3
    assert session.request_timeout == 9


def test_download_file_ok(tmp_path):
    body = b"image-bytes"
    session = FakeSession({"u": FakeResponse(body=body)})
    record = download_file(session, "u", tmp_path / "x.jpg", md5(body), "md5", len(body))
    assert (tmp_path / "x.jpg").read_bytes() == body
    assert record == {"file": "x.jpg", "bytes": len(body), "sha256": sha256(body),
                      "source_url": "u"}


@pytest.mark.parametrize("kwargs, message", [
    ({"expected_hash": "0" * 32, "hash_algorithm": "md5"}, "md5 mismatch"),
    ({"expected_size": 999}, "Size mismatch"),
])
def test_download_file_verification_failures_leave_no_file(tmp_path, kwargs, message):
    session = FakeSession({"u": FakeResponse(body=b"abc")})
    with pytest.raises(DownloadError, match=message):
        download_file(session, "u", tmp_path / "x.jpg", **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_download_file_http_error_and_empty_body(tmp_path):
    session = FakeSession({"empty": FakeResponse(body=b"")})
    with pytest.raises(DownloadError, match="404"):
        download_file(session, "missing", tmp_path / "x.jpg")
    with pytest.raises(DownloadError, match="Empty"):
        download_file(session, "empty", tmp_path / "y.jpg")
    assert list(tmp_path.iterdir()) == []


def test_download_file_network_error(tmp_path):
    class Broken(FakeSession):
        def get(self, *a, **k):
            raise requests.ConnectionError("connection reset")
    with pytest.raises(DownloadError, match="connection reset"):
        download_file(Broken({}), "u", tmp_path / "x.jpg")
