"""Tests for the four dataset download modules, with all HTTP faked."""

from pathlib import Path

import pandas as pd
import pytest

from conftest import FakeResponse, make_zip, md5, sha256
from scripts.common.http_utils import DownloadError
from scripts.ingest import fitzpatrick17k, ham10000, isic, pad_ufes


# --- ISIC -------------------------------------------------------------------

def _isic_record(isic_id, url):
    return {"isic_id": isic_id, "copyright_license": "CC-0", "attribution": "x",
            "files": {"full": {"url": url, "size": 1}},
            "metadata": {"acquisition": {"image_type": "dermoscopic"},
                         "clinical": {"sex": "female", "age_approx": 50,
                                      "diagnosis_1": "Benign"}}}


@pytest.fixture
def isic_params(params):
    cfg = params["ingest"]["isic"]
    cfg["api_url"] = "api"
    cfg["page_size"] = 2
    cfg["strata"] = [{"name": "melanoma", "query": "q1", "max_images": 3},
                     {"name": "benign", "query": "q2", "max_images": 1}]
    return params


def test_flatten_record():
    row = isic.flatten_record(_isic_record("ISIC_1", "u1"), "benign", "full")
    assert row["isic_id"] == "ISIC_1" and row["stratum"] == "benign"
    assert row["image_url"] == "u1" and row["sex"] == "female" and row["age_approx"] == 50


def test_isic_paginates_caps_and_dedupes(isic_params, fake_session):
    pages = {
        "q1": FakeResponse(json_data={"results": [_isic_record("ISIC_1", "i1"),
                                                  _isic_record("ISIC_2", "i2")],
                                      "next": "page2"}),
        "q2": FakeResponse(json_data={"results": [_isic_record("ISIC_2", "i2")], "next": None}),
    }
    routes = {
        "api": lambda p: pages[p["query"]],
        "page2": FakeResponse(json_data={"results": [_isic_record("ISIC_3", "i3"),
                                                     _isic_record("ISIC_4", "i4")],
                                         "next": "page3"}),
        "i1": FakeResponse(body=b"a"), "i2": FakeResponse(body=b"b"),
        "i3": FakeResponse(body=b"c"),
    }
    session = fake_session(isic, routes)
    summary = isic.run(isic_params)

    # melanoma capped at 3 (page3 never requested); ISIC_2 listed twice but kept once
    assert "page3" not in [url for url, _ in session.calls]
    assert summary["listed"] == 3 and summary["images"] == 3 and summary["failed"] == 0
    meta = pd.read_csv(isic_params["paths"]["raw"] + "/isic/metadata.csv")
    assert list(meta["isic_id"]) == ["ISIC_1", "ISIC_2", "ISIC_3"]


def test_isic_skips_already_downloaded(isic_params, fake_session):
    routes = {"api": FakeResponse(json_data={"results": [_isic_record("ISIC_1", "i1")],
                                             "next": None}),
              "i1": FakeResponse(body=b"a")}
    session = fake_session(isic, routes)
    isic.run(isic_params)
    session.calls.clear()
    isic.run(isic_params)
    assert "i1" not in [url for url, _ in session.calls]


def test_isic_fails_when_too_many_images_fail(isic_params, fake_session):
    routes = {"api": FakeResponse(json_data={"results": [_isic_record("ISIC_1", "dead")],
                                             "next": None})}
    fake_session(isic, routes)
    with pytest.raises(DownloadError, match="failed"):
        isic.run(isic_params)


def test_isic_empty_search_fails(isic_params, fake_session):
    fake_session(isic, {"api": FakeResponse(json_data={"results": [], "next": None})})
    with pytest.raises(DownloadError, match="no images"):
        isic.run(isic_params)


# --- HAM10000 ---------------------------------------------------------------

def _ham_routes(zip_bytes, checksum=None):
    base = "https://dataverse.harvard.edu"
    files = [
        {"dataFile": {"id": 1, "filename": "HAM10000_metadata.tab",
                      "checksum": {"type": "MD5", "value": "ignored-for-tab"}}},
        {"dataFile": {"id": 2, "filename": "HAM10000_images_part_1.zip",
                      "checksum": {"type": "MD5", "value": checksum or md5(zip_bytes)}}},
        {"dataFile": {"id": 3, "filename": "HAM10000_images_part_2.zip",
                      "checksum": {"type": "MD5", "value": md5(zip_bytes)}}},
    ]
    return {
        f"{base}/api/datasets/:persistentId/":
            FakeResponse(json_data={"data": {"latestVersion": {"files": files}}}),
        f"{base}/api/access/datafile/1?format=original":
            FakeResponse(body=b"lesion_id,image_id,dx\nHAM_1,ISIC_1,mel\n"),
        f"{base}/api/access/datafile/2": FakeResponse(body=zip_bytes),
        f"{base}/api/access/datafile/3": FakeResponse(body=zip_bytes),
    }


def test_ham10000_downloads_extracts_and_cleans_up(params, fake_session):
    zip_bytes = make_zip({"ISIC_1.jpg": b"x", "ISIC_2.jpg": b"y"})
    fake_session(ham10000, _ham_routes(zip_bytes))
    summary = ham10000.run(params)
    out = params["paths"]["raw"] + "/ham10000"
    assert summary["images"] == 2
    assert (pd.read_csv(out + "/metadata.csv")["dx"] == "mel").all()
    assert not list(Path(out).glob("*.zip"))  # archives removed after extraction


def test_ham10000_checksum_mismatch_fails(params, fake_session):
    zip_bytes = make_zip({"ISIC_1.jpg": b"x"})
    fake_session(ham10000, _ham_routes(zip_bytes, checksum="0" * 32))
    with pytest.raises(DownloadError, match="mismatch"):
        ham10000.run(params)


def test_ham10000_missing_remote_file_fails(params, fake_session):
    routes = _ham_routes(make_zip({"a.jpg": b"x"}))
    key = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
    routes[key] = FakeResponse(json_data={"data": {"latestVersion": {"files": []}}})
    fake_session(ham10000, routes)
    with pytest.raises(DownloadError, match="not found"):
        ham10000.run(params)


# --- PAD-UFES-20 --------------------------------------------------------------

def test_pad_ufes_lists_folders_and_verifies_sha256(params, fake_session):
    meta = b"patient_id,img_id,itch\nPAT_1,PAT_1_1_1.png,True\n"
    zip_bytes = make_zip({"imgs_part_1/PAT_1_1_1.png": b"png"})
    base = "https://data.mendeley.com/public-api/datasets/zr7vgbcyr2"

    def entry(name, body, url):
        return {"filename": name, "size": len(body),
                "content_details": {"sha256_hash": sha256(body), "download_url": url}}

    def files(p):
        listing = {"root": [entry("metadata.csv", meta, "m")],
                   "img": [entry("imgs_part_1.zip", zip_bytes, "z")]}
        return FakeResponse(json_data=listing[p["folder_id"]])

    fake_session(pad_ufes, {
        f"{base}/folders/1": FakeResponse(json_data=[{"id": "img", "name": "images"}]),
        f"{base}/files": files,
        "m": FakeResponse(body=meta), "z": FakeResponse(body=zip_bytes),
    })
    summary = pad_ufes.run(params)
    assert summary == {"dataset": "pad_ufes_20", "files": 2, "images": 1}


# --- Fitzpatrick17k ---------------------------------------------------------

def _fitz_csv(rows):
    header = "md5hash,fitzpatrick_scale,label,url\n"
    return (header + "".join(f"{h},{s},{lbl},{u}\n" for h, s, lbl, u in rows)).encode()


def test_fitzpatrick_tolerates_dead_links_and_verifies_md5(params, fake_session):
    good, wrong = b"good-image", b"tampered"
    rows = [(md5(good), 3, "psoriasis", "ok"),
            (md5(b"expected"), 5, "eczema", "tampered"),
            ("deadbeef" * 4, 6, "acne", "dead"),
            (md5(good), 3, "psoriasis", "ok")]  # duplicate row is dropped
    fake_session(fitzpatrick17k, {
        params["ingest"]["fitzpatrick17k"]["csv_url"]: FakeResponse(body=_fitz_csv(rows)),
        "ok": FakeResponse(body=good), "tampered": FakeResponse(body=wrong),
    })
    params["ingest"]["fitzpatrick17k"]["min_success_rate"] = 0.3
    summary = fitzpatrick17k.run(params)
    assert summary["listed"] == 3 and summary["images"] == 1 and summary["failed"] == 2
    failures = pd.read_csv(params["paths"]["raw"] + "/fitzpatrick17k/failures.csv")
    assert set(failures["url"]) == {"tampered", "dead"}


def test_fitzpatrick_fails_below_min_success_rate(params, fake_session):
    rows = [("deadbeef" * 4, 6, "acne", "dead")]
    fake_session(fitzpatrick17k, {
        params["ingest"]["fitzpatrick17k"]["csv_url"]: FakeResponse(body=_fitz_csv(rows)),
    })
    with pytest.raises(DownloadError, match="Only 0.0%"):
        fitzpatrick17k.run(params)


def test_fitzpatrick_missing_columns_fails(params, fake_session):
    fake_session(fitzpatrick17k, {
        params["ingest"]["fitzpatrick17k"]["csv_url"]: FakeResponse(body=b"md5hash,url\na,b\n"),
    })
    with pytest.raises(DownloadError, match="missing columns"):
        fitzpatrick17k.run(params)


def test_fitzpatrick_skips_links_that_were_404_last_time(params, fake_session):
    good = b"good-image"
    rows = [(md5(good), 3, "psoriasis", "ok"), ("deadbeef" * 4, 6, "acne", "dead")]
    csv_url = params["ingest"]["fitzpatrick17k"]["csv_url"]
    params["ingest"]["fitzpatrick17k"]["min_success_rate"] = 0.3
    routes = {csv_url: FakeResponse(body=_fitz_csv(rows)), "ok": FakeResponse(body=good)}
    session = fake_session(fitzpatrick17k, routes)
    fitzpatrick17k.run(params)                    # "dead" -> 404, recorded in failures.csv
    session.calls.clear()
    summary = fitzpatrick17k.run(params)
    requested = [url for url, _ in session.calls]
    assert "dead" not in requested and "ok" not in requested   # both answered from cache
    assert summary["failed"] == 1 and summary["images"] == 1


def test_fitzpatrick_retries_non_404_failures(params, fake_session):
    rows = [("deadbeef" * 4, 6, "acne", "flaky")]
    csv_url = params["ingest"]["fitzpatrick17k"]["csv_url"]
    params["ingest"]["fitzpatrick17k"]["min_success_rate"] = 0.0
    session = fake_session(fitzpatrick17k, {csv_url: FakeResponse(body=_fitz_csv(rows)),
                                            "flaky": FakeResponse(status=503)})
    fitzpatrick17k.run(params)
    session.calls.clear()
    fitzpatrick17k.run(params)
    assert "flaky" in [url for url, _ in session.calls]
