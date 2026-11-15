"""API test: register a tiny model, export it, and call the FastAPI app."""
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.serving import app as serving
from src.training.registry import export, register
from tests.test_registry import example, make_model, params  # noqa: F401  (fixture)


def photo_bytes():
    buf = io.BytesIO()
    Image.fromarray(np.full((50, 50, 3), 120, dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_triage_endpoint(params, tmp_path, monkeypatch):  # noqa: F811
    ckpt, ev = make_model(params, tmp_path, "v1", auc=0.80)
    register(params, ckpt, ev, image_example=example(tmp_path))
    export(params, tmp_path / "exported")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "exported"))
    serving._state.clear()
    client = TestClient(serving.app)
    assert client.get("/health").json()["model"]["version"] == 1
    r = client.post("/triage", files={"photo": ("a.jpg", photo_bytes(), "image/jpeg")},
                    data={"q_itch": "yes", "q_grew": "no", "q_bleed": ""})
    body = r.json()
    assert r.status_code == 200, body
    assert r.status_code == 200 and body["triage"] in {"routine", "priority", "urgent"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert set(body["unanswered"]) == {"q_hurt", "q_changed", "q_bleed", "q_elevation"}
    assert client.post("/triage", files={"photo": ("a.jpg", b"nope", "image/jpeg")}).status_code == 400
    assert client.post("/triage", files={"photo": ("a.jpg", photo_bytes(), "image/jpeg")},
                       data={"q_itch": "maybe"}).status_code == 422
