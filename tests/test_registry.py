"""Unit tests for src/training/registry.py with tiny fake models and a temp MLflow DB."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.fusion_model import build_fusion_head
from src.models.image_model import build_image_model
from src.training.dataset import answer_feature_columns, load_params
from src.training.registry import gate, register, rollback, status


def metrics(auc, cal_ok=True, recall_ok=False, test_auc=0.80):
    """auc = validation urgent AUC (decides promotion); test values are report-only."""
    return {"tau": 0.07, "recall_target": 0.85,
            "val": {"urgent_auc": auc, "calibration_error": 0.03 if cal_ok else 0.09},
            "test": {"urgent_auc": test_auc, "urgent_recall": 0.74, "calibration_error": 0.03},
            "checks": {"urgent_recall": {"passed": recall_ok}, "calibration_error": {"passed": True}}}


@pytest.fixture
def params(tmp_path):
    p = load_params()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    return p


def make_model(params, tmp_path, name, auc, cal_ok=True):
    image_ckpt = tmp_path / "img.pt"
    if not image_ckpt.exists():
        model = build_image_model(params, "resnet18", pretrained=False)
        torch.save({"state_dict": model.state_dict(), "backbone": "resnet18",
                    "classes": params["data"]["classes"], "epoch": 1, "val_metrics": {}}, image_ckpt)
    head = build_fusion_head(params, 512, 12)
    folder = tmp_path / name
    folder.mkdir()
    torch.save({"state_dict": head.state_dict(), "backbone": "resnet18", "image_checkpoint": str(image_ckpt),
                "num_image_features": 512, "answer_columns": answer_feature_columns(params),
                "hidden_dim": params["fusion"]["hidden_dim"], "classes": params["data"]["classes"],
                "temperature": 1.5, "class_offset": [0.1, 0.0, -0.1], "tau": 0.3}, folder / "best.pt")
    (folder / "eval").mkdir()
    (folder / "eval" / "metrics.json").write_text(json.dumps(metrics(auc, cal_ok)))
    return folder / "best.pt", folder / "eval"


def example(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"photo{i}.jpg"
        Image.fromarray(np.full((40, 40, 3), 60 * i, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return pd.DataFrame({"image_path": paths, "q_itch": [1, np.nan], "q_grew": [0, 1], "q_hurt": [1, 0],
                         "q_changed": [np.nan, 1], "q_bleed": [0, 0], "q_elevation": [1, 1]})


def test_gate_rules(params):
    cfg = params["registry"]
    assert gate(metrics(0.80), None, cfg)["promote"]
    assert not gate(metrics(0.80, cal_ok=False), None, cfg)["promote"]    # validation calibration
    assert not gate(metrics(0.70), None, cfg)["promote"]                  # below min_val_urgent_auc
    assert not gate(metrics(0.76), metrics(0.80), cfg)["promote"]          # worse than Production
    ok = gate(metrics(0.795), metrics(0.80), cfg)                          # within tolerance
    assert ok["promote"] and "test_urgent_recall not met" in ok["warnings"]
    # a much better TEST score cannot rescue a worse validation score
    assert not gate(metrics(0.76, test_auc=0.99), metrics(0.80), cfg)["promote"]


def test_register_promote_reject_and_rollback(params, tmp_path):
    ckpt1, eval1 = make_model(params, tmp_path, "v1", auc=0.80)
    r1 = register(params, ckpt1, eval1, image_example=example(tmp_path))
    assert r1["promote"] and r1["version"] == 1

    ckpt2, eval2 = make_model(params, tmp_path, "v2", auc=0.76)            # clearly worse
    r2 = register(params, ckpt2, eval2)
    assert not r2["promote"]

    ckpt3, eval3 = make_model(params, tmp_path, "v3", auc=0.82)
    assert register(params, ckpt3, eval3)["promote"]
    rows = {row["version"]: row for row in status(params)}
    assert "production" in rows[3]["aliases"] and "previous" in rows[1]["aliases"]
    assert rows[2]["promotion"].startswith("rejected")

    r = rollback(params)
    assert r == {"name": params["mlflow"]["registered_model_name"], "production": 1, "rolled_back": 3}


def test_registered_model_predicts(params, tmp_path):
    ckpt, ev = make_model(params, tmp_path, "v1", auc=0.80)
    register(params, ckpt, ev, image_example=example(tmp_path))
    model = mlflow.pyfunc.load_model(f"models:/{params['mlflow']['registered_model_name']}@production")
    out = model.predict(example(tmp_path))
    assert list(out.columns) == ["p_routine", "p_priority", "p_urgent", "urgent", "triage"]
    assert np.allclose(out[["p_routine", "p_priority", "p_urgent"]].sum(axis=1), 1.0)
    assert ((out["p_urgent"] >= 0.3) == out["urgent"]).all()


class FakeBucket:
    def __init__(self):
        self.uploaded = {}

    def get(self, key):
        return self.uploaded.get(key)

    def list_blobs(self, prefix, max_results=None):
        return [k for k in self.uploaded if k.startswith(prefix)][:max_results]

    def blob(self, key):
        bucket = self

        class Blob:
            def upload_from_filename(self, path):
                bucket.uploaded[key] = open(path, "rb").read()

            def upload_from_string(self, text, content_type=None):
                bucket.uploaded[key] = text

            def exists(self):
                return key in bucket.uploaded

            def download_as_text(self):
                return bucket.uploaded[key]
        return Blob()


class FakeClient:
    def __init__(self):
        self.b = FakeBucket()

    def bucket(self, name):
        return self.b


def test_push_to_gcs_versions_and_never_overwrites(params, tmp_path):
    from src.training.registry import push_to_gcs
    ckpt, ev = make_model(params, tmp_path, "v1", auc=0.80)
    register(params, ckpt, ev, image_example=example(tmp_path))
    params["registry"]["gcs_uri"] = "gs://fake-bucket/models/triage"
    dry = push_to_gcs(params, dry_run=True)
    assert dry["dry_run"] and dry["files"] > 0 and dry["gcs_path"] == "gs://fake-bucket/models/triage/v1/"
    client = FakeClient()
    push_to_gcs(params, client=client, record=tmp_path / "gcs_push.json")
    keys = client.b.uploaded
    assert "models/triage/v1/model_info.json" in keys and "models/triage/latest.json" in keys
    assert keys["models/triage/version.txt"] == "1"
    assert any(k.endswith("MLmodel") for k in keys)
    # a fresh-registry CI run pushes again: no collision
    second = push_to_gcs(params, client=client, record=tmp_path / "gcs_push.json")
    assert second["gcs_version"] == 2 and keys["models/triage/version.txt"] == "2"
    assert json.loads(keys["models/triage/latest.json"])["gcs_version"] == 2
    from src.training.registry import rollback_gcs
    assert rollback_gcs(params, client=client) == {"deployed": 1, "rolled_back_from": 2}
    assert json.loads(keys["models/triage/latest.json"])["gcs_version"] == 1
    with pytest.raises(SystemExit):
        rollback_gcs(params, to=7, client=client)


def test_compare_deployed_blocks_worse_model(params, tmp_path):
    from src.training.registry import compare_deployed, push_to_gcs
    params["registry"]["gcs_uri"] = "gs://fake-bucket/models/triage"
    client = FakeClient()
    _, ev_new = make_model(params, tmp_path, "new", auc=0.76)   # validation AUC below the deployed 0.80
    assert compare_deployed(params, ev_new, client=client)["promote"]          # nothing deployed yet
    ckpt, ev = make_model(params, tmp_path, "v1", auc=0.80)
    register(params, ckpt, ev, image_example=example(tmp_path))
    push_to_gcs(params, client=client, record=tmp_path / "gcs_push.json")
    d = compare_deployed(params, ev_new, client=client)
    assert not d["promote"] and d["deployed_version"] == 1
