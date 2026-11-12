"""Unit tests for src/training/select_model.py selection rules and candidate scoring."""
import json

import numpy as np
import pandas as pd
import pytest

from src.training import select_model
from src.training.dataset import load_params
from src.training.select_model import candidate_summary, choose


def table(rows):
    df = pd.DataFrame(rows, columns=["name", "kind", "val_urgent_auc", "val_calibration_error",
                                     "tau_urgent_cases", "bias_result"])
    df["servable"] = df["kind"] == "fusion"
    df["tau_fitted_on"], df["test_urgent_auc"], df["test_urgent_recall"] = "x", 0.7, 0.8
    return df.sort_values("val_urgent_auc", ascending=False).reset_index(drop=True)


@pytest.fixture
def params():
    return load_params()


def test_choose_filters_small_tau_sets_and_baselines(params):
    t = table([("tuned", "fusion", 0.87, 0.02, 31, "INCONCLUSIVE"),
               ("oof", "fusion", 0.81, 0.03, 209, "FAIL"),
               ("answers", "baseline", 0.90, 0.03, 209, "PASS")])
    d = choose(t, params)
    assert d["selected"] == "oof" and "bias warning" in d["reason"]


def test_choose_prefers_fair_model_within_tolerance(params):
    t = table([("a", "fusion", 0.82, 0.03, 209, "FAIL"),
               ("b", "fusion", 0.81, 0.03, 209, "PASS")])
    assert choose(t, params)["selected"] == "b"


def test_choose_prefers_fewer_failing_slices(params):
    t = table([("a", "fusion", 0.813, 0.03, 209, "FAIL"), ("b", "fusion", 0.807, 0.03, 209, "FAIL")])
    t["bias_failed_slices"] = [3, 1]
    d = choose(t, params)
    assert d["selected"] == "b" and "fewest failing" in d["reason"]


def test_choose_ignores_fair_model_far_behind(params):
    t = table([("a", "fusion", 0.85, 0.03, 209, "FAIL"),
               ("b", "fusion", 0.70, 0.03, 209, "PASS")])
    assert choose(t, params)["selected"] == "a"


def test_choose_rejects_bad_calibration(params):
    t = table([("a", "fusion", 0.85, 0.20, 209, "PASS")])
    d = choose(t, params)
    assert d["selected"] is None


def test_candidate_summary_scores_pad_val_rows(params, tmp_path, monkeypatch):
    pad_val = pd.DataFrame({"image_id": [f"v{i}" for i in range(40)]})
    monkeypatch.setattr(select_model, "load_table", lambda p, s: pad_val)
    rng = np.random.default_rng(0)
    rows = []
    for split, ids in (("val", [f"v{i}" for i in range(40)] + [f"t{i}" for i in range(60)]),
                       ("test", [f"x{i}" for i in range(20)])):
        for i, image_id in enumerate(ids):
            label = 2 if i % 4 == 0 else 0
            p = 0.7 if label == 2 else rng.uniform(0, 0.5)
            rows.append({"image_id": image_id, "eval_split": split, "label": label, "p_urgent": p,
                         "pred": 2 if p >= 0.5 else 0, "sex": "female", "age": 50, "body_site": "trunk",
                         "source": "pad_ufes_20", "fitzpatrick": 2})
    folder = tmp_path / "cand"
    folder.mkdir()
    pd.DataFrame(rows).to_csv(folder / "predictions.csv", index=False)
    (folder / "metrics.json").write_text(json.dumps(
        {"model_kind": "fusion", "checkpoint": "ck.pt",
         "test": {"urgent_auc": 0.7, "urgent_recall": 0.8, "calibration_error": 0.04}}))
    s = candidate_summary(params, "cand", eval_dir=tmp_path)
    assert s["tau_urgent_cases"] == 25 and s["tau_fitted_on"].startswith("out-of-fold")
    assert s["val_urgent_auc"] == pytest.approx(1.0) and s["val_urgent_recall"] == 1.0


def test_select_backbone_falls_back_to_checkpoints_on_empty_mlflow(params, tmp_path):
    import mlflow
    import torch
    from src.training.select_model import select_backbone
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    for name, auc in (("resnet50", 0.90), ("efficientnet_b3", 0.93)):
        (tmp_path / "models" / name).mkdir(parents=True)
        torch.save({"val_metrics": {"urgent_auc": auc, "urgent_recall": 0.8, "macro_f1": 0.7}},
                   tmp_path / "models" / name / "best.pt")
    b = select_backbone(params, models_dir=tmp_path / "models")
    assert list(b["backbone"]) == ["efficientnet_b3", "resnet50"] and set(b["source"]) == {"checkpoint"}
