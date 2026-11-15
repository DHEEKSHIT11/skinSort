"""Unit tests for src/training/evaluate.py on fake data."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.image_model import build_image_model
from src.training.dataset import load_params, load_table
from src.training.evaluate import (calibration_error, evaluate, fit_calibration, pick_tau,
                                   predict_classes, softmax)
from src.training.train_fusion import train_fusion


@pytest.fixture
def params(tmp_path):
    p = load_params()
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(18):
        image_id = f"img{i}"
        Image.fromarray(np.full((64, 64, 3), i * 12, dtype=np.uint8)).save(images / f"{image_id}.jpg")
        row = {"image_id": image_id, "split": ["pad_train", "pad_val", "pad_test"][i // 6],
               "urgency": ["routine", "priority", "urgent"][i % 3], "sex": ["female", "male"][i % 2]}
        for c in p["data"]["answer_columns"]:
            row[c] = 1
            row[f"{c}_missing"] = 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "dataset.csv", index=False)
    p["data"]["dataset_csv"] = str(tmp_path / "dataset.csv")
    p["data"]["images_dir"] = str(images)
    p["train"].update(image_size=32, batch_size=4, num_workers=0)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    return p


def image_checkpoint(params, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_image_model(params, "resnet18", pretrained=False)
    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18",
                "classes": params["data"]["classes"], "epoch": 1,
                "val_metrics": {"urgent_auc": 0.5, "macro_f1": 0.5}}, path)
    return path


def test_pick_tau_reaches_target_with_fewest_flags():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.5, 0.3])
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    tau = pick_tau(p, y, target=0.8)            # need 4 of the 5 urgent cases
    assert tau == 0.6
    assert ((p >= tau) & (y == 1)).sum() / y.sum() >= 0.8


def test_pick_tau_needs_urgent_cases():
    with pytest.raises(ValueError):
        pick_tau(np.array([0.5, 0.4]), np.array([0, 0]), target=0.9)


def test_predict_classes_uses_tau_for_urgent():
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.5, 0.1, 0.4]])
    assert predict_classes(probs, tau=0.35, urgent_idx=2).tolist() == [0, 1, 2]
    assert predict_classes(probs, tau=0.05, urgent_idx=2).tolist() == [2, 2, 2]


def test_calibration_error_zero_when_rates_match():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert calibration_error(p, y) == pytest.approx(0.0)
    assert calibration_error(np.ones(4), np.zeros(4)) == pytest.approx(1.0)


def test_fit_calibration_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3)) * 10       # confident and random: should be softened
    cal = fit_calibration(logits, labels)
    assert cal[0] > 1.0
    assert softmax(logits, cal).max(axis=1).mean() < softmax(logits).max(axis=1).mean()


def test_fit_calibration_undoes_class_shift():
    # informative logits, but every case is pushed towards class 2, as heavy class weights do
    rng = np.random.default_rng(1)
    labels = rng.choice(3, 2000, p=[0.2, 0.7, 0.1])
    logits = np.eye(3)[labels] * 2 + rng.normal(size=(2000, 3)) + np.array([0, 0, 3.0])
    cal = fit_calibration(logits, labels)
    assert 0.2 < cal[0] < 5                        # stays finite, unlike temperature alone
    p = softmax(logits, cal)
    assert abs(p[:, 2].mean() - (labels == 2).mean()) < 0.03
    assert calibration_error(p[:, 2], (labels == 2).astype(int)) < 0.05


def test_evaluate_image_model_writes_outputs(params, tmp_path):
    ckpt = image_checkpoint(params, tmp_path / "resnet18" / "best.pt")
    r = evaluate(params, ckpt, load_table(params, ["pad_val"]), load_table(params, ["pad_test"]),
                 "pad_urgent_recall_target", "img", torch.device("cpu"),
                 extra={"pad_train": load_table(params, ["pad_train"])}, out_dir=tmp_path / "eval")
    out = tmp_path / "eval"
    for f in ("metrics.json", "predictions.csv", "confusion_matrix.png", "calibration.png"):
        assert (out / f).exists()
    assert r["model_kind"] == "image"
    assert r["val"]["urgent_recall"] >= params["evaluate"]["pad_urgent_recall_target"]
    assert json.loads((out / "metrics.json").read_text())["tau"] == r["tau"]
    preds = pd.read_csv(out / "predictions.csv")
    assert set(preds["eval_split"]) == {"val", "test", "pad_train"}
    assert len(preds) == 18 and "sex" in preds and "p_urgent" in preds


def test_evaluate_fusion_model(params, tmp_path):
    ckpt = image_checkpoint(params, tmp_path / "resnet18" / "best.pt")
    train_fusion(params, ckpt, load_table(params, ["pad_train"]), load_table(params, ["pad_val"]),
                 torch.device("cpu"), epochs=1, out_dir=tmp_path / "fusion")
    r = evaluate(params, tmp_path / "fusion" / "best.pt", load_table(params, ["pad_val"]),
                 load_table(params, ["pad_test"]), "pad_urgent_recall_target", "fus",
                 torch.device("cpu"), out_dir=tmp_path / "eval_fus")
    assert r["model_kind"] == "fusion"
    assert set(r["checks"]) == {"urgent_recall", "calibration_error"}


def test_fit_calibration_survives_extreme_logits():
    # a few photos with enormous logits (seen with the batch-8 fine-tune) must not crash
    # the fit or return NaN; temperature stays inside the search grid
    rng = np.random.default_rng(2)
    labels = rng.choice(3, 400, p=[0.2, 0.7, 0.1])
    logits = np.eye(3)[labels] * 3 + rng.normal(size=(400, 3))
    logits[:5] = rng.normal(size=(5, 3)) * 5000
    t, offset = fit_calibration(logits, labels)
    assert 0.05 <= t <= 100 and np.isfinite(offset).all()
    assert np.isfinite(softmax(logits, (t, offset))).all()
