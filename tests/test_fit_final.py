"""Unit tests for src/training/fit_final.py on fake features."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from src.training.dataset import load_params
from src.training.fit_final import fit_final, out_of_fold, train_head
from src.training.train_fusion import fit_head
from src.training.tune import with_trial


def fake(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 3
    feats = torch.randn(n, 8, generator=g)
    feats[:, 0] += 2.0 * (labels == 2)
    return feats, torch.randint(0, 2, (n, 12), generator=g).float(), labels


def frame(n, split):
    return pd.DataFrame({"image_id": [f"{split}{i}" for i in range(n)],
                         "patient_id": [f"{split}p{i // 2}" for i in range(n)],
                         "split": split, "sex": "female", "age": 50.0, "body_site": "trunk",
                         "source": "pad_ufes_20", "fitzpatrick": 2.0})


@pytest.fixture
def params(tmp_path):
    p = load_params()
    p["train"]["batch_size"] = 16
    p["tuning"].update(cv_folds=3, max_epochs=6, patience=2)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    return with_trial(p, {"lr": 0.01, "weight_decay": 0.0, "dropout": 0.0, "urgent_weight": 3.0,
                          "hidden_dim": 16})


def test_fit_head_fixed_epochs_runs_all_epochs(params):
    data = fake(60, 0)
    seen = []
    best, _ = fit_head(train_head(params, data, 1, 0), data, data, torch.device("cpu"), 2, lr=0.01,
                       weight_decay=0.0, class_weights=[1, 1, 1], batch_size=16, epochs=5, patience=1,
                       on_epoch=lambda e, m, b: seen.append(e), select_best=False)
    assert seen == [1, 2, 3, 4, 5] and best["epoch"] == 5


def test_out_of_fold_scores_every_row(params):
    data = fake(90, 1)
    groups = np.repeat(np.arange(45), 2)
    cv = out_of_fold(params, data, groups)
    assert cv["oof_logits"].shape == (90, 3)
    assert not np.all(cv["oof_logits"] == 0, axis=1).any()
    assert 1 <= cv["epochs"] <= params["tuning"]["max_epochs"]


def test_fit_final_saves_tau_in_checkpoint(params, tmp_path):
    pooled_df = pd.concat([frame(60, "pad_train"), frame(30, "pad_val")], ignore_index=True)
    pooled = tuple(torch.cat(t) for t in zip(fake(60, 2), fake(30, 3)))
    test_df, test = frame(30, "pad_test"), fake(30, 4)
    image_ckpt = tmp_path / "img.pt"
    torch.save({"backbone": "resnet18"}, image_ckpt)
    r = fit_final(params, image_ckpt, pooled_df, pooled, test_df, test, "oof", "params", 0.85,
                  models_dir=tmp_path / "models", eval_dir=tmp_path / "eval")
    assert r["val"]["urgent_recall"] >= 0.85                 # tau reaches the target out-of-fold
    ckpt = torch.load(tmp_path / "models" / "oof" / "best.pt", weights_only=False)
    assert ckpt["tau"] == r["tau"] and len(ckpt["class_offset"]) == 3
    preds = pd.read_csv(tmp_path / "eval" / "oof" / "predictions.csv")
    assert set(preds["eval_split"]) == {"val", "test"} and len(preds) == 120
    assert json.loads((tmp_path / "eval" / "oof" / "metrics.json").read_text())["tau"] == r["tau"]


def test_reweighing_matches_data_pipeline_formula():
    import sys
    from src.training.dataset import ROOT
    from src.training.fit_final import reweighing_weights
    groups = np.array(["f", "f", "f", "m", "m", "m", "m", "u"])
    urgent = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    w = reweighing_weights(groups, urgent).numpy()
    # weighted data: group and label independent -> weighted urgent share equal per group (f, m)
    share = {g: w[(groups == g) & (urgent == 1)].sum() / w[groups == g].sum() for g in ("f", "m")}
    assert abs(share["f"] - share["m"]) < 1e-6
    dp = ROOT / "Data-Pipeline"
    if (dp / "scripts").exists():
        sys.path.insert(0, str(dp))
        try:
            from scripts.validation.bias import reweighing_weights as dp_weights
        finally:
            sys.path.remove(str(dp))
        assert np.allclose(w, dp_weights(pd.Series(groups), pd.Series(urgent)).to_numpy())


def test_fit_final_with_reweighing(params, tmp_path):
    pooled_df = pd.concat([frame(60, "pad_train"), frame(30, "pad_val")], ignore_index=True)
    pooled_df["sex"] = ["female", "male"] * 45
    pooled = tuple(torch.cat(t) for t in zip(fake(60, 2), fake(30, 3)))
    image_ckpt = tmp_path / "img.pt"
    torch.save({"backbone": "resnet18"}, image_ckpt)
    r = fit_final(params, image_ckpt, pooled_df, pooled, frame(30, "pad_test"), fake(30, 4), "rw", "params",
                  0.85, models_dir=tmp_path / "models", eval_dir=tmp_path / "eval", reweight=["sex"])
    assert r["reweight"] == ["sex"] and r["val"]["urgent_recall"] >= 0.85
