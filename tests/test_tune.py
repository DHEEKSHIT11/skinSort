"""Unit tests for src/training/tune.py and fit_head on fake features."""
import mlflow
import numpy as np
import pytest
import torch

from src.models.fusion_model import build_fusion_head
from src.training.dataset import load_params
from src.training.train_fusion import fit_head
from src.training.tune import cross_validate, run_search, sensitivity_plots, with_trial


def fake_data(n=120, n_features=8, seed=0):
    """Urgent cases have a shifted first image feature, so a head can learn them."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 3
    feats = torch.randn(n, n_features, generator=g)
    feats[:, 0] += 2.0 * (labels == 2)
    answers = torch.randint(0, 2, (n, 12), generator=g).float()
    return feats, answers, labels


@pytest.fixture
def params(tmp_path):
    p = load_params()
    p["train"]["batch_size"] = 16
    p["tuning"].update(cv_folds=3, max_epochs=5, patience=2)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    return p


def test_with_trial_writes_fusion_settings(params):
    trial = {"lr": 0.001, "weight_decay": 0.01, "dropout": 0.1, "urgent_weight": 5.0, "hidden_dim": 32}
    p = with_trial(params, trial)
    assert p["fusion"]["hidden_dim"] == 32 and p["fusion"]["lr"] == 0.001
    assert p["fusion"]["class_weights"] == [1.0, 1.5, 5.0]
    assert params["fusion"]["hidden_dim"] == 64          # original params untouched


def test_fit_head_returns_best_epoch_and_weights(params):
    data = fake_data()
    head = build_fusion_head(params, 8, 12)
    seen = []
    best, state = fit_head(head, data, data, torch.device("cpu"), 2, lr=0.01, weight_decay=0.0,
                           class_weights=[1, 1, 1], batch_size=16, epochs=4, patience=10,
                           on_epoch=lambda e, m, b: seen.append(e))
    assert seen == [1, 2, 3, 4]
    assert best["urgent_auc"] > 0.8
    assert set(state) == set(head.state_dict())


def test_cross_validate_keeps_patients_together(params):
    data = fake_data()
    groups = np.repeat(np.arange(40), 3)                  # 3 photos per patient
    result = cross_validate(with_trial(params, {"lr": 0.01, "weight_decay": 0.0, "dropout": 0.0,
                                                "urgent_weight": 3.0, "hidden_dim": 16}),
                            data, groups, torch.device("cpu"))
    assert 0.5 < result["cv_urgent_auc"] <= 1.0
    assert result["cv_urgent_auc_std"] >= 0


def test_run_search_logs_nested_trials(params, tmp_path):
    data = fake_data()
    groups = np.arange(120)
    with mlflow.start_run(run_name="parent") as parent:
        best, table = run_search(params, data, groups, torch.device("cpu"), n_trials=3)
    assert len(table) == 3 and best["cv_urgent_auc"] == table["cv_urgent_auc"].max()
    children = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(children) == 3
    corr = sensitivity_plots(table, ["lr", "dropout"], tmp_path)
    assert set(corr) == {"lr", "dropout"}
    assert (tmp_path / "sensitivity.png").exists() and (tmp_path / "search_progress.png").exists()


def test_fusion_head_from_checkpoint_uses_saved_size(params):
    from src.models.fusion_model import fusion_head_from_checkpoint
    head = build_fusion_head(with_trial(params, {"lr": 0.001, "weight_decay": 0.0, "dropout": 0.0,
                                                 "urgent_weight": 3.0, "hidden_dim": 32}), 8, 12)
    ckpt = {"state_dict": head.state_dict(), "num_image_features": 8,
            "answer_columns": [f"a{i}" for i in range(12)]}      # old checkpoint: no hidden_dim key
    rebuilt = fusion_head_from_checkpoint(params, ckpt)
    assert rebuilt.answer_branch[0].out_features == 32


def test_lr_sweep_writes_plot(params, tmp_path):
    from src.training.tune import lr_sweep
    table = lr_sweep(params, fake_data(), np.arange(120), torch.device("cpu"), tmp_path, n=3)
    assert len(table) == 3 and (tmp_path / "lr_sweep.png").exists()
