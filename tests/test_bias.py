"""Unit tests for src/training/bias.py on fake predictions."""
import json
import math

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.training.bias import (add_slice_columns, apply_thresholds, fit_group_thresholds,
                               mitigate, patient_half, run_bias_check, slice_report,
                               wilson_interval)
from src.training.dataset import load_params


def fake_predictions(n_per_group=40, recall_a=1.0, recall_b=0.5, split="test"):
    """Two sexes, all images urgent; the model catches recall_a of A and recall_b of B."""
    rows = []
    for sex, recall in (("female", recall_a), ("male", recall_b)):
        for i in range(n_per_group):
            caught = i < round(recall * n_per_group)
            p_urgent = 0.9 if caught else 0.3 + 0.001 * i
            rows.append({"image_id": f"{sex}{i}{split}", "group_id": f"{sex}{i}{split}",
                         "eval_split": split, "label": 2, "pred": 2 if caught else 0,
                         "p_routine": 1 - p_urgent, "p_priority": 0.0, "p_urgent": p_urgent,
                         "sex": sex, "age": 20 + i, "body_site": "trunk", "source": "isic",
                         "fitzpatrick": np.nan})
    return pd.DataFrame(rows)


@pytest.fixture
def params(tmp_path):
    p = load_params()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    return p


def test_add_slice_columns_maps_age_and_skin_tone():
    df = pd.DataFrame({"age": [10, 50, np.nan], "fitzpatrick": [1, 5, np.nan],
                       "sex": ["male", None, "female"], "body_site": ["trunk"] * 3,
                       "source": ["isic"] * 3})
    out = add_slice_columns(df)
    assert out["age_group"].tolist() == ["0-30", "45-60", "unknown"]
    assert out["skin_tone"].tolist() == ["I-II (light)", "V-VI (dark)", "unknown"]
    assert out["sex"].tolist() == ["male", "unknown", "female"]


def test_slice_report_flags_recall_gap():
    df = add_slice_columns(fake_predictions(recall_a=1.0, recall_b=0.5))
    report = slice_report(df, "sex", min_urgent=20, max_gap=0.05)
    assert report["status"] == "FAIL"
    assert report["recall_gap"] == pytest.approx(0.5)
    assert report["worst_group"] == "male"


def test_slice_report_ignores_small_groups():
    df = add_slice_columns(fake_predictions(n_per_group=10))
    report = slice_report(df, "sex", min_urgent=20, max_gap=0.05)
    assert report["status"] == "TOO_FEW_CASES"
    assert math.isnan(report["recall_gap"])


def test_run_bias_check_inconclusive_when_groups_small(params, tmp_path):
    summary = run_bias_check(params, fake_predictions(n_per_group=10), "small",
                             out_dir=tmp_path / "bias")
    assert summary["result"] == "INCONCLUSIVE" and summary["passed"]


def test_slice_report_skips_single_group_slices():
    df = add_slice_columns(fake_predictions())
    assert slice_report(df, "source", min_urgent=20, max_gap=0.05) is None


def test_wilson_interval_wider_for_fewer_cases():
    small = wilson_interval(15, 20)
    large = wilson_interval(150, 200)
    assert small[0] < 0.75 < small[1]
    assert small[1] - small[0] > large[1] - large[0]


def test_run_bias_check_writes_report(params, tmp_path):
    preds = pd.concat([fake_predictions(split="val"), fake_predictions()])
    summary = run_bias_check(params, preds, "fake", out_dir=tmp_path / "bias")
    assert not summary["passed"] and summary["result"] == "FAIL"
    assert "test/sex" in summary["failed_slices"]
    assert "val" not in summary["splits"]          # val rows are only for mitigation
    saved = json.loads((tmp_path / "bias" / "bias_report.json").read_text())
    assert saved["failed_slices"] == summary["failed_slices"]
    assert (tmp_path / "bias" / "test_sex.png").exists()


def test_group_thresholds_close_the_gap():
    val = add_slice_columns(fake_predictions(split="val"))
    test = add_slice_columns(fake_predictions())
    taus = fit_group_thresholds(val, "sex", target=0.9, min_urgent=20, default_tau=0.5)
    assert taus["female"] == 0.5                   # never above the global threshold
    assert taus["male"] < 0.5
    after = apply_thresholds(test, "sex", taus, default_tau=0.5)
    male = after[after["sex"] == "male"]
    assert (male["pred"] == 2).mean() >= 0.9


def test_mitigate_reports_before_and_after():
    preds = add_slice_columns(pd.concat([fake_predictions(split="val"), fake_predictions()]))
    [result] = mitigate(preds, "sex", target=0.9, global_tau=0.5, min_urgent=20, max_gap=0.05)
    assert result["fitted_on"] == "val"
    assert result["before"]["status"] == "FAIL"
    assert result["after"]["recall_gap"] < result["before"]["recall_gap"]
    assert result["after"]["flagged_share"] >= result["before"]["flagged_share"]


def test_patient_half_keeps_groups_together():
    df = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "group_id": ["g1", "g1", "g2", None]})
    half = patient_half(df)
    assert half[0] == half[1]
    assert set(half.unique()) <= {0, 1}


def test_mitigate_cross_fits_out_of_fold_rows():
    preds = add_slice_columns(fake_predictions(n_per_group=80, split="val"))
    assert mitigate(preds, "sex", target=0.9, global_tau=0.5, min_urgent=20, max_gap=0.05) == []
    [r] = mitigate(preds, "sex", target=0.9, global_tau=0.5, min_urgent=20, max_gap=0.05, include_val=True)
    assert r["split"] == "val" and "cross-fitted" in r["fitted_on"] and len(r["group_taus_per_fold"]) == 2
    assert r["before"]["status"] == "FAIL" and r["after"]["recall_gap"] < r["before"]["recall_gap"]
    assert r["group_taus"]["male"] < 0.5
