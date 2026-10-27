"""Tests for bias slicing and mitigation (scripts/validation/bias.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.common.config import resolve_path
from scripts.validation import bias

AGE_BINS = [0, 30, 45, 60, 75, 120]


def make_dataset(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset where urgency (unfairly) depends on sex, so a gap exists."""
    rng = np.random.default_rng(seed)
    split = rng.choice(["train", "val", "test", "pad_train", "pad_val", "pad_test",
                        "fairness_test"], n, p=[.45, .1, .1, .1, .05, .05, .15])
    sex = rng.choice(["female", "male", "unknown"], n, p=[.42, .53, .05])
    age = rng.normal(55, 15, n).round()
    age[rng.random(n) < .15] = np.nan
    on_phone = np.isin(split, ["pad_train", "pad_val", "pad_test", "fairness_test"])
    fitz = np.where(on_phone, rng.integers(1, 7, n).astype(float), np.nan)
    logit = -1.5 + 0.03 * np.nan_to_num(age - 55) + np.where(sex == "male", 0.6, 0)
    urgent = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(n)], "split": split, "sex": sex,
        "body_site": rng.choice(["head_neck", "trunk", "upper_extremity", "acral"], n),
        "age": age, "fitzpatrick": fitz,
        "urgency": np.where(urgent, "urgent", rng.choice(["priority", "routine"], n)),
        "source": np.where(on_phone, "pad_ufes_20", "isic"),
        "brightness": rng.normal(140, 20, n) + urgent * 8, "contrast": rng.normal(40, 8, n),
        "sharpness": rng.gamma(4, 5, n), "width": rng.integers(600, 1200, n),
        "height": rng.integers(450, 900, n),
    })


# --- slices and representation --------------------------------------------------

def test_slices_cover_age_edges_and_skin_tone():
    df = pd.DataFrame({"age": [10, 44, 45, 120, np.nan], "fitzpatrick": [1, 3, 4, 6, np.nan],
                       "urgency": ["urgent", "routine", "priority", "urgent", "routine"]})
    out = bias.add_slices(df, AGE_BINS)
    assert out["age_group"].tolist() == ["0-29", "30-44", "45-59", "unknown", "unknown"]
    assert out["skin_tone"].tolist() == ["light (I-III)", "light (I-III)", "dark (IV-VI)",
                                         "dark (IV-VI)", "unknown"]
    assert out["is_urgent"].tolist() == [1, 0, 0, 1, 0]


def test_representation_counts_shares_and_urgent_rate():
    df = bias.add_slices(pd.DataFrame({
        "split": ["train"] * 4 + ["test"], "sex": ["female", "female", "male", "male", "male"],
        "urgency": ["urgent", "routine", "routine", "routine", "urgent"],
        "age": 50, "fitzpatrick": np.nan}), AGE_BINS)
    rep = bias.representation(df, ["sex"], ["train", "test", "fairness_test"])["sex"]
    assert rep["train"]["female"] == {"images": 2, "share": 0.5}
    assert rep["test"]["male"]["images"] == 1
    assert "fairness_test" not in rep  # empty splits are skipped
    assert rep["urgent_rate_in_train"] == {"female": 0.5, "male": 0.0}


def test_feature_matrix_has_no_missing_values():
    df = make_dataset(200)
    X = bias.feature_matrix(df, age_fill=55.0)
    assert not X.isna().any().any()
    assert {"age_missing", "brightness", "sex_female", "site_acral"} <= set(X.columns)


# --- metrics and mitigation -----------------------------------------------------

def test_slice_metrics_only_judge_groups_with_enough_urgent_cases():
    y_true = pd.Series([1] * 30 + [0] * 10 + [1] * 30 + [1] * 5)
    y_pred = np.array([1] * 30 + [0] * 10 + [1] * 15 + [0] * 15 + [0] * 5)
    groups = pd.Series(["a"] * 40 + ["b"] * 30 + ["c"] * 5)
    m = bias.slice_metrics(y_true, y_pred, groups, min_positives=20)
    assert m["by_group"]["a"]["urgent_recall"] == 1.0
    assert m["by_group"]["b"]["urgent_recall"] == 0.5
    assert m["by_group"]["c"]["judged"] is False  # only 5 urgent cases, recall 0 ignored
    assert m["recall_gap"] == 0.5
    assert (m["lowest_group"], m["highest_group"]) == ("b", "a")


def test_reweighing_makes_group_and_label_independent():
    groups = pd.Series(["a"] * 80 + ["b"] * 20)
    labels = pd.Series([1] * 60 + [0] * 20 + [1] * 5 + [0] * 15)
    w = bias.reweighing_weights(groups, labels)
    joint = pd.crosstab(groups, labels, values=w, aggfunc="sum", normalize=True)
    expected = np.outer(groups.value_counts(normalize=True).sort_index(),
                        labels.value_counts(normalize=True).sort_index())
    assert np.allclose(joint.to_numpy(), expected)
    assert w.mean() == pytest.approx(1.0)


def test_mitigation_compares_three_methods_on_the_same_slice():
    df = bias.add_slices(make_dataset(), AGE_BINS)
    data = {"train": df[df["split"] == "train"],
            "val": df[df["split"].isin(["val", "pad_val"])],
            "eval": df[df["split"].isin(["test", "pad_test"])]}
    result = bias.mitigate(data, "sex", {"min_urgent_cases": 20}, seed=42, age_fill=55.0)
    assert result["feature"] == "sex"
    assert result["baseline"]["recall_gap"] > 0.05  # the planted sex bias is visible
    assert result["reweighing"]["recall_gap"] < result["baseline"]["recall_gap"]
    optimizer = result["threshold_optimizer"]
    assert "error" in optimizer or "overall_urgent_recall" in optimizer


# --- run() end to end -----------------------------------------------------------

@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(bias, "send_alert", lambda title, body: sent.append((title, body)))
    return sent


@pytest.fixture
def dataset(params):
    processed = resolve_path(params, "processed")
    processed.mkdir(parents=True, exist_ok=True)
    make_dataset().to_csv(processed / "dataset.csv", index=False)
    return params


def test_run_writes_reports_weights_and_alerts_on_large_gap(dataset, alerts):
    summary = bias.run(dataset)
    worst = summary["worst_slice"]  # age also drives urgency here, so it may beat sex
    assert worst in bias.DEMOGRAPHIC_SLICES and "sex" in summary["slices_over_target"]

    logs = resolve_path(dataset, "logs")
    report = json.loads((logs / "bias_report.json").read_text())
    assert set(report["performance"]) == set(dataset["bias"]["slices"])
    assert report["mitigation"]["feature"] == worst
    assert f"Mitigation on `{worst}`" in (logs / "bias_report.md").read_text()

    weights = pd.read_csv(resolve_path(dataset, "processed") / "sample_weights.csv")
    assert set(weights["split"]) == {"train", "pad_train"}
    assert weights.groupby("split")["weight"].mean().round(6).tolist() == [1.0, 1.0]

    assert len(alerts) == 1 and "sex" in alerts[0][1] and "continued" in alerts[0][1]


def test_run_respects_alert_switch(dataset, alerts):
    dataset["bias"]["alert_on_gap"] = False
    bias.run(dataset)
    assert alerts == []


def test_run_never_changes_dataset_csv(dataset, alerts):
    path = resolve_path(dataset, "processed") / "dataset.csv"
    before = path.read_bytes()
    bias.run(dataset)
    assert path.read_bytes() == before
