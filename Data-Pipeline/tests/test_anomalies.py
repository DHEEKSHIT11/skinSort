"""Tests for anomaly detection and alerts (scripts/validation/anomalies.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.common.config import resolve_path
from scripts.validation import anomalies


# --- helpers --------------------------------------------------------------------

def make_dataset(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Realistic-looking numbers for the columns the anomaly checks use."""
    rng = np.random.default_rng(seed)
    urgency = np.array(["routine"] * 6 + ["priority"] * 2 + ["urgent"] * 2)
    df = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(n)],
        "split": np.where(np.arange(n) % 5 == 0, "val", "train"),
        "urgency": urgency[np.arange(n) % 10],
        "age": rng.normal(55, 15, n).round(),
        "width": rng.integers(600, 1200, n),
        "height": rng.integers(450, 900, n),
        "brightness": rng.normal(140, 20, n),
        "contrast": rng.normal(40, 8, n),
        "sharpness": rng.gamma(4, 5, n),
    })
    df.loc[df.index % 20 == 0, "age"] = np.nan  # 5% missing, like real metadata
    return df


@pytest.fixture
def cfg(params):
    return params["anomaly"]


@pytest.fixture
def baseline(cfg):
    return anomalies.build_profile(make_dataset(), cfg)


def checks_found(findings, severity=None):
    return {(f["check"], f["target"]) for f in findings
            if severity is None or f["severity"] == severity}


# --- profile --------------------------------------------------------------------

def test_profile_records_spread_fences_and_label_mix(cfg):
    profile = anomalies.build_profile(make_dataset(), cfg)
    assert profile["rows"] == 400
    assert profile["missing_rate"]["age"] == pytest.approx(0.05)
    bright = profile["numeric"]["brightness"]
    assert bright["lower_fence"] < bright["q1"] < bright["q3"] < bright["upper_fence"]
    # every 5th row is val, so train keeps 4 of the 6 routine rows in each block of 10
    assert profile["label_mix"]["train"] == {"routine": 0.5, "priority": 0.25, "urgent": 0.25}
    json.dumps(profile)  # must be storable as the baseline file


def test_profile_skips_columns_that_are_entirely_missing(cfg):
    df = make_dataset()
    df["age"] = np.nan
    assert "age" not in anomalies.build_profile(df, cfg)["numeric"]


# --- checks ---------------------------------------------------------------------

def test_same_data_has_no_findings(baseline, cfg):
    assert anomalies.detect(make_dataset(), baseline, cfg) == []


def test_new_sample_of_same_distribution_is_not_flagged(baseline, cfg):
    """A fresh random sample must not cause false alarms."""
    assert anomalies.detect(make_dataset(seed=1), baseline, cfg) == []


def test_missing_values_spike_is_critical(baseline, cfg):
    df = make_dataset()
    df.loc[: len(df) // 2, "age"] = np.nan  # ~50% missing vs 5% baseline
    assert ("missing", "age") in checks_found(anomalies.detect(df, baseline, cfg), "critical")


def test_small_missing_increase_is_only_a_warning(baseline, cfg):
    df = make_dataset()
    df.loc[1:20, "age"] = np.nan  # +5 points: above warn (2), below critical (10)
    findings = anomalies.detect(df, baseline, cfg)
    assert ("missing", "age") in checks_found(findings, "warn")
    assert ("missing", "age") not in checks_found(findings, "critical")


def test_brightness_drift_is_detected(baseline, cfg):
    df = make_dataset()
    df["brightness"] += 30  # 1.5 standard deviations brighter
    found = checks_found(anomalies.detect(df, baseline, cfg), "critical")
    assert ("drift", "brightness") in found


def test_extreme_values_are_reported_as_outliers(baseline, cfg):
    df = make_dataset()
    df.loc[:59, "sharpness"] = 10_000  # 15% far outside the fences
    found = checks_found(anomalies.detect(df, baseline, cfg), "critical")
    assert ("outliers", "sharpness") in found


def test_urgency_mix_shift_is_detected(baseline, cfg):
    df = make_dataset()
    train = df.index[df["split"] == "train"]
    df.loc[train[:80], "urgency"] = "urgent"  # many routine/priority cases become urgent
    found = checks_found(anomalies.detect(df, baseline, cfg), "critical")
    assert ("label_mix", "train") in found


def test_new_split_is_a_warning(baseline, cfg):
    df = make_dataset()
    df.loc[:9, "split"] = "holdout"
    assert ("label_mix", "holdout") in checks_found(anomalies.detect(df, baseline, cfg), "warn")


def test_row_count_drop_is_critical(baseline, cfg):
    df = make_dataset().iloc[:200]  # half the rows disappeared
    assert ("row_count", "dataset") in checks_found(anomalies.detect(df, baseline, cfg),
                                                    "critical")


def test_constant_baseline_column_does_not_divide_by_zero(cfg):
    df = make_dataset()
    df["width"] = 384
    base = anomalies.build_profile(df, cfg)
    df["width"] = 400
    findings = anomalies.detect(df, base, cfg)
    assert ("drift", "width") not in checks_found(findings)
    assert ("outliers", "width") in checks_found(findings)


# --- run(): report, alerts and stopping -----------------------------------------

@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(anomalies, "send_alert", lambda title, body: sent.append((title, body)))
    return sent


@pytest.fixture
def pipeline(params, tmp_path):
    """Write a dataset and a baseline into the temp folders the params point at."""
    params["anomaly"]["baseline"] = str(tmp_path / "baseline_profile.json")
    processed = resolve_path(params, "processed")
    processed.mkdir(parents=True, exist_ok=True)

    def _write(df):
        df.to_csv(processed / "dataset.csv", index=False)
    _write(make_dataset())
    anomalies.update_baseline(params)
    return _write


def read_report(params):
    return json.loads((resolve_path(params, "logs") / "anomaly_report.json").read_text())


def test_clean_run_writes_report_and_sends_no_alert(params, pipeline, alerts):
    summary = anomalies.run(params)
    assert summary["findings"] == 0 and not summary["stopped"]
    assert read_report(params)["baseline_found"] is True
    assert alerts == []


def test_warning_alerts_but_does_not_stop(params, pipeline, alerts):
    df = make_dataset()
    df.loc[1:20, "age"] = np.nan
    pipeline(df)
    summary = anomalies.run(params)
    assert summary["warn"] >= 1 and summary["critical"] == 0 and not summary["stopped"]
    assert len(alerts) == 1 and "continued" in alerts[0][1]


def test_critical_anomaly_alerts_and_stops_the_run(params, pipeline, alerts):
    df = make_dataset()
    df["brightness"] += 30
    pipeline(df)
    with pytest.raises(anomalies.AnomalyError):
        anomalies.run(params)
    assert read_report(params)["stopped"] is True
    assert "STOPPED" in alerts[0][1] and "brightness" in alerts[0][1]


def test_stop_on_never_only_alerts(params, pipeline, alerts):
    params["anomaly"]["stop_on"] = "never"
    df = make_dataset()
    df["brightness"] += 30
    pipeline(df)
    assert anomalies.run(params)["stopped"] is False
    assert len(alerts) == 1


def test_missing_baseline_alerts_and_skips_checks(params, pipeline, alerts, tmp_path):
    params["anomaly"]["baseline"] = str(tmp_path / "does_not_exist.json")
    summary = anomalies.run(params)
    assert summary["findings"] == 0
    assert "no anomaly baseline" in alerts[0][0]
    assert read_report(params)["baseline_found"] is False


def test_alert_text_is_capped_for_long_lists():
    findings = [{"severity": "warn", "message": f"issue {i}"} for i in range(20)]
    text = anomalies.format_alert(findings, stopped=False)
    assert "issue 14" in text and "issue 15" not in text and "5 more" in text
