"""Tests for the schema and statistics step (scripts/validation/schema.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.common.config import load_yaml, resolve_path
from scripts.validation import schema


# --- helpers --------------------------------------------------------------------

def valid_dataset(n: int = 20) -> pd.DataFrame:
    """A small dataset.csv that satisfies every schema rule."""
    ids = [f"ISIC_{i:07d}" for i in range(n)]
    df = pd.DataFrame({
        "image_id": ["isic__" + i for i in ids],
        "source": "isic",
        "source_image_id": ids,
        "image_path": ["isic/images/" + i + ".jpg" for i in ids],
        "image_type": "dermoscopic",
        "patient_id": np.nan,
        "lesion_id": [f"L{i}" for i in range(n)],
        "diagnosis_raw": "Benign | Benign melanocytic proliferations | Nevus",
        "diagnosis": "nevus",
        "urgency": "routine",
        "age": 50.0,
        "sex": ["female", "male"] * (n // 2) + ["female"] * (n % 2),
        "body_site": "trunk",
        "fitzpatrick": np.nan,
        "license": "CC-0",
        "readable": True,
        "error": np.nan,
        "width": 600,
        "height": 450,
        "brightness": 120.0,
        "contrast": 40.0,
        "sharpness": 15.0,
        "phash": [f"{i:016x}" for i in range(n)],
        "group_id": [f"isic/L{i}" for i in range(n)],
        "split": "train",
    })
    for column in ["q_itch", "q_grew", "q_hurt", "q_changed", "q_bleed", "q_elevation"]:
        df[column] = np.nan
    df["processed_image"] = "images/" + df["image_id"] + ".jpg"
    return df


@pytest.fixture
def small_params(params):
    """Real params.yaml with the row-count limits loosened for tiny test tables."""
    params["validation"]["min_rows"] = 1
    params["validation"]["max_rows"] = 1000
    return params


def validate(df, params):
    urgency = load_yaml(resolve_path(params, "urgency_mapping"))
    return schema.validate_dataframe(df, schema.build_expectations(params, urgency))


def failed_rules(report):
    return {(c["rule"], c["column"]) for c in report["failed"]}


# --- rules ----------------------------------------------------------------------

def test_required_columns_include_labels_stats_and_split():
    columns = schema.required_columns()
    for column in ["image_id", "urgency", "brightness", "phash", "group_id", "split"]:
        assert column in columns
    assert len(columns) == len(set(columns))


def test_valid_dataset_passes(small_params):
    report = validate(valid_dataset(), small_params)
    assert report["success"], report["failed"]
    assert report["rules_failed"] == 0
    assert report["rules_checked"] == len(report["checks"]) > 20


def test_unknown_category_fails(small_params):
    df = valid_dataset()
    df.loc[0, "urgency"] = "very_urgent"
    df.loc[1, "split"] = "holdout"
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_be_in_set", "urgency") in failed
    assert ("expect_column_values_to_be_in_set", "split") in failed


def test_diagnosis_missing_from_urgency_mapping_fails(small_params):
    df = valid_dataset()
    df.loc[0, "diagnosis"] = "not_a_real_diagnosis"
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_be_in_set", "diagnosis") in failed


def test_duplicate_image_id_fails(small_params):
    df = valid_dataset()
    df.loc[1, "image_id"] = df.loc[0, "image_id"]
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_be_unique", "image_id") in failed


def test_missing_urgency_fails(small_params):
    df = valid_dataset()
    df.loc[0, "urgency"] = np.nan
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_not_be_null", "urgency") in failed


def test_age_out_of_range_fails_but_missing_age_is_allowed(small_params):
    df = valid_dataset()
    df.loc[0, "age"] = 150
    df.loc[1, "age"] = np.nan  # 1 of 20 missing = 5%, under the 25% limit
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_be_between", "age") in failed
    assert ("expect_column_values_to_not_be_null", "age") not in failed


def test_too_many_missing_ages_fails(small_params):
    df = valid_dataset()
    df.loc[:9, "age"] = np.nan  # 50% missing, limit is 25%
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_not_be_null", "age") in failed


def test_image_quality_limits_come_from_params(small_params):
    df = valid_dataset()
    df.loc[0, "brightness"] = small_params["preprocess"]["quality"]["min_brightness"] - 1
    df.loc[1, "width"] = small_params["preprocess"]["quality"]["min_side_px"] - 1
    df.loc[2, "readable"] = False
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_be_between", "brightness") in failed
    assert ("expect_column_values_to_be_between", "width") in failed
    assert ("expect_column_values_to_be_in_set", "readable") in failed


def test_bad_answer_and_phash_formats_fail(small_params):
    df = valid_dataset()
    df.loc[0, "q_itch"] = 2
    df.loc[1, "phash"] = "not-a-hash"
    failed = failed_rules(validate(df, small_params))
    assert ("expect_column_values_to_be_in_set", "q_itch") in failed
    assert ("expect_column_values_to_match_regex", "phash") in failed


def test_missing_column_fails(small_params):
    df = valid_dataset().drop(columns=["group_id"])
    report = validate(df, small_params)
    assert not report["success"]
    assert ("expect_table_columns_to_match_set", None) in failed_rules(report)


def test_row_count_outside_limits_fails(params):
    report = validate(valid_dataset(), params)  # real limits: at least 20,000 rows
    assert ("expect_table_row_count_to_be_between", None) in failed_rules(report)


# --- statistics -----------------------------------------------------------------

def test_statistics_summarise_counts_missing_and_ranges():
    df = valid_dataset()
    df.loc[0, "age"] = np.nan
    stats = schema.compute_statistics(df)
    assert stats["rows"] == 20
    assert stats["missing_rate"]["age"] == 0.05
    assert stats["numeric"]["brightness"]["mean"] == 120.0
    assert stats["numeric"]["fitzpatrick"]["mean"] is None  # all missing, no crash
    assert stats["categorical"]["sex"] == {"female": 10, "male": 10}
    assert stats["urgency_by_split"] == {"train": {"routine": 20}}
    json.dumps(stats)  # must be writable as JSON


# --- run() end to end -----------------------------------------------------------

def write_dataset(params, df):
    processed = resolve_path(params, "processed")
    processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed / "dataset.csv", index=False)


def test_run_writes_both_reports_on_success(small_params):
    write_dataset(small_params, valid_dataset())
    summary = schema.run(small_params)
    assert summary == {"rows": 20, "rules_checked": summary["rules_checked"], "rules_failed": 0}
    logs = resolve_path(small_params, "logs")
    assert json.loads((logs / "schema_report.json").read_text())["success"] is True
    assert json.loads((logs / "data_statistics.json").read_text())["rows"] == 20


def test_run_raises_and_still_writes_report_on_failure(small_params):
    df = valid_dataset()
    df.loc[0, "source"] = "unknown_dataset"
    write_dataset(small_params, df)
    with pytest.raises(schema.SchemaValidationError, match="source"):
        schema.run(small_params)
    report = json.loads((resolve_path(small_params, "logs") / "schema_report.json").read_text())
    assert report["success"] is False
    source_check = next(c for c in report["failed"] if c["column"] == "source")
    assert source_check["unexpected_examples"] == ["unknown_dataset"]
