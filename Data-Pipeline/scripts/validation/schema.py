"""Validate dataset.csv against the SkinSort schema and record its statistics.

Input : data/processed/dataset.csv
Output: logs/schema_report.json     (every rule, passed or failed)
        logs/data_statistics.json   (counts, missing rates, ranges, label mix)

The rules are Great Expectations "expectations". Every limit comes from
params.yaml or configs/, so changing a threshold never means editing code.
If any rule fails, both files are still written and the step raises
SchemaValidationError, which stops the Airflow run and fires the alert hook.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from scripts.common.config import load_params, load_yaml, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger
from scripts.preprocess.harmonize import ANSWER_COLUMNS, COMMON_COLUMNS
from scripts.preprocess.image_stats import STATS_COLUMNS
from scripts.preprocess.transform import BODY_SITES, SEXES

# Great Expectations sends anonymous usage data by default; turn that off
# before it is imported so pipeline runs never call out to the internet.
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402

logger = get_logger("validation.schema")

SOURCES = ["isic", "ham10000", "pad_ufes_20", "fitzpatrick17k"]
SPLITS = ["train", "val", "test", "pad_train", "pad_val", "pad_test", "fairness_test"]
URGENCY_TIERS = ["urgent", "priority", "routine"]
# Columns added after harmonize/image_stats: by dedupe (group_id) and split.
EXTRA_COLUMNS = ["group_id", "split", "processed_image"]
NUMERIC_STATS_COLUMNS = ["age", "fitzpatrick", "width", "height",
                         "brightness", "contrast", "sharpness"]
CATEGORICAL_STATS_COLUMNS = ["source", "split", "urgency", "diagnosis", "sex", "body_site"]


class SchemaValidationError(Exception):
    """Raised when dataset.csv breaks one or more schema rules."""


def required_columns() -> List[str]:
    """Every column the training code relies on, taken from the pipeline's own constants."""
    stats = [c for c in STATS_COLUMNS if c != "image_id"]
    return list(dict.fromkeys(COMMON_COLUMNS + stats + EXTRA_COLUMNS))


def build_expectations(params: Dict[str, Any], urgency_cfg: Dict[str, Any]) -> list:
    """Turn the schema rules into a list of Great Expectations expectations."""
    cfg = params["validation"]
    quality = params["preprocess"]["quality"]
    rules = [
        # table shape
        gxe.ExpectTableColumnsToMatchSet(column_set=required_columns(), exact_match=False),
        gxe.ExpectTableRowCountToBeBetween(min_value=cfg["min_rows"], max_value=cfg["max_rows"]),
        # identifiers
        gxe.ExpectColumnValuesToNotBeNull(column="image_id"),
        gxe.ExpectColumnValuesToBeUnique(column="image_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="group_id"),
        # labels and categories (missing values are checked separately)
        gxe.ExpectColumnValuesToBeInSet(column="source", value_set=SOURCES),
        gxe.ExpectColumnValuesToBeInSet(column="split", value_set=SPLITS),
        gxe.ExpectColumnValuesToNotBeNull(column="urgency"),
        gxe.ExpectColumnValuesToBeInSet(column="urgency", value_set=URGENCY_TIERS),
        gxe.ExpectColumnValuesToNotBeNull(column="diagnosis"),
        gxe.ExpectColumnValuesToBeInSet(column="diagnosis",
                                        value_set=sorted(urgency_cfg["mapping"])),
        gxe.ExpectColumnValuesToBeInSet(column="sex", value_set=SEXES),
        gxe.ExpectColumnValuesToBeInSet(column="body_site", value_set=BODY_SITES),
        # demographics: valid when present, limited share missing
        gxe.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=120),
        gxe.ExpectColumnValuesToBeBetween(column="fitzpatrick", min_value=1, max_value=6),
        # image quality: same limits the quality filter applied
        gxe.ExpectColumnValuesToBeInSet(column="readable", value_set=[True]),
        gxe.ExpectColumnValuesToBeNull(column="error"),
        gxe.ExpectColumnValuesToBeBetween(column="width", min_value=quality["min_side_px"]),
        gxe.ExpectColumnValuesToBeBetween(column="height", min_value=quality["min_side_px"]),
        gxe.ExpectColumnValuesToBeBetween(column="brightness",
                                          min_value=quality["min_brightness"],
                                          max_value=quality["max_brightness"]),
        gxe.ExpectColumnValuesToBeBetween(column="contrast", min_value=quality["min_contrast"]),
        gxe.ExpectColumnValuesToBeBetween(column="sharpness", min_value=quality["min_sharpness"]),
        gxe.ExpectColumnValuesToMatchRegex(column="phash", regex=r"^[0-9a-f]{16}$"),
    ]
    # patient answers are 1/0 when known (missing = not asked)
    rules += [gxe.ExpectColumnValuesToBeInSet(column=c, value_set=[0, 1]) for c in ANSWER_COLUMNS]
    # "mostly" = share of rows that must be present, e.g. age at most 25% missing
    for column, max_missing in cfg.get("max_missing", {}).items():
        rules.append(gxe.ExpectColumnValuesToNotBeNull(column=column, mostly=1 - max_missing))
    return rules


def validate_dataframe(df: pd.DataFrame, expectations: list) -> Dict[str, Any]:
    """Run the expectations on df and return a JSON-friendly report."""
    context = gx.get_context(mode="ephemeral")
    batch = (context.data_sources.add_pandas("skinsort")
             .add_dataframe_asset("dataset")
             .add_batch_definition_whole_dataframe("full")
             .get_batch(batch_parameters={"dataframe": df}))
    suite = context.suites.add(gx.ExpectationSuite(name="skinsort_dataset",
                                                   expectations=expectations))
    result = batch.validate(suite)

    checks = []
    for r in result.results:
        cfg = r.expectation_config
        details = r.result or {}
        checks.append({
            "rule": cfg.type,
            "column": cfg.kwargs.get("column"),
            "success": bool(r.success),
            "unexpected_count": details.get("unexpected_count"),
            "unexpected_examples": details.get("partial_unexpected_list", [])[:5],
            "observed_value": details.get("observed_value"),
        })
    failed = [c for c in checks if not c["success"]]
    return {"success": not failed, "rules_checked": len(checks),
            "rules_failed": len(failed), "failed": failed, "checks": checks}


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Summary statistics used as the baseline for later anomaly and drift checks."""
    def counts(series: pd.Series) -> Dict[str, int]:
        return {str(k): int(v) for k, v in series.fillna("missing").value_counts().items()}

    numeric = {}
    for column in NUMERIC_STATS_COLUMNS:
        if column in df.columns:
            described = pd.to_numeric(df[column], errors="coerce").describe()
            numeric[column] = {k: (None if pd.isna(v) else round(float(v), 3))
                               for k, v in described.items()}
    urgency_by_split = {
        str(split): counts(group["urgency"]) for split, group in df.groupby("split")
    } if {"split", "urgency"} <= set(df.columns) else {}
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_rate": {c: round(float(v), 4) for c, v in df.isna().mean().items()},
        "numeric": numeric,
        "categorical": {c: counts(df[c]) for c in CATEGORICAL_STATS_COLUMNS if c in df.columns},
        "urgency_by_split": urgency_by_split,
    }


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    processed, logs = resolve_path(params, "processed"), resolve_path(params, "logs")
    df = pd.read_csv(processed / "dataset.csv", low_memory=False)
    urgency_cfg = load_yaml(resolve_path(params, "urgency_mapping"))

    report = validate_dataframe(df, build_expectations(params, urgency_cfg))
    statistics = compute_statistics(df)

    logs.mkdir(parents=True, exist_ok=True)
    (logs / "schema_report.json").write_text(json.dumps(report, indent=2, default=str))
    (logs / "data_statistics.json").write_text(json.dumps(statistics, indent=2, default=str))

    summary = {"rows": statistics["rows"], "rules_checked": report["rules_checked"],
               "rules_failed": report["rules_failed"]}
    if not report["success"]:
        names = [f"{c['rule']}({c['column']})" for c in report["failed"]]
        logger.error("Schema validation failed: %s", names)
        raise SchemaValidationError(
            f"{report['rules_failed']} schema rule(s) failed: {names}. "
            f"See {logs / 'schema_report.json'}"
        )
    logger.info("Schema validation passed: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset.csv and write statistics")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
