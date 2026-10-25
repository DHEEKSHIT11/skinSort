"""Drop images that are unreadable, too small, too dark/bright, flat or blurry.

Input : harmonized.csv + image_stats.csv
Output: data/interim/quality_passed.csv, quality_dropped.csv (with reason),
        logs/quality_report.json (drop counts per dataset and reason)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from scripts.common.config import load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("preprocess.quality")

# Checked in this order; an image is labelled with the first reason it fails.
REASONS = ["unreadable", "too_small", "too_dark", "too_bright", "low_contrast", "blurry"]


def quality_reasons(stats: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Return the drop reason per image ('' = passed)."""
    readable = stats["readable"].astype(str).str.lower().eq("true")
    checks = {
        "unreadable": ~readable,
        "too_small": stats[["width", "height"]].min(axis=1) < cfg["min_side_px"],
        "too_dark": stats["brightness"] < cfg["min_brightness"],
        "too_bright": stats["brightness"] > cfg["max_brightness"],
        "low_contrast": stats["contrast"] < cfg["min_contrast"],
        "blurry": stats["sharpness"] < cfg["min_sharpness"],
    }
    reason = pd.Series("", index=stats.index)
    for name in reversed(REASONS):  # reversed so the first listed reason wins
        reason = reason.mask(checks[name].fillna(False), name)
    return reason


def filter_quality(df: pd.DataFrame, stats: pd.DataFrame, cfg: Dict[str, Any]):
    """Join stats onto df and split into (passed, dropped, report)."""
    merged = df.merge(stats, on="image_id", how="left", validate="one_to_one")
    merged["drop_reason"] = quality_reasons(merged, cfg)
    passed = merged[merged["drop_reason"] == ""].drop(columns="drop_reason")
    dropped = merged[merged["drop_reason"] != ""]
    report = {}
    for source, group in merged.groupby("source"):
        counts = group["drop_reason"].value_counts().to_dict()
        report[source] = {
            "total": int(len(group)),
            "passed": int(counts.pop("", 0)),
            "dropped": {k: int(v) for k, v in counts.items()},
        }
        share = 1 - report[source]["passed"] / max(len(group), 1)
        report[source]["dropped_share"] = round(share, 4)
    return passed, dropped, report


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    interim = resolve_path(params, "interim")
    interim.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(interim / "harmonized.csv", low_memory=False)
    stats = pd.read_csv(interim / "image_stats.csv", low_memory=False)
    passed, dropped, report = filter_quality(df, stats, params["preprocess"]["quality"])
    passed.to_csv(interim / "quality_passed.csv", index=False)
    dropped.to_csv(interim / "quality_dropped.csv", index=False)
    logs_dir = resolve_path(params, "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "quality_report.json").write_text(json.dumps(report, indent=2))
    for source, entry in report.items():
        level = logger.warning if entry["dropped_share"] > 0.2 else logger.info
        level("%s: kept %d / %d, dropped %s", source, entry["passed"], entry["total"],
              entry["dropped"])
    return {"passed": int(len(passed)), "dropped": int(len(dropped))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter out low-quality images")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
