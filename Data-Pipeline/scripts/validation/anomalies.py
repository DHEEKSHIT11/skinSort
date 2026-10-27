"""Detect data anomalies by comparing this run's data with a reviewed baseline.

Input : data/processed/dataset.csv
        reference/baseline_profile.json   (a snapshot of a known-good run, kept in Git)
Output: logs/anomaly_report.json

Checks (all limits in params.yaml -> anomaly):
  row_count   - the dataset grew or shrank a lot
  missing     - a column's share of missing values went up
  outliers    - more values fall outside the baseline's IQR fences
  drift       - a column's mean moved, measured in baseline standard deviations
  label_mix   - the urgency mix of a split changed (total variation distance)

Each finding is "warn" or "critical". Any finding sends an alert (Slack/email via
scripts.common.alerts). Critical findings also stop the run (anomaly.stop_on), so
bad data is never versioned. Schema violations are caught earlier by schema.py.

Refresh the baseline only after a person has checked a run:
    python -m scripts.validation.anomalies --update-baseline
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.common.alerts import send_alert
from scripts.common.config import PIPELINE_ROOT, load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("validation.anomalies")

SEVERITY_ORDER = {"warn": 1, "critical": 2}


class AnomalyError(Exception):
    """Raised when anomalies at or above anomaly.stop_on severity are found."""


# --- profile: the numbers a run is compared on ------------------------------------

def build_profile(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise df: row count, missing rates, numeric spread and urgency mix per split."""
    k = cfg["outliers"]["iqr_multiplier"]
    numeric = {}
    for column in cfg["numeric_columns"]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        lower, upper = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        numeric[column] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "q1": q1, "q3": q3, "lower_fence": lower, "upper_fence": upper,
            "outlier_share": float(((values < lower) | (values > upper)).mean()),
        }
    label_mix = {
        str(split): {str(tier): float(share) for tier, share in
                     group["urgency"].value_counts(normalize=True).items()}
        for split, group in df.groupby("split")
    }
    return {
        "rows": int(len(df)),
        "missing_rate": {c: float(v) for c, v in df.isna().mean().items()},
        "numeric": numeric,
        "label_mix": label_mix,
    }


# --- checks: each returns a list of findings ----------------------------------------

def _severity(change: float, limits: Dict[str, float]) -> Optional[str]:
    if change >= limits["critical"]:
        return "critical"
    if change >= limits["warn"]:
        return "warn"
    return None


def _finding(check: str, target: str, severity: str, current: float, baseline: float,
             change: float, message: str) -> Dict[str, Any]:
    return {"check": check, "target": target, "severity": severity,
            "current": round(current, 4), "baseline": round(baseline, 4),
            "change": round(change, 4), "message": message}


def check_row_count(profile, baseline, cfg) -> List[Dict[str, Any]]:
    change = abs(profile["rows"] - baseline["rows"]) / max(baseline["rows"], 1)
    severity = _severity(change, cfg["row_count"])
    if not severity:
        return []
    return [_finding("row_count", "dataset", severity, profile["rows"], baseline["rows"],
                     change, f"Row count changed by {change:.1%} "
                             f"({baseline['rows']} -> {profile['rows']})")]


def check_missing(profile, baseline, cfg) -> List[Dict[str, Any]]:
    findings = []
    for column, rate in profile["missing_rate"].items():
        if column not in baseline["missing_rate"]:
            continue
        base = baseline["missing_rate"][column]
        severity = _severity(rate - base, cfg["missing"])
        if severity:
            findings.append(_finding(
                "missing", column, severity, rate, base, rate - base,
                f"{column}: missing values rose from {base:.1%} to {rate:.1%}"))
    return findings


def check_outliers(df, baseline, cfg) -> List[Dict[str, Any]]:
    """Share of values outside the *baseline's* fences, compared with the baseline share."""
    findings = []
    for column, base in baseline["numeric"].items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        share = float(((values < base["lower_fence"]) | (values > base["upper_fence"])).mean())
        severity = _severity(share - base["outlier_share"], cfg["outliers"])
        if severity:
            findings.append(_finding(
                "outliers", column, severity, share, base["outlier_share"],
                share - base["outlier_share"],
                f"{column}: {share:.1%} of values are outliers "
                f"(baseline {base['outlier_share']:.1%})"))
    return findings


def check_drift(profile, baseline, cfg) -> List[Dict[str, Any]]:
    """Mean shift in units of the baseline standard deviation (standardised difference)."""
    findings = []
    for column, base in baseline["numeric"].items():
        current = profile["numeric"].get(column)
        if current is None or base["std"] == 0:
            continue
        shift = abs(current["mean"] - base["mean"]) / base["std"]
        severity = _severity(shift, cfg["drift"])
        if severity:
            findings.append(_finding(
                "drift", column, severity, current["mean"], base["mean"], shift,
                f"{column}: mean moved {shift:.2f} standard deviations "
                f"({base['mean']:.2f} -> {current['mean']:.2f})"))
    return findings


def check_label_mix(profile, baseline, cfg) -> List[Dict[str, Any]]:
    """Total variation distance between the urgency mixes: 0 = identical, 1 = no overlap."""
    findings = []
    for split, mix in profile["label_mix"].items():
        base = baseline["label_mix"].get(split)
        if base is None:
            findings.append(_finding("label_mix", split, "warn", 1.0, 0.0, 1.0,
                                     f"Split '{split}' is new; it is not in the baseline"))
            continue
        tiers = set(mix) | set(base)
        distance = 0.5 * sum(abs(mix.get(t, 0.0) - base.get(t, 0.0)) for t in tiers)
        severity = _severity(distance, cfg["label_mix"])
        if severity:
            findings.append(_finding(
                "label_mix", split, severity, distance, 0.0, distance,
                f"{split}: urgency mix shifted by {distance:.1%} "
                f"(now {', '.join(f'{t} {mix.get(t, 0):.0%}' for t in sorted(tiers))})"))
    return findings


def detect(df: pd.DataFrame, baseline: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict]:
    profile = build_profile(df, cfg)
    findings = (check_row_count(profile, baseline, cfg) + check_missing(profile, baseline, cfg)
                + check_outliers(df, baseline, cfg) + check_drift(profile, baseline, cfg)
                + check_label_mix(profile, baseline, cfg))
    return sorted(findings, key=lambda f: -SEVERITY_ORDER[f["severity"]])


# --- run ----------------------------------------------------------------------------

def baseline_path(cfg: Dict[str, Any]) -> Path:
    path = Path(cfg["baseline"])
    return path if path.is_absolute() else PIPELINE_ROOT / path


def update_baseline(params: Dict[str, Any]) -> Path:
    """Save this run's profile as the new baseline (do this only after reviewing the run)."""
    cfg = params["anomaly"]
    df = pd.read_csv(resolve_path(params, "processed") / "dataset.csv", low_memory=False)
    path = baseline_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_profile(df, cfg), indent=2, sort_keys=True))
    logger.info("Baseline written to %s. Commit it to Git to make it official.", path)
    return path


def format_alert(findings: List[Dict[str, Any]], stopped: bool) -> str:
    lines = [f"[{f['severity'].upper()}] {f['message']}" for f in findings[:15]]
    if len(findings) > 15:
        lines.append(f"... and {len(findings) - 15} more (see logs/anomaly_report.json)")
    lines.append("Pipeline STOPPED before versioning." if stopped
                 else "Pipeline continued; please review.")
    return "\n".join(lines)


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = params["anomaly"]
    df = pd.read_csv(resolve_path(params, "processed") / "dataset.csv", low_memory=False)
    path = baseline_path(cfg)
    if not path.is_file():
        send_alert("SkinSort data: no anomaly baseline",
                   f"{path} is missing, so anomaly checks were skipped. Create it with "
                   "`python -m scripts.validation.anomalies --update-baseline`.")
        findings: List[Dict[str, Any]] = []
    else:
        findings = detect(df, json.loads(path.read_text()), cfg)

    stop_level = SEVERITY_ORDER.get(cfg["stop_on"])  # None for "never"
    stopped = bool(stop_level) and any(SEVERITY_ORDER[f["severity"]] >= stop_level
                                       for f in findings)
    counts = {s: sum(f["severity"] == s for f in findings) for s in SEVERITY_ORDER}
    report = {"baseline": str(path), "baseline_found": path.is_file(), "rows": int(len(df)),
              "counts": counts, "stopped": stopped, "findings": findings}
    logs = resolve_path(params, "logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "anomaly_report.json").write_text(json.dumps(report, indent=2))

    if findings:
        send_alert(f"SkinSort data anomalies: {counts['critical']} critical, "
                   f"{counts['warn']} warning", format_alert(findings, stopped))
    summary = {"findings": len(findings), **counts, "stopped": stopped}
    if stopped:
        raise AnomalyError(f"Critical data anomalies found: {summary}. "
                           f"See {logs / 'anomaly_report.json'}")
    logger.info("Anomaly check complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dataset.csv with the baseline")
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--update-baseline", action="store_true",
                        help="save the current data as the new baseline instead of checking")
    args = parser.parse_args()
    configure_console_logging()
    params = load_params(args.params)
    if args.update_baseline:
        update_baseline(params)
    else:
        run(params)


if __name__ == "__main__":
    main()
