"""Bias check: does the model miss urgent cases more often for some patient groups?

Reads the per-image predictions that evaluate.py saved (outputs/evaluation/<name>/
predictions.csv, already using the tau picked on validation) and slices them with
Fairlearn's MetricFrame over the groups in params.yaml (bias section):

    sex, age_group, body_site, skin_tone, source

For every slice it reports urgent recall, precision and flagged share per group.
Groups with fewer than `min_urgent_cases` urgent images are shown but not used for the
gap, because a recall from a handful of cases is mostly noise. A slice fails when the
urgent recall gap (best group minus worst group) is above `max_recall_gap`.

Mitigation (--mitigate): per-group thresholds. For each group of a slice, tau is
re-picked so that group alone reaches the recall target, using validation rows only
(groups with too few urgent validation cases keep the global tau). The new thresholds
are then applied to the test rows and the gap / flagged share are compared. For the
fairness split (fitzpatrick17k) there is no validation data from that source, so its
thresholds are fitted on one half of the patients and reported on the other half.

    python -m src.training.bias --name image_main --mitigate age_group body_site source skin_tone
    python -m src.training.bias --name fusion_pad --fail-on-gap

Writes outputs/bias/<name>/: bias_report.json, bias_report.md, one bar chart per slice
and mitigation.json, and logs them to MLflow as run bias-<name>.
"""
import hashlib
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.training.dataset import ROOT, data_version, load_params
from src.training.evaluate import pick_tau
from src.training.train import setup_mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL_DIR = ROOT / "outputs" / "evaluation"
OUT_DIR = ROOT / "outputs" / "bias"
URGENT = 2   # label index of "urgent" (classes: routine, priority, urgent)

AGE_BINS = [0, 30, 45, 60, 75, 120]          # same bins as the age_bin_* features
AGE_LABELS = ["0-30", "30-45", "45-60", "60-75", "75+"]
SKIN_TONE = {1: "I-II (light)", 2: "I-II (light)", 3: "III-IV (medium)",
             4: "III-IV (medium)", 5: "V-VI (dark)", 6: "V-VI (dark)"}


def add_slice_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived slice columns named in params.yaml (age_group, skin_tone)."""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS,
                             include_lowest=True).astype(object).fillna("unknown")
    df["skin_tone"] = df["fitzpatrick"].map(SKIN_TONE).fillna("unknown")
    for col in ("sex", "body_site", "source"):
        df[col] = df[col].fillna("unknown")
    return df


def _recall(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, zero_division=np.nan)


def _precision(y_true, y_pred) -> float:
    return precision_score(y_true, y_pred, zero_division=np.nan)


def _urgent_cases(y_true, y_pred) -> int:
    return int(np.sum(y_true))


def wilson_interval(hits: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """95% interval for a recall measured on n urgent cases (shows how noisy small groups are)."""
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, zero_division=np.nan)


METRICS = {"images": count, "urgent_cases": _urgent_cases, "urgent_recall": _recall,
           "urgent_precision": _precision, "urgent_f1": _f1, "accuracy": accuracy_score,
           "flagged_share": selection_rate}


def slice_report(df: pd.DataFrame, slice_col: str, min_urgent: int,
                 max_gap: float) -> Optional[Dict[str, Any]]:
    """MetricFrame for one slice. None when the slice has only one known group here."""
    known = df[df[slice_col] != "unknown"]
    if known[slice_col].nunique() < 2:
        return None
    y_true = (df["label"] == URGENT).astype(int)
    y_pred = (df["pred"] == URGENT).astype(int)
    frame = MetricFrame(metrics=METRICS, y_true=y_true, y_pred=y_pred,
                        sensitive_features=df[slice_col])
    groups = frame.by_group.reset_index().rename(columns={slice_col: "group"})
    groups[["images", "urgent_cases"]] = groups[["images", "urgent_cases"]].astype(int)
    ci = [wilson_interval(round(r * n), n) if n else (float("nan"), float("nan"))
          for r, n in zip(groups["urgent_recall"].fillna(0), groups["urgent_cases"])]
    groups["recall_ci_low"], groups["recall_ci_high"] = zip(*ci)
    groups["used_for_gap"] = (groups["group"] != "unknown") & (groups["urgent_cases"] >= min_urgent)
    used = groups[groups["used_for_gap"]]
    if len(used) >= 2:
        best, worst = used.loc[used["urgent_recall"].idxmax()], used.loc[used["urgent_recall"].idxmin()]
        gap = float(best["urgent_recall"] - worst["urgent_recall"])
        status = "FAIL" if gap > max_gap else "PASS"
        worst_group, best_group = str(worst["group"]), str(best["group"])
    else:
        gap, status, worst_group, best_group = float("nan"), "TOO_FEW_CASES", None, None
    return {
        "slice": slice_col, "status": status, "recall_gap": gap,
        "worst_group": worst_group, "best_group": best_group,
        "overall_recall": float(frame.overall["urgent_recall"]),
        "groups": json.loads(groups.to_json(orient="records")),
    }


def save_slice_chart(report: Dict[str, Any], title: str, path: Path) -> None:
    groups = pd.DataFrame(report["groups"])
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(groups)), 3.6))
    colors = ["#3b6fb6" if used else "#b8c2cc" for used in groups["used_for_gap"]]
    bars = ax.bar(groups["group"].astype(str), groups["urgent_recall"].fillna(0), color=colors)
    for bar, n in zip(bars, groups["urgent_cases"]):
        ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(report["overall_recall"], color="#444", linestyle="--", linewidth=1,
               label=f"overall {report['overall_recall']:.2f}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("urgent recall")
    gap = report["recall_gap"]
    gap_text = "n/a" if math.isnan(gap) else f"{gap:.3f}"
    ax.set_title(f"{title} — {report['slice']}: gap {gap_text} ({report['status']})", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_bias_check(params: Dict[str, Any], predictions: pd.DataFrame, name: str,
                   out_dir: Optional[Path] = None, include_val: bool = False) -> Dict[str, Any]:
    cfg = params["bias"]
    out_dir = Path(out_dir or OUT_DIR / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = add_slice_columns(predictions)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for split, df in predictions.groupby("eval_split", sort=False):
        if split == "val" and not include_val:   # only used to fit mitigation thresholds
            continue
        results[split] = []
        for slice_col in cfg["slices"]:
            report = slice_report(df, slice_col, cfg["min_urgent_cases"], cfg["max_recall_gap"])
            if report is None:
                continue
            results[split].append(report)
            save_slice_chart(report, f"{name} / {split}", out_dir / f"{split}_{slice_col}.png")

    failed = [f"{split}/{r['slice']}" for split, reps in results.items() for r in reps
              if r["status"] == "FAIL"]
    checked = [r for reps in results.values() for r in reps if r["status"] in ("PASS", "FAIL")]
    # INCONCLUSIVE: no slice had enough urgent cases to measure a gap, so nothing was proven
    result = "FAIL" if failed else ("PASS" if checked else "INCONCLUSIVE")
    summary = {"name": name, "max_recall_gap": cfg["max_recall_gap"],
               "min_urgent_cases": cfg["min_urgent_cases"], "result": result,
               "failed_slices": failed, "passed": not failed, "splits": results}
    (out_dir / "bias_report.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "bias_report.md").write_text(to_markdown(summary))

    with mlflow.start_run(run_name=f"bias-{name}"):
        mlflow.set_tags(data_version(params))
        mlflow.log_params({"max_recall_gap": cfg["max_recall_gap"],
                           "min_urgent_cases": cfg["min_urgent_cases"],
                           "slices": ",".join(cfg["slices"])})
        for split, reps in results.items():
            for r in reps:
                if not math.isnan(r["recall_gap"]):
                    mlflow.log_metric(f"{split}_{r['slice']}_recall_gap", r["recall_gap"])
        mlflow.log_metric("bias_passed", float(summary["passed"]))
        mlflow.log_artifacts(str(out_dir))
    return summary


def to_markdown(summary: Dict[str, Any]) -> str:
    lines = [f"# Bias report: {summary['name']}", "",
             f"Urgent recall gap allowed: {summary['max_recall_gap']}. Groups with fewer than "
             f"{summary['min_urgent_cases']} urgent images are listed but not used for the gap.", "",
             f"**Result: {summary['result']}**"
             + (f" (failed: {', '.join(summary['failed_slices'])})" if summary["failed_slices"] else ""), ""]
    for split, reps in summary["splits"].items():
        lines += [f"## {split}", ""]
        for r in reps:
            gap = "n/a" if math.isnan(r["recall_gap"]) else f"{r['recall_gap']:.3f}"
            lines += [f"### {r['slice']}: gap {gap}, {r['status']}", "",
                      "| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 "
                      "| Accuracy (urgent vs not) | Flagged | Used for gap |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for g in r["groups"]:
                fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
                lines.append(f"| {g['group']} | {g['images']} | {g['urgent_cases']} | "
                             f"{fmt(g['urgent_recall'])} ({fmt(g['recall_ci_low'])}–{fmt(g['recall_ci_high'])}) | "
                             f"{fmt(g['urgent_precision'])} | {fmt(g.get('urgent_f1'))} | {fmt(g.get('accuracy'))} | "
                             f"{fmt(g['flagged_share'])} | {'yes' if g['used_for_gap'] else 'no'} |")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- mitigation

def apply_thresholds(df: pd.DataFrame, slice_col: str, taus: Dict[str, float],
                     default_tau: float) -> pd.DataFrame:
    """Re-predict with one urgent threshold per group (other classes: argmax as before)."""
    df = df.copy()
    tau = df[slice_col].map(taus).fillna(default_tau).to_numpy()
    others = np.where(df["p_priority"] > df["p_routine"], 1, 0)
    df["pred"] = np.where(df["p_urgent"] >= tau, URGENT, others)
    return df


def fit_group_thresholds(fit_df: pd.DataFrame, slice_col: str, target: float,
                         min_urgent: int, default_tau: float) -> Dict[str, float]:
    """Per-group tau reaching the recall target on fit_df. A group never gets a
    threshold above the global one, so mitigation only adds flags (never misses more)."""
    taus = {}
    for group, g in fit_df.groupby(slice_col):
        is_urgent = (g["label"] == URGENT).astype(int).to_numpy()
        if group == "unknown" or is_urgent.sum() < min_urgent:
            continue
        taus[str(group)] = min(default_tau, pick_tau(g["p_urgent"].to_numpy(), is_urgent, target))
    return taus


def patient_half(df: pd.DataFrame) -> pd.Series:
    """Stable 50/50 split by patient / lesion group, so one person never lands in both halves."""
    key = df["group_id"].fillna(df["image_id"]).astype(str)
    return key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 2)


def summarise(df: pd.DataFrame, slice_col: str, min_urgent: int, max_gap: float) -> Dict[str, Any]:
    is_urgent = df["label"] == URGENT
    flagged = df["pred"] == URGENT
    report = slice_report(df, slice_col, min_urgent, max_gap)
    return {"recall_gap": report["recall_gap"] if report else float("nan"),
            "status": report["status"] if report else "n/a",
            "worst_group": report["worst_group"] if report else None,
            "urgent_recall": float((flagged & is_urgent).sum() / max(is_urgent.sum(), 1)),
            "flagged_share": float(flagged.mean()),
            "groups": report["groups"] if report else []}


def cross_fit(df: pd.DataFrame, slice_col: str, target: float, global_tau: float,
              min_urgent: int) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    """Two-fold cross-fitting by patient: thresholds fitted on one half are applied to the
    other half and vice versa, so every row is re-predicted by thresholds that never saw it."""
    half = patient_half(df).to_numpy()
    parts, taus = [], []
    for h in (0, 1):
        t = fit_group_thresholds(df[half != h], slice_col, target, min_urgent, global_tau)
        parts.append(apply_thresholds(df[half == h], slice_col, t, global_tau))
        taus.append(t)
    return pd.concat(parts).loc[df.index], taus


def mitigate(predictions: pd.DataFrame, slice_col: str, target: float, global_tau: float,
             min_urgent: int, max_gap: float, include_val: bool = False) -> List[Dict[str, Any]]:
    """Per-group thresholds for one slice, for each evaluated split where it applies.

    test: thresholds fitted on the val rows, applied to test.
    val (only with include_val, i.e. out-of-fold predictions from fit_final.py) and any split
    without its own validation rows (fairness): two-fold cross-fitting by patient.
    """
    out = []
    for split, df in predictions.groupby("eval_split", sort=False):
        if (split == "val" and not include_val) or df[df[slice_col] != "unknown"][slice_col].nunique() < 2:
            continue
        if split == "test":
            fit_df = predictions[predictions["eval_split"] == "val"]
            if fit_df[fit_df[slice_col] != "unknown"][slice_col].nunique() < 2:
                continue
            taus = fit_group_thresholds(fit_df, slice_col, target, min_urgent, global_tau)
            after_df, fitted_on, tau_sets = apply_thresholds(df, slice_col, taus, global_tau), "val", [taus]
        else:
            after_df, tau_sets = cross_fit(df, slice_col, target, global_tau, min_urgent)
            fitted_on = f"{split}, cross-fitted by patient halves"
        groups = sorted({g for t in tau_sets for g in t})
        group_taus = {g: float(np.mean([t.get(g, global_tau) for t in tau_sets])) for g in groups}
        before = summarise(df, slice_col, min_urgent, max_gap)
        after = summarise(after_df, slice_col, min_urgent, max_gap)
        out.append({"split": split, "slice": slice_col, "fitted_on": fitted_on,
                    "global_tau": global_tau, "group_taus": group_taus,
                    "group_taus_per_fold": tau_sets,
                    "images": len(df), "before": before, "after": after})
    return out


def mitigation_markdown(results: List[Dict[str, Any]]) -> str:
    fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    lines = ["# Mitigation: per-group thresholds", "",
             "Each group gets its own urgent threshold (never higher than the global one), fitted "
             "so the group reaches the recall target on the fitting rows.", "",
             "| Split | Slice | Fitted on | Gap before → after | Worst group before → after | "
             "Urgent recall before → after | Flagged before → after | Status after |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        b, a = r["before"], r["after"]
        lines.append(f"| {r['split']} | {r['slice']} | {r['fitted_on']} | {fmt(b['recall_gap'])} → "
                     f"{fmt(a['recall_gap'])} | {b['worst_group']} → {a['worst_group']} | "
                     f"{fmt(b['urgent_recall'])} → {fmt(a['urgent_recall'])} | "
                     f"{b['flagged_share']:.1%} → {a['flagged_share']:.1%} | {a['status']} |")
    lines += ["", "Group thresholds:", ""]
    for r in results:
        taus = ", ".join(f"{g} {t:.3f}" for g, t in r["group_taus"].items()) or "none (too few cases)"
        lines.append(f"- {r['split']} / {r['slice']} (global {r['global_tau']:.3f}): {taus}")
    lines += ["", "## Per group, before → after", ""]
    for r in results:
        lines += [f"### {r['split']} / {r['slice']}", "",
                  "| Group | Urgent | Recall before → after | Flagged before → after |", "|---|---|---|---|"]
        after = {g["group"]: g for g in r["after"]["groups"]}
        for g in r["before"]["groups"]:
            a = after.get(g["group"], g)
            lines.append(f"| {g['group']} | {g['urgent_cases']} | {fmt(g['urgent_recall'])} → "
                         f"{fmt(a['urgent_recall'])} | {g['flagged_share']:.1%} → {a['flagged_share']:.1%} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice-based bias check on saved predictions")
    parser.add_argument("--name", required=True, help="folder under outputs/evaluation")
    parser.add_argument("--fail-on-gap", action="store_true", help="exit 1 if any slice fails")
    parser.add_argument("--include-val", action="store_true",
                        help="also slice the val rows (use for out-of-fold predictions from fit_final.py)")
    parser.add_argument("--mitigate", nargs="*", default=[],
                        help="slices to try per-group thresholds on (needs val rows in predictions.csv)")
    args = parser.parse_args()

    params = load_params()
    setup_mlflow(params)
    predictions = pd.read_csv(EVAL_DIR / args.name / "predictions.csv")
    summary = run_bias_check(params, predictions, args.name, include_val=args.include_val)
    for split, reps in summary["splits"].items():
        for r in reps:
            gap = "n/a  " if math.isnan(r["recall_gap"]) else f"{r['recall_gap']:.3f}"
            worst = f"worst {r['worst_group']}" if r["worst_group"] else ""
            print(f"{split:9s} {r['slice']:10s} gap {gap} {r['status']:13s} {worst}")
    print("bias check", summary["result"], summary["failed_slices"] or "")

    if args.mitigate:
        cfg = params["bias"]
        metrics = json.loads((EVAL_DIR / args.name / "metrics.json").read_text())
        prepared = add_slice_columns(predictions)
        results = [r for col in args.mitigate
                   for r in mitigate(prepared, col, metrics["recall_target"], metrics["tau"],
                                     cfg["min_urgent_cases"], cfg["max_recall_gap"],
                                     include_val=args.include_val)]
        out_dir = OUT_DIR / args.name
        (out_dir / "mitigation.json").write_text(json.dumps(results, indent=2))
        (out_dir / "mitigation.md").write_text(mitigation_markdown(results))
        with mlflow.start_run(run_name=f"bias-mitigation-{args.name}"):
            mlflow.set_tags(data_version(params))
            mlflow.log_param("mitigated_slices", ",".join(args.mitigate))
            for r in results:
                key = f"{r['split']}_{r['slice']}"
                for when in ("before", "after"):
                    if not math.isnan(r[when]["recall_gap"]):
                        mlflow.log_metric(f"{key}_recall_gap_{when}", r[when]["recall_gap"])
                    mlflow.log_metric(f"{key}_flagged_{when}", r[when]["flagged_share"])
            mlflow.log_artifact(str(out_dir / "mitigation.json"))
            mlflow.log_artifact(str(out_dir / "mitigation.md"))
        print("\nmitigation (per-group thresholds):")
        for r in results:
            b, a = r["before"], r["after"]
            print(f"{r['split']:9s} {r['slice']:10s} gap {b['recall_gap']:.3f} -> {a['recall_gap']:.3f}  "
                  f"flagged {b['flagged_share']:.1%} -> {a['flagged_share']:.1%}  ({a['status']})")
    if args.fail_on_gap and not summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
