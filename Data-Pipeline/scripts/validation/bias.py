"""Detect bias with data slicing and mitigate it (Fairlearn).

Input : data/processed/dataset.csv
Output: logs/bias_report.json and logs/bias_report.md   (findings and trade-offs)
        data/processed/sample_weights.csv              (reweighing weights for training)

1. Representation - how many images, and what share are urgent, in each slice
   (sex, age group, body site, skin tone, source) of the train and test sets.
2. Performance - a quick proxy model (logistic regression on metadata and image
   statistics) is trained on `train` and scored on `test` + `pad_test`. Fairlearn's
   MetricFrame reports urgent recall per slice. The real image model is sliced the
   same way later; the proxy shows where the data alone already creates gaps.
3. Mitigation - on the demographic slice with the largest recall gap:
   a. reweighing (Kamiran & Calders, 2012): weight = P(group) * P(label) / P(group, label)
   b. Fairlearn ThresholdOptimizer with true-positive-rate parity (equal urgent recall)
   The gap and overall recall / balanced accuracy before and after are reported.

The dataset itself is valid, so a large gap sends a warning alert but never stops
the pipeline: bias is acted on in training (weights, thresholds, more data).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.common.alerts import send_alert
from scripts.common.config import load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("validation.bias")

DEMOGRAPHIC_SLICES = ["sex", "age_group", "body_site", "skin_tone"]
SEXES = ["female", "male", "unknown"]
BODY_SITES = ["head_neck", "trunk", "upper_extremity", "lower_extremity", "acral",
              "genital", "unknown"]
IMAGE_FEATURES = ["brightness", "contrast", "sharpness", "width", "height"]


# --- slices -------------------------------------------------------------------------

def add_slices(df: pd.DataFrame, age_bins: List[float]) -> pd.DataFrame:
    """Add the slicing columns: age_group and skin_tone (sex, body_site, source exist)."""
    out = df.copy()
    age = pd.to_numeric(out["age"], errors="coerce")
    labels = [f"{int(lo)}-{int(hi) - 1}" for lo, hi in zip(age_bins[:-1], age_bins[1:])]
    # ages outside the bins (e.g. exactly 120) and missing ages both become "unknown"
    out["age_group"] = pd.cut(age, bins=age_bins, labels=labels, right=False) \
        .astype(object).fillna("unknown")
    fitz = pd.to_numeric(out["fitzpatrick"], errors="coerce")
    out["skin_tone"] = np.select([fitz.between(1, 3), fitz.between(4, 6)],
                                 ["light (I-III)", "dark (IV-VI)"], default="unknown")
    out["is_urgent"] = (out["urgency"] == "urgent").astype(int)
    return out


def representation(df: pd.DataFrame, slices: List[str], splits: List[str]) -> Dict:
    """Images per slice group in each split, plus the urgent rate in train."""
    result = {}
    for column in slices:
        table = {}
        for split in splits:
            part = df[df["split"] == split]
            if part.empty:
                continue
            counts = part[column].value_counts()
            table[split] = {str(g): {"images": int(n), "share": round(n / len(part), 4)}
                            for g, n in counts.items()}
        train = df[df["split"] == "train"]
        if not train.empty:
            table["urgent_rate_in_train"] = {
                str(g): round(float(v), 4)
                for g, v in train.groupby(column)["is_urgent"].mean().items()}
        result[column] = table
    return result


# --- proxy model --------------------------------------------------------------------

def feature_matrix(df: pd.DataFrame, age_fill: float) -> pd.DataFrame:
    """Numeric features available for every image: metadata and image statistics."""
    age = pd.to_numeric(df["age"], errors="coerce")
    X = pd.DataFrame({"age": age.fillna(age_fill), "age_missing": age.isna().astype(int)},
                     index=df.index)
    for column in IMAGE_FEATURES:
        X[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    for sex in SEXES:
        X[f"sex_{sex}"] = (df["sex"] == sex).astype(int)
    for site in BODY_SITES:
        X[f"site_{site}"] = (df["body_site"] == site).astype(int)
    return X


def train_proxy(X: pd.DataFrame, y: pd.Series, seed: int,
                sample_weight: Optional[np.ndarray] = None):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000,
                                             random_state=seed))
    model.fit(X, y, logisticregression__sample_weight=sample_weight)
    return model


def recall_urgent(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def slice_metrics(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series,
                  min_positives: int) -> Dict[str, Any]:
    """Fairlearn MetricFrame per group; the gap only counts groups with enough urgent cases."""
    frame = MetricFrame(metrics={"recall": recall_urgent, "selection_rate": selection_rate,
                                 "count": count},
                        y_true=y_true, y_pred=y_pred, sensitive_features=groups)
    positives = y_true.groupby(groups).sum()
    by_group = {}
    for group, row in frame.by_group.iterrows():
        n_pos = int(positives.get(group, 0))
        by_group[str(group)] = {"images": int(row["count"]), "urgent_cases": n_pos,
                                "urgent_recall": round(float(row["recall"]), 4),
                                "flagged_urgent_rate": round(float(row["selection_rate"]), 4),
                                "judged": n_pos >= min_positives}
    judged = {g: m["urgent_recall"] for g, m in by_group.items() if m["judged"]}
    gap = round(max(judged.values()) - min(judged.values()), 4) if len(judged) > 1 else None
    return {"overall_urgent_recall": round(float(frame.overall["recall"]), 4),
            "by_group": by_group, "recall_gap": gap,
            "lowest_group": min(judged, key=judged.get) if gap is not None else None,
            "highest_group": max(judged, key=judged.get) if gap is not None else None}


# --- mitigation ---------------------------------------------------------------------

def reweighing_weights(groups: pd.Series, labels: pd.Series) -> pd.Series:
    """Kamiran & Calders reweighing: makes group and label independent in the weighted data."""
    p_group = groups.value_counts(normalize=True)
    p_label = labels.value_counts(normalize=True)
    p_joint = pd.crosstab(groups, labels, normalize=True)
    weights = [p_group[g] * p_label[y] / p_joint.loc[g, y] for g, y in zip(groups, labels)]
    return pd.Series(weights, index=groups.index, name="weight")


def coarsen(groups: pd.Series, keep: List[str]) -> pd.Series:
    """Merge groups too small to calibrate into 'other' so thresholds can be fitted."""
    return groups.where(groups.isin(keep), "other")


def summarise(y_true, y_pred, groups, min_positives) -> Dict[str, Any]:
    metrics = slice_metrics(y_true, y_pred, groups, min_positives)
    return {"overall_urgent_recall": metrics["overall_urgent_recall"],
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "recall_gap": metrics["recall_gap"], "by_group": metrics["by_group"]}


def mitigate(data: Dict[str, pd.DataFrame], feature: str, cfg: Dict[str, Any],
             seed: int, age_fill: float) -> Dict[str, Any]:
    """Compare baseline, reweighing and equal-recall thresholds on one slice."""
    train, val, test = data["train"], data["val"], data["eval"]
    X_train, X_val, X_test = (feature_matrix(d, age_fill) for d in (train, val, test))
    y_train, y_val, y_test = train["is_urgent"], val["is_urgent"], test["is_urgent"]
    min_pos = cfg["min_urgent_cases"]

    baseline = train_proxy(X_train, y_train, seed)
    results = {"feature": feature,
               "baseline": summarise(y_test, baseline.predict(X_test), test[feature], min_pos)}

    weights = reweighing_weights(train[feature], y_train)
    reweighed = train_proxy(X_train, y_train, seed, sample_weight=weights.to_numpy())
    results["reweighing"] = summarise(y_test, reweighed.predict(X_test), test[feature], min_pos)

    # Thresholds are fitted on the validation split, then applied to the test split.
    val_pos = val.groupby(feature)["is_urgent"].agg(["sum", "count"])
    keep = [g for g, row in val_pos.iterrows()
            if row["sum"] >= min_pos and row["count"] - row["sum"] >= min_pos]
    try:
        optimizer = ThresholdOptimizer(estimator=baseline,
                                       constraints="true_positive_rate_parity",
                                       objective="balanced_accuracy_score",
                                       predict_method="predict_proba", prefit=True)
        optimizer.fit(X_val, y_val, sensitive_features=coarsen(val[feature], keep))
        predicted = optimizer.predict(X_test, sensitive_features=coarsen(test[feature], keep),
                                      random_state=seed)
        results["threshold_optimizer"] = summarise(y_test, np.asarray(predicted),
                                                   test[feature], min_pos)
    except Exception as exc:  # noqa: BLE001 - e.g. a group with one class; reported, not fatal
        results["threshold_optimizer"] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


# --- report -------------------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = ["# SkinSort bias report (generated)", "",
             "Proxy model: logistic regression on metadata and image statistics, trained on",
             "`train`, scored on `test` + `pad_test`. Gaps only count groups with at least",
             f"{report['settings']['min_urgent_cases']} urgent cases.", "",
             "## Urgent-recall gap per slice", "",
             "| Slice | Gap | Lowest group | Highest group | Overall recall |",
             "| --- | --- | --- | --- | --- |"]
    for column, m in report["performance"].items():
        lines.append(f"| {column} | {_pct(m['recall_gap'])} | {m['lowest_group'] or '-'} | "
                     f"{m['highest_group'] or '-'} | {_pct(m['overall_urgent_recall'])} |")
    skin = report["representation"].get("skin_tone", {})
    lines += ["", "## Skin tone representation (images)", "",
              "| Split | " + " | ".join(["light (I-III)", "dark (IV-VI)", "unknown"]) + " |",
              "| --- | --- | --- | --- |"]
    for split, groups in skin.items():
        if split == "urgent_rate_in_train":
            continue
        cells = [str(groups.get(g, {}).get("images", 0))
                 for g in ["light (I-III)", "dark (IV-VI)", "unknown"]]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    mit = report.get("mitigation")
    if mit:
        lines += ["", f"## Mitigation on `{mit['feature']}`", "",
                  "| Method | Recall gap | Overall urgent recall | Balanced accuracy |",
                  "| --- | --- | --- | --- |"]
        for method in ["baseline", "reweighing", "threshold_optimizer"]:
            m = mit.get(method, {})
            if "error" in m:
                lines.append(f"| {method} | not possible: {m['error']} | | |")
            else:
                lines.append(f"| {method} | {_pct(m.get('recall_gap'))} | "
                             f"{_pct(m.get('overall_urgent_recall'))} | "
                             f"{_pct(m.get('balanced_accuracy'))} |")
    path.write_text("\n".join(lines) + "\n")


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg, seed = params["bias"], params["seed"]
    processed, logs = resolve_path(params, "processed"), resolve_path(params, "logs")
    df = add_slices(pd.read_csv(processed / "dataset.csv", low_memory=False),
                    params["preprocess"]["transform"]["age_bins"])
    slices, min_pos = cfg["slices"], cfg["min_urgent_cases"]

    report: Dict[str, Any] = {
        "settings": {"min_urgent_cases": min_pos, "max_recall_gap": cfg["max_recall_gap"]},
        "representation": representation(df, slices, ["train", "test", "pad_train",
                                                      "pad_test", "fairness_test"]),
    }

    train = df[df["split"] == "train"]
    val = df[df["split"].isin(["val", "pad_val"])]
    test = df[df["split"].isin(["test", "pad_test"])]
    age_fill = float(pd.to_numeric(train["age"], errors="coerce").median())
    model = train_proxy(feature_matrix(train, age_fill), train["is_urgent"], seed)
    predictions = model.predict(feature_matrix(test, age_fill))
    report["performance"] = {c: slice_metrics(test["is_urgent"], predictions, test[c], min_pos)
                             for c in slices}

    gaps = {c: report["performance"][c]["recall_gap"] for c in DEMOGRAPHIC_SLICES
            if c in slices and report["performance"][c]["recall_gap"] is not None}
    worst = max(gaps, key=gaps.get) if gaps else None
    weights_feature = worst or "sex"
    if worst:
        report["mitigation"] = mitigate({"train": train, "val": val, "eval": test},
                                        worst, cfg, seed, age_fill)

    # Reweighing weights for the training pools, for the model team to use.
    pools = df[df["split"].isin(["train", "pad_train"])]
    weights = pd.concat([reweighing_weights(p[weights_feature], p["is_urgent"])
                         for _, p in pools.groupby("split")])
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": pools["image_id"], "split": pools["split"],
                  "group": pools[weights_feature], "weight": weights.round(6)}) \
        .to_csv(processed / "sample_weights.csv", index=False)
    report["sample_weights"] = {"feature": weights_feature, "rows": int(len(pools))}

    logs.mkdir(parents=True, exist_ok=True)
    (logs / "bias_report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, logs / "bias_report.md")

    flagged = {c: g for c, g in gaps.items() if g > cfg["max_recall_gap"]}
    if flagged and cfg.get("alert_on_gap", True):
        send_alert("SkinSort data bias: urgent-recall gaps above target",
                   "\n".join(f"{c}: {g:.1%} gap (target <= {cfg['max_recall_gap']:.0%})"
                             for c, g in flagged.items())
                   + "\nPipeline continued; see logs/bias_report.md.")
    summary = {"worst_slice": worst, "worst_gap": gaps.get(worst) if worst else None,
               "slices_over_target": sorted(flagged)}
    logger.info("Bias check complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias slicing and mitigation report")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
