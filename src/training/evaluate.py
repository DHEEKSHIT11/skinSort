"""Validate a trained model on a hold-out test set.

Steps, all on data the model never trained on:
  1. Score the validation split and fit a calibration (temperature + per-class
     offset) so P(urgent) matches the real urgent rate.
  2. Pick the urgent threshold tau on validation: the highest tau that reaches
     the urgent recall target in params.yaml (evaluate section).
  3. Score the test split once with that temperature and tau, and check the targets.

    # image model on the main (dermoscopy) test set, also scoring fairness_test for bias.py
    python -m src.training.evaluate --checkpoint outputs/models/efficientnet_b3/best.pt \\
        --val-split val --test-split test --target urgent_recall_target \\
        --extra-splits fairness --name image_main

    # fusion model on the phone-photo test set
    python -m src.training.evaluate --checkpoint outputs/models/fusion/best.pt \\
        --val-split pad_val --test-split pad_test --target pad_urgent_recall_target --name fusion_pad

Writes outputs/evaluation/<name>/: metrics.json, predictions.csv (one row per image,
used by bias.py), confusion_matrix.png and calibration.png. Everything is also
logged to MLflow. With --fail-on-target the exit code is 1 when a target is missed,
so CI can block a model.
"""
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
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.models.fusion_model import fusion_head_from_checkpoint
from src.training.dataset import ROOT, load_params, load_table, data_version
from src.training.train import pick_device, set_seed, setup_mlflow
from src.training.train_fusion import extract_features, load_image_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "outputs" / "evaluation"


class Scorer:
    """Turns a checkpoint (image model or fusion head) into logits for a split."""

    def __init__(self, params: Dict[str, Any], checkpoint: Path, device: torch.device):
        self.params, self.device = params, device
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.kind = "fusion" if "image_checkpoint" in ckpt else "image"
        self.classes = ckpt["classes"]
        if self.kind == "fusion":
            self.image_model = load_image_model(params, Path(ckpt["image_checkpoint"]), device)
            self.head = fusion_head_from_checkpoint(params, ckpt).to(device).eval()
        else:
            self.image_model = load_image_model(params, checkpoint, device)
        self.backbone = ckpt["backbone"]

    def logits(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feats, answers, labels = extract_features(self.image_model, df, self.params, self.device)
        with torch.no_grad():
            feats = feats.to(self.device)
            if self.kind == "fusion":
                out = self.head(feats, answers.to(self.device))
            else:
                out = self.image_model.head(feats)
        return out.cpu().numpy(), labels.numpy()


Calibration = Tuple[float, np.ndarray]   # (temperature, per-class offset)


def softmax(logits: np.ndarray, calibration: Optional[Calibration] = None) -> np.ndarray:
    temperature, offset = calibration or (1.0, np.zeros(logits.shape[1]))
    z = logits / temperature + offset
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_calibration(logits: np.ndarray, labels: np.ndarray) -> Calibration:
    """Temperature T plus one offset per class, fitted to minimise validation log loss.
    T fixes over/under-confidence; the offsets undo the shift that class weights in
    training put on the probabilities (temperature alone cannot, and runs off to
    infinity when the shift is large)."""
    # Search T on a grid (a few extreme logits make a joint gradient fit diverge);
    # for each T the offsets are a convex problem, solved exactly with LBFGS.
    x, y = torch.tensor(logits, dtype=torch.float64), torch.tensor(labels)
    best: Tuple[float, float, np.ndarray] = (math.inf, 1.0, np.zeros(logits.shape[1]))
    for temperature in np.geomspace(0.05, 100, 81):
        offset = torch.zeros(logits.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([offset], lr=1.0, max_iter=100,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(x / temperature + offset, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            loss = float(torch.nn.functional.cross_entropy(x / temperature + offset, y))
        if loss < best[0]:
            best = (loss, float(temperature), offset.detach().numpy())
    _, temperature, b = best
    return temperature, b - b.mean()


def calibration_error(p_urgent: np.ndarray, is_urgent: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of P(urgent): average gap between predicted and
    observed urgent rate across equal-width probability bins, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_urgent, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p_urgent[mask].mean() - is_urgent[mask].mean())
    return float(ece)


def pick_tau(p_urgent: np.ndarray, is_urgent: np.ndarray, target: float) -> float:
    """Highest threshold whose urgent recall reaches the target (flags the fewest cases)."""
    if is_urgent.sum() == 0:
        raise ValueError("No urgent cases in the validation split; cannot pick tau")
    scores = np.sort(p_urgent[is_urgent == 1])[::-1]
    needed = math.ceil(target * len(scores))
    return float(scores[needed - 1])


def predict_classes(probs: np.ndarray, tau: float, urgent_idx: int) -> np.ndarray:
    """Urgent when P(urgent) >= tau; otherwise the more likely of the other classes."""
    others = probs.copy()
    others[:, urgent_idx] = -1
    preds = others.argmax(axis=1)
    preds[probs[:, urgent_idx] >= tau] = urgent_idx
    return preds


def split_metrics(probs: np.ndarray, labels: np.ndarray, tau: float,
                  urgent_idx: int) -> Dict[str, float]:
    preds = predict_classes(probs, tau, urgent_idx)
    is_urgent = (labels == urgent_idx).astype(int)
    flagged = (preds == urgent_idx).astype(int)
    metrics = {
        "urgent_recall": recall_score(is_urgent, flagged, zero_division=0),
        "urgent_precision": precision_score(is_urgent, flagged, zero_division=0),
        "flagged_share": flagged.mean(),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "calibration_error": calibration_error(probs[:, urgent_idx], is_urgent),
        "urgent_auc": float("nan"),
        "images": len(labels),
        "urgent_cases": int(is_urgent.sum()),
    }
    if 0 < is_urgent.sum() < len(is_urgent):
        metrics["urgent_auc"] = roc_auc_score(is_urgent, probs[:, urgent_idx])
    return {k: float(v) for k, v in metrics.items()}


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, classes: List[str],
                          title: str, path: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_calibration_plot(p_raw: np.ndarray, p_cal: np.ndarray, is_urgent: np.ndarray,
                          title: str, path: Path, bins: int = 10) -> None:
    edges = np.linspace(0, 1, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    for p, name in [(p_raw, "uncalibrated"), (p_cal, "calibrated")]:
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        observed = [is_urgent[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)]
        ax.plot(centers, observed, marker="o", label=name)
    ax.set_xlabel("predicted P(urgent)")
    ax.set_ylabel("observed urgent rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(params: Dict[str, Any], checkpoint: Path, val_df: pd.DataFrame,
             test_df: pd.DataFrame, target_key: str, name: str, device: torch.device,
             extra: Optional[Dict[str, pd.DataFrame]] = None,
             out_dir: Optional[Path] = None) -> Dict[str, Any]:
    cfg = params["evaluate"]
    target = cfg[target_key]
    out_dir = Path(out_dir or OUT_DIR / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer = Scorer(params, checkpoint, device)
    urgent_idx = scorer.classes.index("urgent")

    val_logits, val_labels = scorer.logits(val_df)
    calibration = fit_calibration(val_logits, val_labels)
    temperature = calibration[0]
    val_probs = softmax(val_logits, calibration)
    tau = pick_tau(val_probs[:, urgent_idx], (val_labels == urgent_idx).astype(int), target)

    test_logits, test_labels = scorer.logits(test_df)
    test_raw = softmax(test_logits)
    test_probs = softmax(test_logits, calibration)
    val_m = split_metrics(val_probs, val_labels, tau, urgent_idx)
    test_m = split_metrics(test_probs, test_labels, tau, urgent_idx)
    test_m["calibration_error_uncalibrated"] = calibration_error(
        test_raw[:, urgent_idx], (test_labels == urgent_idx).astype(int))

    checks = {
        "urgent_recall": {"value": test_m["urgent_recall"], "target": target,
                          "passed": test_m["urgent_recall"] >= target},
        "calibration_error": {"value": test_m["calibration_error"],
                              "target": cfg["max_calibration_error"],
                              "passed": test_m["calibration_error"] <= cfg["max_calibration_error"]},
    }

    # per-image predictions: val (bias.py fits per-group thresholds on it), test, and any
    # extra splits (e.g. fairness_test for bias.py)
    frames = [("val", val_df, val_probs, val_labels), ("test", test_df, test_probs, test_labels)]
    for split_name, df in (extra or {}).items():
        logits, labels = scorer.logits(df)
        frames.append((split_name, df, softmax(logits, calibration), labels))
    rows = []
    for split_name, df, probs, labels in frames:
        part = df.drop(columns=[c for c in df.columns if c.startswith(("q_", "split"))],
                       errors="ignore").copy()
        part["eval_split"] = split_name
        part["label"] = labels
        for i, cls in enumerate(scorer.classes):
            part[f"p_{cls}"] = probs[:, i]
        part["pred"] = predict_classes(probs, tau, urgent_idx)
        rows.append(part)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    test_preds = predict_classes(test_probs, tau, urgent_idx)
    save_confusion_matrix(test_labels, test_preds, scorer.classes,
                          f"{name}: test (tau {tau:.3f})", out_dir / "confusion_matrix.png")
    save_calibration_plot(test_raw[:, urgent_idx], test_probs[:, urgent_idx],
                          (test_labels == urgent_idx).astype(int), f"{name}: test calibration",
                          out_dir / "calibration.png")

    result = {
        "name": name, "checkpoint": str(checkpoint), "model_kind": scorer.kind,
        "backbone": scorer.backbone, "temperature": temperature,
        "class_offset": dict(zip(scorer.classes, calibration[1].round(4).tolist())), "tau": tau,
        "recall_target": target, "val": val_m, "test": test_m, "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    with mlflow.start_run(run_name=f"evaluate-{name}"):
        mlflow.set_tags(data_version(params))
        mlflow.log_params({"checkpoint": str(checkpoint), "model_kind": scorer.kind,
                           "backbone": scorer.backbone, "val_images": len(val_df),
                           "test_images": len(test_df), "recall_target": target})
        mlflow.log_metrics({"temperature": temperature, "tau": tau,
                            "passed": float(result["passed"])})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items() if not math.isnan(v)})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items() if not math.isnan(v)})
        mlflow.log_artifacts(str(out_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SkinSort model on a hold-out test set")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-split", required=True, help="key under data.splits used to pick tau")
    parser.add_argument("--test-split", required=True, help="key under data.splits held out for testing")
    parser.add_argument("--target", required=True, choices=["urgent_recall_target", "pad_urgent_recall_target"],
                        help="which recall target in params.yaml (evaluate section) applies")
    parser.add_argument("--extra-splits", nargs="*", default=[],
                        help="more data.splits keys to score into predictions.csv (no tau fitting)")
    parser.add_argument("--name", required=True, help="folder name under outputs/evaluation")
    parser.add_argument("--fail-on-target", action="store_true", help="exit 1 if a target is missed")
    args = parser.parse_args()

    params = load_params()
    set_seed(params["seed"])
    device = pick_device()
    splits = params["data"]["splits"]
    val_df = load_table(params, [splits[args.val_split]])
    test_df = load_table(params, [splits[args.test_split]])
    extra = {key: load_table(params, [splits[key]]) for key in args.extra_splits}
    print(f"device {device}, {args.checkpoint}: {len(val_df)} val, {len(test_df)} test images")

    setup_mlflow(params)
    r = evaluate(params, args.checkpoint.resolve(), val_df, test_df, args.target, args.name,
                 device, extra=extra)

    t = r["test"]
    print(f"\ntemperature {r['temperature']:.3f}, offsets {r['class_offset']}, tau {r['tau']:.3f} (picked on {args.val_split} "
          f"for urgent recall >= {r['recall_target']:.2f})")
    print(f"test: urgent recall {t['urgent_recall']:.3f} ({t['urgent_cases']:.0f} urgent), "
          f"precision {t['urgent_precision']:.3f}, flagged {t['flagged_share']:.1%}, "
          f"AUC {t['urgent_auc']:.3f}, macro F1 {t['macro_f1']:.3f}, "
          f"calibration error {t['calibration_error']:.3f} "
          f"(uncalibrated {t['calibration_error_uncalibrated']:.3f})")
    for check, c in r["checks"].items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {check} {c['value']:.3f} (target {c['target']})")
    if args.fail_on_target and not r["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
