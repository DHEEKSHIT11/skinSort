"""Final phone-photo model with a threshold that transfers: out-of-fold tau.

Problem it fixes: picking tau (and the best epoch) on pad_val alone uses only 31
urgent photos, so tau is noisy and the head can fit pad_val; on pad_test the tuned head
caught only 19/35 urgent photos.

Steps:
  1. Pool pad_train + pad_val (209 urgent photos) and split them into K folds grouped by
     patient. Pick the epoch count E as the median best epoch across folds.
  2. Out-of-fold (OOF) scores: for each fold, train a head for exactly E epochs on the
     other folds and score the held-out fold. Every pooled photo is scored by a head
     that never saw it.
  3. Fit calibration (temperature + per-class offset) and tau on the pooled OOF scores.
  4. Train the final head on all pooled photos for E epochs and save it with its
     calibration and tau inside the checkpoint.
  5. Score pad_test once with that calibration and tau.

The image backbone must not have trained on the pooled photos, otherwise the OOF scores
are too optimistic (efficientnet_b3_pad learned from pad_train: image-only urgent AUC
0.84 on pad_train vs 0.74 on pad_val). The checkpoint's "trained_on" list is checked;
older checkpoints without it need --trained-on.

    python -m src.training.fit_final --image-checkpoint outputs/models/efficientnet_b3/best.pt \\
        --trained-on train --settings params --name fusion_oof

Writes outputs/models/<name>/best.pt and outputs/evaluation/<name>/ (metrics.json,
predictions.csv with the OOF rows as "val", plots) so bias.py and explain.py work on it.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from src.models.fusion_model import build_fusion_head
from src.training.dataset import ROOT, answer_feature_columns, load_params, load_table, data_version
from src.training.evaluate import (calibration_error, fit_calibration, pick_tau, predict_classes,
                                   save_calibration_plot, save_confusion_matrix, softmax,
                                   split_metrics)
from src.training.train import pick_device, set_seed, setup_mlflow
from src.training.train_fusion import MODELS_DIR, fit_head
from src.training.tune import OUT_DIR as TUNING_DIR
from src.training.tune import cached_features, with_trial

EVAL_DIR = ROOT / "outputs" / "evaluation"


def head_logits(head: torch.nn.Module, feats: torch.Tensor, answers: torch.Tensor) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(feats, answers).numpy()


def reweighing_weights(groups: np.ndarray, is_urgent: np.ndarray) -> torch.Tensor:
    """Kamiran & Calders (2012) reweighing, the same formula the data pipeline uses
    (Data-Pipeline/scripts/validation/bias.py reweighing_weights):
    weight = P(group) * P(label) / P(group, label), so group and urgency are independent
    in the weighted training data and under-served groups' urgent cases count more."""
    df = pd.DataFrame({"g": groups, "y": is_urgent})
    p_group = df["g"].value_counts(normalize=True)
    p_label = df["y"].value_counts(normalize=True)
    p_joint = df.groupby(["g", "y"]).size() / len(df)
    w = [p_group[g] * p_label[y] / p_joint[(g, y)] for g, y in zip(df["g"], df["y"])]
    return torch.tensor(w, dtype=torch.float32)


def fold_weights(reweight_groups: Optional[np.ndarray], labels: torch.Tensor, rows, urgent_idx: int):
    """Reweighing weights computed on the training rows of one fold only (no leak)."""
    if reweight_groups is None:
        return None
    return reweighing_weights(reweight_groups[rows], (labels[rows] == urgent_idx).numpy().astype(int))


def train_head(params: Dict[str, Any], data, epochs: int, seed: int,
               val_data=None, sample_weights: Optional[torch.Tensor] = None) -> torch.nn.Module:
    """Head trained for exactly `epochs` epochs (no epoch picking)."""
    feats, answers, labels = data
    torch.manual_seed(seed)
    head = build_fusion_head(params, feats.shape[1], answers.shape[1])
    _, state = fit_head(head, data, val_data or data, torch.device("cpu"),
                        params["data"]["classes"].index("urgent"), lr=params["fusion"]["lr"],
                        weight_decay=params["fusion"]["weight_decay"],
                        class_weights=params["fusion"]["class_weights"],
                        batch_size=params["train"]["batch_size"], epochs=epochs, patience=epochs,
                        seed=seed, select_best=False, sample_weights=sample_weights)
    head.load_state_dict(state)
    return head


def out_of_fold(params: Dict[str, Any], data, groups: np.ndarray,
                reweight_groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Pick E by grouped CV, then OOF logits with heads trained for exactly E epochs.
    groups = patient ids for the folds; reweight_groups = optional fairness groups for
    Kamiran & Calders sample weights (computed per fold on its training rows)."""
    cfg = params["tuning"]
    feats, answers, labels = data
    folds = list(StratifiedGroupKFold(n_splits=cfg["cv_folds"], shuffle=True,
                                      random_state=params["seed"]).split(feats.numpy(), labels.numpy(), groups))
    urgent_idx = params["data"]["classes"].index("urgent")

    best_epochs = []
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(params["seed"] + k)
        head = build_fusion_head(params, feats.shape[1], answers.shape[1])
        best, _ = fit_head(head, (feats[tr], answers[tr], labels[tr]), (feats[va], answers[va], labels[va]),
                           torch.device("cpu"), urgent_idx, lr=params["fusion"]["lr"],
                           weight_decay=params["fusion"]["weight_decay"],
                           class_weights=params["fusion"]["class_weights"],
                           batch_size=params["train"]["batch_size"], epochs=cfg["max_epochs"],
                           patience=cfg["patience"], seed=params["seed"] + k,
                           sample_weights=fold_weights(reweight_groups, labels, tr, urgent_idx))
        best_epochs.append(int(best["epoch"]))
    epochs = int(np.median(best_epochs))

    oof = np.zeros((len(labels), len(params["data"]["classes"])))
    for k, (tr, va) in enumerate(folds):
        head = train_head(params, (feats[tr], answers[tr], labels[tr]), epochs, params["seed"] + k,
                          sample_weights=fold_weights(reweight_groups, labels, tr, urgent_idx))
        oof[va] = head_logits(head, feats[va], answers[va])
    return {"epochs": epochs, "fold_best_epochs": best_epochs, "oof_logits": oof}


def reweight_keys(df: pd.DataFrame, columns: Optional[List[str]]) -> Optional[np.ndarray]:
    """One fairness group per row from the bias slices (e.g. sex, or sex + body_site)."""
    if not columns:
        return None
    from src.training.bias import add_slice_columns
    sliced = add_slice_columns(df)
    return sliced[columns].astype(str).agg("|".join, axis=1).to_numpy()


def settings_params(params: Dict[str, Any], settings: str, tuning_dir: Path = TUNING_DIR) -> Dict[str, Any]:
    if settings == "tuned":
        best = json.loads((Path(tuning_dir) / "best_params.json").read_text())["best_params"]
        return with_trial(params, best)
    return with_trial(params, {"lr": params["fusion"]["lr"], "weight_decay": params["train"]["weight_decay"],
                               "dropout": params["fusion"]["dropout"],
                               "hidden_dim": params["fusion"]["hidden_dim"],
                               "urgent_weight": params["train"]["class_weights"][-1]})


def report_oof(params: Dict[str, Any], name: str, pooled_df: pd.DataFrame, oof_probs: np.ndarray,
               labels: np.ndarray, test_df: pd.DataFrame, test_raw: np.ndarray, test_probs: np.ndarray,
               test_labels: np.ndarray, tau: float, calibration, target: float,
               eval_dir: Path = EVAL_DIR, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Metrics, predictions.csv (out-of-fold rows as "val", then "test"), plots and metrics.json
    for a model whose calibration and tau were fitted on out-of-fold scores of the pooled photos."""
    urgent_idx = params["data"]["classes"].index("urgent")
    oof_m = split_metrics(oof_probs, labels, tau, urgent_idx)
    in_val = (pooled_df["split"] == params["data"]["splits"]["pad_val"]).to_numpy()
    oof_val_m = split_metrics(oof_probs[in_val], labels[in_val], tau, urgent_idx)
    test_m = split_metrics(test_probs, test_labels, tau, urgent_idx)
    test_m["calibration_error_uncalibrated"] = calibration_error(
        test_raw[:, urgent_idx], (test_labels == urgent_idx).astype(int))
    max_cal = params["evaluate"]["max_calibration_error"]
    checks = {
        "urgent_recall": {"value": test_m["urgent_recall"], "target": target,
                          "passed": test_m["urgent_recall"] >= target},
        "calibration_error": {"value": test_m["calibration_error"], "target": max_cal,
                              "passed": test_m["calibration_error"] <= max_cal},
    }
    out_dir = Path(eval_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name, df, probs, lab in [("val", pooled_df, oof_probs, labels),
                                       ("test", test_df, test_probs, test_labels)]:
        part = df.drop(columns=[c for c in df.columns if c.startswith(("q_", "split"))], errors="ignore").copy()
        part["eval_split"], part["label"] = split_name, lab
        for i, cls in enumerate(params["data"]["classes"]):
            part[f"p_{cls}"] = probs[:, i]
        part["pred"] = predict_classes(probs, tau, urgent_idx)
        rows.append(part)
    pd.concat(rows, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    save_confusion_matrix(test_labels, predict_classes(test_probs, tau, urgent_idx), params["data"]["classes"],
                          f"{name}: test (tau {tau:.3f})", out_dir / "confusion_matrix.png")
    save_calibration_plot(test_raw[:, urgent_idx], test_probs[:, urgent_idx],
                          (test_labels == urgent_idx).astype(int), f"{name}: test calibration",
                          out_dir / "calibration.png")
    result = {"name": name, **(extra or {}), "threshold_from": "out-of-fold pad_train + pad_val",
              "temperature": calibration[0],
              "class_offset": dict(zip(params["data"]["classes"], np.round(calibration[1], 4).tolist())),
              "tau": tau, "recall_target": target, "val": oof_m, "val_pad_val_rows": oof_val_m,
              "test": test_m, "checks": checks, "passed": all(c["passed"] for c in checks.values())}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def fit_final(params: Dict[str, Any], image_checkpoint: Path, pooled_df: pd.DataFrame, pooled_data,
              test_df: pd.DataFrame, test_data, name: str, settings: str,
              target: float, models_dir: Path = MODELS_DIR, eval_dir: Path = EVAL_DIR,
              reweight: Optional[List[str]] = None) -> Dict[str, Any]:
    urgent_idx = params["data"]["classes"].index("urgent")
    groups = pooled_df["patient_id"].fillna(pooled_df["image_id"]).astype(str).to_numpy()
    labels = pooled_data[2].numpy()
    reweight_groups = reweight_keys(pooled_df, reweight)

    cv = out_of_fold(params, pooled_data, groups, reweight_groups)
    calibration = fit_calibration(cv["oof_logits"], labels)
    oof_probs = softmax(cv["oof_logits"], calibration)
    tau = pick_tau(oof_probs[:, urgent_idx], (labels == urgent_idx).astype(int), target)

    all_rows = np.arange(len(labels))
    head = train_head(params, pooled_data, cv["epochs"], params["seed"],
                      sample_weights=fold_weights(reweight_groups, pooled_data[2], all_rows, urgent_idx))
    test_logits = head_logits(head, test_data[0], test_data[1])
    test_labels = test_data[2].numpy()
    test_probs = softmax(test_logits, calibration)

    test_raw = softmax(test_logits)
    model_dir = Path(models_dir) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "best.pt"
    oof_m = split_metrics(oof_probs, labels, tau, urgent_idx)
    backbone = torch.load(image_checkpoint, map_location="cpu", weights_only=False)["backbone"]
    torch.save({"state_dict": head.state_dict(), "backbone": backbone,
                "image_checkpoint": str(image_checkpoint), "num_image_features": pooled_data[0].shape[1],
                "answer_columns": answer_feature_columns(params), "hidden_dim": params["fusion"]["hidden_dim"],
                "dropout": params["fusion"]["dropout"], "classes": params["data"]["classes"],
                "epoch": cv["epochs"], "val_metrics": oof_m,
                "temperature": calibration[0], "class_offset": calibration[1].tolist(), "tau": tau},
               checkpoint)
    result = report_oof(params, name, pooled_df, oof_probs, labels, test_df, test_raw, test_probs, test_labels,
                        tau, calibration, target, eval_dir,
                        extra={"checkpoint": str(checkpoint), "model_kind": "fusion", "settings": settings,
                               "image_checkpoint": str(image_checkpoint), "epochs": cv["epochs"],
                               "fold_best_epochs": cv["fold_best_epochs"], "reweight": reweight or []})
    out_dir = Path(eval_dir) / name
    oof_val_m, test_m = result["val_pad_val_rows"], result["test"]

    with mlflow.start_run(run_name=f"fit-final-{name}"):
        mlflow.set_tags(data_version(params))
        mlflow.log_params({"image_checkpoint": str(image_checkpoint), "settings": settings,
                           "reweight": ",".join(reweight or []) or "none",
                           "epochs": cv["epochs"], "pooled_images": len(pooled_df),
                           "pooled_urgent": int((labels == urgent_idx).sum()), "recall_target": target,
                           **{f"fusion_{k}": v for k, v in params["fusion"].items()}})
        mlflow.log_metrics({"temperature": calibration[0], "tau": tau, "passed": float(result["passed"])})
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_m.items() if not math.isnan(v)})
        mlflow.log_metrics({f"oof_pad_val_{k}": v for k, v in oof_val_m.items() if not math.isnan(v)})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items() if not math.isnan(v)})
        mlflow.log_artifacts(str(out_dir))
        mlflow.log_artifact(str(checkpoint))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion head with an out-of-fold threshold")
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-on", nargs="*",
                        help="splits the image model learned from (only for checkpoints without 'trained_on')")
    parser.add_argument("--tuning-dir", type=Path, default=TUNING_DIR,
                        help="folder with best_params.json for --settings tuned")
    parser.add_argument("--settings", choices=["params", "tuned"], default="params",
                        help="params.yaml fusion settings, or outputs/tuning/best_params.json")
    parser.add_argument("--name", required=True)
    parser.add_argument("--reweight", nargs="*", default=[],
                        help="bias slices to reweigh by (Kamiran & Calders), e.g. sex body_site")
    parser.add_argument("--cv-only", action="store_true",
                        help="stop after the out-of-fold scores (no final head, pad_test untouched)")
    args = parser.parse_args()

    params = load_params()
    set_seed(params["seed"])
    device = pick_device()
    splits = params["data"]["splits"]
    checkpoint = args.image_checkpoint.resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trained_on: Optional[List[str]] = ckpt.get("trained_on", args.trained_on)
    if trained_on is None:
        sys.exit(f"{checkpoint} does not record which splits it trained on; pass --trained-on")
    pooled_names = [splits["pad_train"], splits["pad_val"]]
    if set(trained_on) & set(pooled_names):
        sys.exit(f"{checkpoint} trained on {sorted(set(trained_on) & set(pooled_names))}: "
                 "out-of-fold scores would be too optimistic. Use a backbone that never saw these photos.")

    p = settings_params(params, args.settings, args.tuning_dir)
    train_df = load_table(params, [splits["pad_train"]])
    val_df = load_table(params, [splits["pad_val"]])
    pooled_df = pd.concat([train_df, val_df], ignore_index=True)
    tr, va = (cached_features(params, checkpoint, d, s, device) for d, s in
              [(train_df, "pad_train"), (val_df, "pad_val")])
    pooled_data = tuple(torch.cat([a, b]) for a, b in zip(tr, va))
    target = params["evaluate"]["pad_urgent_recall_target"]
    print(f"{len(pooled_df)} pooled photos ({int((pooled_data[2] == 2).sum())} urgent), "
          f"settings {args.settings}: {p['fusion']}")

    setup_mlflow(params)
    if args.cv_only:
        groups = pooled_df["patient_id"].fillna(pooled_df["image_id"]).astype(str).to_numpy()
        cv = out_of_fold(p, pooled_data, groups, reweight_keys(pooled_df, args.reweight))
        labels = pooled_data[2].numpy()
        probs = softmax(cv["oof_logits"], fit_calibration(cv["oof_logits"], labels))
        in_val = (pooled_df["split"] == splits["pad_val"]).to_numpy()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels == 2, probs[:, 2])
        auc_val = roc_auc_score(labels[in_val] == 2, probs[in_val, 2])
        print(f"epochs {cv['epochs']} (folds {cv['fold_best_epochs']}): OOF urgent AUC {auc:.3f}, "
              f"on pad_val rows {auc_val:.3f}")
        return

    test_df = load_table(params, [splits["pad_test"]])
    test_data = cached_features(params, checkpoint, test_df, "pad_test", device)
    r = fit_final(p, checkpoint, pooled_df, pooled_data, test_df, test_data, args.name, args.settings, target,
                  reweight=args.reweight)
    o, ov, t = r["val"], r["val_pad_val_rows"], r["test"]
    print(f"\nepochs {r['epochs']} (fold best {r['fold_best_epochs']}), temperature {r['temperature']:.3f}, "
          f"tau {r['tau']:.3f} (out-of-fold, {o['urgent_cases']:.0f} urgent)")
    print(f"OOF all:      urgent AUC {o['urgent_auc']:.3f}, recall {o['urgent_recall']:.3f}, flagged {o['flagged_share']:.1%}")
    print(f"OOF pad_val:  urgent AUC {ov['urgent_auc']:.3f}, recall {ov['urgent_recall']:.3f}, flagged {ov['flagged_share']:.1%}")
    print(f"test: urgent recall {t['urgent_recall']:.3f} ({t['urgent_cases']:.0f} urgent), precision "
          f"{t['urgent_precision']:.3f}, flagged {t['flagged_share']:.1%}, AUC {t['urgent_auc']:.3f}, "
          f"calibration error {t['calibration_error']:.3f}")
    for check, c in r["checks"].items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {check} {c['value']:.3f} (target {c['target']})")


if __name__ == "__main__":
    main()
