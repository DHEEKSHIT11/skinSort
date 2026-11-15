"""Hyperparameter tuning of the fusion head with Hyperopt (TPE, Bayesian search).

Follows the course MLflow lab pattern: one parent MLflow run, one nested child run per
trial, best trial picked with fmin/TPE. Only the small fusion head is tuned: the image
features are computed once with the frozen image model and cached in outputs/features/.

Each trial is scored with 5-fold cross-validation on pad_train, grouped by patient so
one patient's photos never sit in both train and validation folds. pad_val (only 31
urgent cases) is not used during the search, so it stays an honest check for the final
model; pad_test is not touched at all.

Search space (params.yaml, tuning section): lr, weight_decay, dropout (continuous),
urgent class weight, hidden_dim (choice).

    python -m src.training.tune --image-checkpoint outputs/models/efficientnet_b3_pad/best.pt
    python -m src.training.tune --image-checkpoint ... --trials 5     # quick run

Writes outputs/tuning/: trials.csv, best_params.json, one sensitivity plot per
hyperparameter. Then retrains the best setting on all of pad_train (best epoch on
pad_val) and saves outputs/models/fusion_tuned/best.pt.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from src.models.fusion_model import build_fusion_head
from src.training.dataset import ROOT, load_params, load_table, data_version
from src.training.train import pick_device, set_seed, setup_mlflow
from src.training.train_fusion import (MODELS_DIR, extract_features, fit_head,
                                       load_image_model, train_fusion)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURES_DIR = ROOT / "outputs" / "features"
OUT_DIR = ROOT / "outputs" / "tuning"
Tensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def cached_features(params: Dict[str, Any], checkpoint: Path, df: pd.DataFrame, split: str,
                    device: torch.device, cache_dir: Path = FEATURES_DIR) -> Tensors:
    """Frozen image features for a split, cached per checkpoint file + rows."""
    key = hashlib.md5(f"{checkpoint.resolve()}|{checkpoint.stat().st_mtime}|"
                      f"{','.join(df['image_id'])}".encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{checkpoint.parent.name}_{split}_{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    model = load_image_model(params, checkpoint, device)
    data = extract_features(model, df, params, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def search_space(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lr": hp.loguniform("lr", math.log(cfg["lr"][0]), math.log(cfg["lr"][1])),
        "weight_decay": hp.loguniform("weight_decay", math.log(cfg["weight_decay"][0]),
                                      math.log(cfg["weight_decay"][1])),
        "dropout": hp.uniform("dropout", *cfg["dropout"]),
        "urgent_weight": hp.uniform("urgent_weight", *cfg["urgent_weight"]),
        "hidden_dim": hp.choice("hidden_dim", cfg["hidden_dim"]),
    }


def with_trial(params: Dict[str, Any], trial: Dict[str, Any]) -> Dict[str, Any]:
    """params.yaml with one trial's values written into the fusion section."""
    p = copy.deepcopy(params)
    weights = list(p["train"]["class_weights"])
    weights[p["data"]["classes"].index("urgent")] = float(trial["urgent_weight"])
    p["fusion"].update(lr=float(trial["lr"]), weight_decay=float(trial["weight_decay"]),
                       dropout=float(trial["dropout"]), hidden_dim=int(trial["hidden_dim"]),
                       class_weights=weights)
    return p


def cross_validate(params: Dict[str, Any], data: Tensors, groups: np.ndarray,
                   device: torch.device) -> Dict[str, float]:
    """Mean / std urgent AUC over grouped, stratified folds of the training split."""
    cfg = params["tuning"]
    feats, answers, labels = data
    folds = StratifiedGroupKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=params["seed"])
    urgent_idx = params["data"]["classes"].index("urgent")
    aucs, epochs = [], []
    for fold, (tr, va) in enumerate(folds.split(feats.numpy(), labels.numpy(), groups)):
        torch.manual_seed(params["seed"] + fold)
        head = build_fusion_head(params, feats.shape[1], answers.shape[1])
        best, _ = fit_head(head, (feats[tr], answers[tr], labels[tr]),
                           (feats[va], answers[va], labels[va]), device, urgent_idx,
                           lr=params["fusion"]["lr"], weight_decay=params["fusion"]["weight_decay"],
                           class_weights=params["fusion"]["class_weights"],
                           batch_size=params["train"]["batch_size"], epochs=cfg["max_epochs"],
                           patience=cfg["patience"], seed=params["seed"] + fold)
        aucs.append(best["urgent_auc"])
        epochs.append(best["epoch"])
    return {"cv_urgent_auc": float(np.mean(aucs)), "cv_urgent_auc_std": float(np.std(aucs)),
            "cv_best_epoch_median": float(np.median(epochs))}


def run_search(params: Dict[str, Any], data: Tensors, groups: np.ndarray, device: torch.device,
               n_trials: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Hyperopt TPE search; each trial is a nested MLflow run. Returns best trial + table."""
    cfg = params["tuning"]
    rows: List[Dict[str, Any]] = []

    def objective(trial: Dict[str, Any]) -> Dict[str, Any]:
        trial_params = with_trial(params, trial)
        with mlflow.start_run(run_name=f"trial-{len(rows) + 1}", nested=True):
            mlflow.set_tags(data_version(params))
            mlflow.log_params({k: (int(v) if k == "hidden_dim" else float(v)) for k, v in trial.items()})
            result = cross_validate(trial_params, data, groups, device)
            mlflow.log_metrics(result)
        rows.append({"trial": len(rows) + 1, **{k: float(v) for k, v in trial.items()}, **result})
        print(f"trial {len(rows):2d}  cv urgent AUC {result['cv_urgent_auc']:.3f} "
              f"± {result['cv_urgent_auc_std']:.3f}  "
              + "  ".join(f"{k} {float(v):.3g}" for k, v in trial.items()))
        return {"loss": -result["cv_urgent_auc"], "status": STATUS_OK}

    space = search_space(cfg)
    trials = Trials()
    fmin(objective, space, algo=tpe.suggest, max_evals=n_trials, trials=trials,
         rstate=np.random.default_rng(params["seed"]), show_progressbar=False)
    table = pd.DataFrame(rows)
    best = table.loc[table["cv_urgent_auc"].idxmax()].to_dict()
    return best, table


def sensitivity_plots(table: pd.DataFrame, names: List[str], out_dir: Path) -> Dict[str, float]:
    """CV AUC against each hyperparameter, with the Spearman correlation as a sensitivity score."""
    corr = {}
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        rho = spearmanr(table[name], table["cv_urgent_auc"]).statistic if table[name].nunique() > 1 else float("nan")
        corr[name] = float(rho)
        ax.errorbar(table[name], table["cv_urgent_auc"], yerr=table["cv_urgent_auc_std"],
                    fmt="o", ms=4, alpha=0.7, color="#3b6fb6", ecolor="#b8c2cc")
        best = table.loc[table["cv_urgent_auc"].idxmax()]
        ax.plot(best[name], best["cv_urgent_auc"], "*", ms=12, color="#d9822b")
        if name in ("lr", "weight_decay"):
            ax.set_xscale("log")
        ax.set_xlabel(name)
        ax.set_title(f"Spearman {rho:+.2f}", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("CV urgent AUC (pad_train)")
    fig.suptitle("Hyperparameter sensitivity (each dot = one Hyperopt trial, star = best)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity.png", dpi=120)
    plt.close(fig)

    # search progress: does TPE find better settings over time?
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(table["trial"], table["cv_urgent_auc"], "o", color="#3b6fb6", ms=4, label="trial")
    ax.plot(table["trial"], table["cv_urgent_auc"].cummax(), "-", color="#d9822b", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV urgent AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "search_progress.png", dpi=120)
    plt.close(fig)
    return corr


def lr_sweep(params: Dict[str, Any], data: Tensors, groups: np.ndarray, device: torch.device,
             out_dir: Path, n: int = 7) -> pd.DataFrame:
    """One-at-a-time sensitivity: CV AUC for a grid of learning rates, everything else at the
    params.yaml settings (TPE samples are not independent, so this is the cleaner read)."""
    lo, hi = params["tuning"]["lr"]
    rows = []
    for lr in np.geomspace(lo, hi, n):
        trial = {"lr": float(lr), "weight_decay": params["train"]["weight_decay"],
                 "dropout": params["fusion"]["dropout"], "hidden_dim": params["fusion"]["hidden_dim"],
                 "urgent_weight": params["train"]["class_weights"][-1]}
        r = cross_validate(with_trial(params, trial), data, groups, device)
        rows.append({"lr": float(lr), **r})
        print(f"lr {lr:.2e}  cv urgent AUC {r['cv_urgent_auc']:.3f} ± {r['cv_urgent_auc_std']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "lr_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(table["lr"], table["cv_urgent_auc"], yerr=table["cv_urgent_auc_std"], fmt="o-", color="#3b6fb6")
    ax.axvline(params["fusion"]["lr"], color="#d9822b", linestyle="--", linewidth=1, label="params.yaml lr")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (other settings fixed)")
    ax.set_ylabel("CV urgent AUC (pad_train)")
    ax.set_title("Learning-rate sensitivity, one at a time", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "lr_sweep.png", dpi=130)
    plt.close(fig)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperopt tuning of the fusion head")
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--trials", type=int, help="override tuning.n_trials")
    parser.add_argument("--no-retrain", action="store_true", help="skip the final retrain")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where trials and plots go")
    parser.add_argument("--lr-sweep", action="store_true",
                        help="only run the one-at-a-time learning-rate sweep (no search)")
    parser.add_argument("--reuse-trials", action="store_true",
                        help="skip the search and use outputs/tuning/trials.csv from an earlier run")
    args = parser.parse_args()

    params = load_params()
    set_seed(params["seed"])
    device = pick_device()
    splits = params["data"]["splits"]
    checkpoint = args.image_checkpoint.resolve()
    n_trials = args.trials or params["tuning"]["n_trials"]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_table(params, [splits["pad_train"]])
    data = cached_features(params, checkpoint, train_df, "pad_train", device)
    groups = train_df["patient_id"].fillna(train_df["image_id"]).astype(str).to_numpy()
    # the head is tiny: the CPU is faster than moving small batches to the GPU
    tune_device = torch.device("cpu")
    print(f"{len(train_df)} pad_train photos, {len(set(groups))} patients, "
          f"{params['tuning']['cv_folds']}-fold CV, {n_trials} trials")

    setup_mlflow(params)
    if args.lr_sweep:
        with mlflow.start_run(run_name="lr-sweep"):
            mlflow.set_tags(data_version(params))
            mlflow.log_param("image_checkpoint", str(checkpoint))
            table = lr_sweep(params, data, groups, tune_device, out_dir)
            for r in table.itertuples():
                mlflow.log_metric("cv_urgent_auc", r.cv_urgent_auc, step=int(round(np.log10(r.lr) * 100)))
            mlflow.log_artifact(str(out_dir / "lr_sweep.png"))
            mlflow.log_artifact(str(out_dir / "lr_sweep.csv"))
        return
    with mlflow.start_run(run_name="tune-fusion"):
        mlflow.set_tags(data_version(params))
        mlflow.log_params({"image_checkpoint": str(checkpoint), "n_trials": n_trials,
                           "cv_folds": params["tuning"]["cv_folds"], "algo": "hyperopt-tpe",
                           "search_space": json.dumps({k: v for k, v in params["tuning"].items()
                                                       if isinstance(v, list)})})
        if args.reuse_trials:
            table = pd.read_csv(out_dir / "trials.csv")
            best = table.loc[table["cv_urgent_auc"].idxmax()].to_dict()
            n_trials = len(table)
            mlflow.set_tag("reused_trials", "true")
        else:
            best, table = run_search(params, data, groups, tune_device, n_trials)
        names = ["lr", "weight_decay", "dropout", "urgent_weight", "hidden_dim"]
        corr = sensitivity_plots(table, names, out_dir)
        table.to_csv(out_dir / "trials.csv", index=False)

        # baseline = the current params.yaml fusion settings, scored the same way
        baseline = cross_validate(with_trial(params, {
            "lr": params["fusion"]["lr"], "weight_decay": params["train"]["weight_decay"],
            "dropout": params["fusion"]["dropout"], "hidden_dim": params["fusion"]["hidden_dim"],
            "urgent_weight": params["train"]["class_weights"][-1]}), data, groups, tune_device)
        summary = {"image_checkpoint": str(checkpoint), "n_trials": n_trials,
                   "best_trial": int(best["trial"]),
                   "best_params": {k: (int(best[k]) if k == "hidden_dim" else float(best[k])) for k in names},
                   "best_cv_urgent_auc": best["cv_urgent_auc"],
                   "best_cv_urgent_auc_std": best["cv_urgent_auc_std"],
                   "baseline_cv_urgent_auc": baseline["cv_urgent_auc"],
                   "baseline_cv_urgent_auc_std": baseline["cv_urgent_auc_std"],
                   "sensitivity_spearman": corr}
        mlflow.log_metrics({"best_cv_urgent_auc": best["cv_urgent_auc"],
                            "baseline_cv_urgent_auc": baseline["cv_urgent_auc"]})
        mlflow.log_params({f"best_{k}": v for k, v in summary["best_params"].items()})

        if not args.no_retrain:
            val_df = load_table(params, [splits["pad_val"]])
            final = train_fusion(with_trial(params, summary["best_params"]), checkpoint, train_df,
                                 val_df, device, epochs=params["tuning"]["max_epochs"],
                                 out_dir=MODELS_DIR / "fusion_tuned")
            summary["retrained"] = {"checkpoint": final["checkpoint"], "best_epoch": final["epoch"],
                                    "pad_val_urgent_auc": final["urgent_auc"],
                                    "pad_val_image_only_auc": final["image_only"]["urgent_auc"]}
            mlflow.log_metric("retrained_pad_val_urgent_auc", final["urgent_auc"])
        (out_dir / "best_params.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifacts(str(out_dir))

    print(f"\nbest trial {summary['best_trial']}: cv urgent AUC {summary['best_cv_urgent_auc']:.3f} "
          f"± {summary['best_cv_urgent_auc_std']:.3f} (params.yaml settings "
          f"{summary['baseline_cv_urgent_auc']:.3f} ± {summary['baseline_cv_urgent_auc_std']:.3f})")
    print("best params:", summary["best_params"])
    print("sensitivity (Spearman with CV AUC):", {k: round(v, 2) for k, v in corr.items()})
    if "retrained" in summary:
        r = summary["retrained"]
        print(f"retrained on all pad_train: pad_val urgent AUC {r['pad_val_urgent_auc']:.3f} "
              f"(epoch {r['best_epoch']}), saved {r['checkpoint']}")


if __name__ == "__main__":
    main()
