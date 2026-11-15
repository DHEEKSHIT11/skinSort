"""Model selection after validation AND bias checking (assignment §2.2, §4, §8.3, §8.6).

Follows the course MLflow lab pattern: every candidate is an MLflow run and the best one
is found with mlflow.search_runs(order_by=[... DESC]). Two stages:

Stage 1 — image backbone (dermoscopy). The training runs of the backbones in params.yaml
(smoke runs excluded) are ranked by validation urgent AUC. A size / CPU-speed table is
added because deployment constraints matter when picking a model (course Session 10).

Stage 2 — phone triage model (photo + answers). Candidates are the saved evaluations
in params.yaml (selection.candidates) plus three baselines built here (course Session 10,
"establish a baseline"):
    baseline_prevalence   every photo gets the urgent base rate (AUC 0.5 by construction)
    baseline_answers_lr   logistic regression on the 6 answers + missing flags only
    baseline_image_only   the dermoscopy model's own head on the phone photos, no answers
Every candidate is scored the same way on the pad_val photos, logged as a nested MLflow
run, then:
    1. hard filters: validation calibration error <= evaluate.max_calibration_error, and
       tau fitted on >= selection.min_tau_urgent_cases urgent photos (31 is too few: the
       recall at tau has a ~±0.15 95% interval)
    2. ranking: search_runs(order_by=["metrics.val_urgent_auc DESC"])
    3. bias: if a candidate whose bias check passes is within selection.max_auc_drop_for_bias
       of the best, it is chosen instead; otherwise the best is chosen with a bias warning
Test metrics are logged for the report only; they never enter the selection.

    python -m src.training.select_model

Writes outputs/selection/: model_selection.json / .md, backbone_comparison.png,
phone_comparison.png. registry.py register --from-selection uses model_selection.json.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.models.image_model import build_image_model, count_parameters
from src.training.bias import add_slice_columns, slice_report
from src.training.dataset import ROOT, data_version, load_params, load_table
from src.training.evaluate import calibration_error, fit_calibration, pick_tau, softmax
from src.training.fit_final import EVAL_DIR, report_oof
from src.training.train import pick_device, set_seed, setup_mlflow
from src.training.tune import cached_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "outputs" / "selection"
MODELS_DIR = ROOT / "outputs" / "models"


# ------------------------------------------------------------ stage 1: backbones

def select_backbone(params: Dict[str, Any], models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Best run per backbone from MLflow, ranked by validation urgent AUC (course pattern).

    The backbones are trained on the GPU VM, so a fresh MLflow DB (e.g. a CI runner) has no
    training runs; then the same ranking is done on the validation metrics saved inside
    outputs/models/<backbone>/best.pt."""
    cols = {"tags.mlflow.runName": "backbone", "run_id": "run_id", "metrics.best_urgent_auc": "val_urgent_auc",
            "metrics.best_urgent_recall": "val_urgent_recall", "metrics.best_macro_f1": "val_macro_f1"}
    runs = mlflow.search_runs(experiment_names=[params["mlflow"]["experiment_name"]],
                              filter_string="attributes.status = 'FINISHED'",
                              order_by=["metrics.best_urgent_auc DESC"])
    if "tags.mlflow.runName" in runs and "params.train_images" in runs:
        runs = runs[runs["tags.mlflow.runName"].isin(params["train"]["backbones"])]
        runs = runs[pd.to_numeric(runs["params.train_images"], errors="coerce") > params["train"]["smoke_test_rows"]]
        runs = runs.drop_duplicates("tags.mlflow.runName")       # best run per backbone (already sorted)
        if not runs.empty:
            out = runs[list(cols)].rename(columns=cols).reset_index(drop=True)
            out["source"] = "mlflow"
            return out
    rows = []
    for name in params["train"]["backbones"]:
        path = models_dir / name / "best.pt"
        if path.exists():
            m = torch.load(path, map_location="cpu", weights_only=False)["val_metrics"]
            rows.append({"backbone": name, "run_id": "", "val_urgent_auc": m["urgent_auc"],
                         "val_urgent_recall": m["urgent_recall"], "val_macro_f1": m["macro_f1"],
                         "source": "checkpoint"})
    if not rows:
        raise FileNotFoundError(f"no backbone training runs in MLflow and no checkpoints in {models_dir}")
    return pd.DataFrame(rows).sort_values("val_urgent_auc", ascending=False).reset_index(drop=True)


def size_and_speed(params: Dict[str, Any], backbones: List[str], repeats: int = 5) -> pd.DataFrame:
    """Parameters, checkpoint size and CPU time per photo at the training image size."""
    rows = []
    size = params["train"]["image_size"]
    torch.set_num_threads(4)
    for name in backbones:
        model = build_image_model(params, name, pretrained=False).eval()
        ckpt = MODELS_DIR / name / "best.pt"
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            model(x)                                               # warm-up
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - start)
        rows.append({"backbone": name, "parameters_m": count_parameters(model) / 1e6,
                     "checkpoint_mb": ckpt.stat().st_size / 1e6 if ckpt.exists() else float("nan"),
                     "cpu_ms_per_photo": 1000 * float(np.median(times))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ stage 2: baselines

def pooled_phone_data(params: Dict[str, Any], image_checkpoint: Path, device: torch.device):
    splits = params["data"]["splits"]
    train_df = load_table(params, [splits["pad_train"]])
    val_df = load_table(params, [splits["pad_val"]])
    test_df = load_table(params, [splits["pad_test"]])
    tr = cached_features(params, image_checkpoint, train_df, "pad_train", device)
    va = cached_features(params, image_checkpoint, val_df, "pad_val", device)
    te = cached_features(params, image_checkpoint, test_df, "pad_test", device)
    pooled_df = pd.concat([train_df, val_df], ignore_index=True)
    pooled = tuple(torch.cat([a, b]) for a, b in zip(tr, va))
    return pooled_df, pooled, test_df, te


def build_baselines(params: Dict[str, Any], image_checkpoint: Path, device: torch.device,
                    eval_dir: Path = EVAL_DIR) -> List[str]:
    """The three baselines, each with calibration + tau from the pooled phone photos."""
    from src.training.train_fusion import load_image_model

    urgent_idx = params["data"]["classes"].index("urgent")
    target = params["evaluate"]["pad_urgent_recall_target"]
    pooled_df, (p_feats, p_answers, p_labels), test_df, (t_feats, t_answers, t_labels) = \
        pooled_phone_data(params, image_checkpoint, device)
    labels, test_labels = p_labels.numpy(), t_labels.numpy()
    is_urgent = (labels == urgent_idx).astype(int)
    names = []

    def finish(name: str, oof_logits: np.ndarray, test_logits: np.ndarray, note: str,
               calibrate: bool = True) -> None:
        calibration = fit_calibration(oof_logits, labels) if calibrate else (1.0, np.zeros(oof_logits.shape[1]))
        oof_probs = softmax(oof_logits, calibration)
        tau = pick_tau(oof_probs[:, urgent_idx], is_urgent, target)
        report_oof(params, name, pooled_df, oof_probs, labels, test_df, softmax(test_logits),
                   softmax(test_logits, calibration), test_labels, tau, calibration, target, eval_dir,
                   extra={"model_kind": "baseline", "note": note})
        names.append(name)

    # 1. prevalence: same score for every photo
    prior = np.log(np.bincount(labels, minlength=3) / len(labels))
    finish("baseline_prevalence", np.tile(prior, (len(labels), 1)), np.tile(prior, (len(test_labels), 1)),
           "every photo gets the class base rates", calibrate=False)

    # 2. answers only: logistic regression, out-of-fold on the pooled photos
    groups = pooled_df["patient_id"].fillna(pooled_df["image_id"]).astype(str).to_numpy()
    folds = StratifiedGroupKFold(n_splits=params["tuning"]["cv_folds"], shuffle=True,
                                 random_state=params["seed"])
    weights = dict(enumerate(params["train"]["class_weights"]))
    x, xt = p_answers.numpy(), t_answers.numpy()
    oof = np.zeros((len(labels), 3))
    for tr, va in folds.split(x, labels, groups):
        lr = LogisticRegression(max_iter=2000, class_weight=weights).fit(x[tr], labels[tr])
        oof[va] = lr.predict_log_proba(x[va])
    lr = LogisticRegression(max_iter=2000, class_weight=weights).fit(x, labels)
    finish("baseline_answers_lr", oof, lr.predict_log_proba(xt), "logistic regression on the answers only")

    # 3. image only: the dermoscopy model's head, never trained on phone photos (no OOF needed)
    model = load_image_model(params, image_checkpoint, torch.device("cpu"))
    with torch.no_grad():
        finish("baseline_image_only", model.head(p_feats).numpy(), model.head(t_feats).numpy(),
               "dermoscopy image model, no answers, no phone-photo training")
    return names


# ------------------------------------------------------------ stage 2: scoring

def candidate_summary(params: Dict[str, Any], name: str, eval_dir: Path = EVAL_DIR) -> Dict[str, Any]:
    """Same validation numbers for every candidate, on the pad_val photos."""
    metrics = json.loads((eval_dir / name / "metrics.json").read_text())
    preds = pd.read_csv(eval_dir / name / "predictions.csv")
    val = preds[preds["eval_split"] == "val"]
    pad_val_ids = set(load_table(params, [params["data"]["splits"]["pad_val"]])["image_id"])
    on_pad_val = val[val["image_id"].isin(pad_val_ids)]
    urgent = (on_pad_val["label"] == 2).astype(int)
    flagged = (on_pad_val["pred"] == 2).astype(int)
    bias = [r for r in (slice_report(add_slice_columns(val), col, params["bias"]["min_urgent_cases"],
                                     params["bias"]["max_recall_gap"]) for col in params["bias"]["slices"])
            if r is not None]
    statuses = {r["slice"]: r["status"] for r in bias}
    bias_result = ("FAIL" if "FAIL" in statuses.values()
                   else "PASS" if "PASS" in statuses.values() else "INCONCLUSIVE")
    kind = metrics.get("model_kind", "fusion")
    return {
        "name": name, "kind": kind,
        "tau_fitted_on": "out-of-fold pad_train + pad_val" if len(val) > len(on_pad_val) else "pad_val",
        "tau_urgent_cases": int((val["label"] == 2).sum()),
        "val_urgent_auc": float(roc_auc_score(urgent, on_pad_val["p_urgent"])),
        "val_calibration_error": calibration_error(on_pad_val["p_urgent"].to_numpy(), urgent.to_numpy()),
        "val_urgent_recall": float((flagged & urgent).sum() / max(urgent.sum(), 1)),
        "val_flagged_share": float(flagged.mean()),
        "bias_result": bias_result, "bias_slices": statuses,
        "bias_failed_slices": sum(v == "FAIL" for v in statuses.values()),
        "test_urgent_auc": metrics["test"]["urgent_auc"], "test_urgent_recall": metrics["test"]["urgent_recall"],
        "test_calibration_error": metrics["test"]["calibration_error"],
        "checkpoint": metrics.get("checkpoint", ""), "eval_dir": str(eval_dir / name),
    }


def choose(table: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filters -> best validation AUC -> bias consideration. table is already sorted by AUC.

    Bias: among eligible candidates within selection.max_auc_drop_for_bias of the best AUC,
    the one with the fewest failing bias slices wins (ties: higher AUC)."""
    cfg = params["selection"]
    max_cal = params["evaluate"]["max_calibration_error"]
    table = table.copy()
    if "bias_failed_slices" not in table:
        table["bias_failed_slices"] = (table["bias_result"] == "FAIL").astype(int)
    table["passes_calibration"] = table["val_calibration_error"] <= max_cal
    table["passes_tau_support"] = table["tau_urgent_cases"] >= cfg["min_tau_urgent_cases"]
    table["eligible"] = table["passes_calibration"] & table["passes_tau_support"] & table["servable"]
    eligible = table[table["eligible"]]
    if eligible.empty:
        return {"selected": None, "reason": "no candidate passes the validation filters", "table": table}
    best = eligible.iloc[0]
    close = eligible[eligible["val_urgent_auc"] >= best["val_urgent_auc"] - cfg["max_auc_drop_for_bias"]]
    chosen = close.sort_values(["bias_failed_slices", "val_urgent_auc"], ascending=[True, False]).iloc[0]
    n_fail = int(chosen["bias_failed_slices"])
    if chosen["name"] == best["name"]:
        reason = "best validation AUC" + ("" if n_fail == 0 else f"; {n_fail} bias slice(s) still fail (bias warning)")
    else:
        reason = (f"fewest failing bias slices ({n_fail} vs {int(best['bias_failed_slices'])} for {best['name']}) "
                  f"within {cfg['max_auc_drop_for_bias']} validation AUC of the best")
    return {"selected": chosen["name"], "reason": reason, "table": table}


# ------------------------------------------------------------ outputs

def save_charts(backbones: pd.DataFrame, speed: pd.DataFrame, table: pd.DataFrame, selected: Optional[str],
                out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    b = backbones.merge(speed, on="backbone")
    colors = ["#d9822b" if i == 0 else "#3b6fb6" for i in range(len(b))]
    axes[0].bar(b["backbone"], b["val_urgent_auc"], color=colors)
    for x, v in enumerate(b["val_urgent_auc"]):
        axes[0].annotate(f"{v:.3f}", (x, v), ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("val urgent AUC (dermoscopy)")
    axes[0].set_title("Image backbones (orange = selected)", fontsize=10)
    if b["cpu_ms_per_photo"].notna().any():
        axes[1].bar(b["backbone"], b["cpu_ms_per_photo"], color=colors)
        for x, (ms, p) in enumerate(zip(b["cpu_ms_per_photo"], b["parameters_m"])):
            axes[1].annotate(f"{ms:.0f} ms\n{p:.1f}M params", (x, ms), ha="center", va="bottom", fontsize=8)
        axes[1].set_ylim(0, b["cpu_ms_per_photo"].max() * 1.35)
    else:
        axes[1].text(0.5, 0.5, "not measured (--no-speed)", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_ylabel("CPU ms per photo")
    axes[1].set_title("Size and speed (deployment cost)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "backbone_comparison.png", dpi=130)
    plt.close(fig)

    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(t) + 1.6))
    colors = ["#d9822b" if n == selected else ("#3b6fb6" if e else "#b8c2cc")
              for n, e in zip(t["name"], t["eligible"])]
    ax.barh(t["name"], t["val_urgent_auc"], color=colors)
    ax.scatter(t["test_urgent_auc"], t["name"], marker="|", s=200, color="#444", label="test AUC (report only)")
    for y, (v, row) in enumerate(zip(t["val_urgent_auc"], t.itertuples())):
        why = [] if row.eligible else (["calibration"] * (not row.passes_calibration)
                                       + [f"tau from {row.tau_urgent_cases} urgent"] * (not row.passes_tau_support)
                                       + ["not servable"] * (not row.servable))
        label = f"{v:.3f}  bias {row.bias_result} ({row.bias_failed_slices} failing)" + (f"  ✗ {', '.join(why)}" if why else "")
        ax.annotate(label, (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xlim(0.4, 1.15)
    ax.set_xlabel("urgent AUC on pad_val photos (selection metric)")
    ax.set_title("Phone triage candidates (orange = selected, grey = filtered out)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "phone_comparison.png", dpi=130)
    plt.close(fig)


def to_markdown(backbones: pd.DataFrame, speed: pd.DataFrame, table: pd.DataFrame,
                decision: Dict[str, Any], params: Dict[str, Any]) -> str:
    b = backbones.merge(speed, on="backbone")
    lines = ["# Model selection", "",
             "Selection uses validation data only. Test columns are shown for the report and never used to choose.", "",
             "## Stage 1: image backbone (`mlflow.search_runs(order_by=['metrics.best_urgent_auc DESC'])`)", "",
             "| Backbone | Val urgent AUC | Val urgent recall | Val macro F1 | Params (M) | Checkpoint MB | CPU ms/photo |",
             "|---|---|---|---|---|---|---|"]
    for r in b.itertuples():
        lines.append(f"| {r.backbone}{' ✅' if r.Index == 0 else ''} | {r.val_urgent_auc:.3f} | {r.val_urgent_recall:.3f} | "
                     f"{r.val_macro_f1:.3f} | {r.parameters_m:.1f} | {r.checkpoint_mb:.0f} | {r.cpu_ms_per_photo:.0f} |")
    cfg = params["selection"]
    lines += ["", f"Selected: **{b.iloc[0]['backbone']}** (highest validation urgent AUC).", "",
              "## Stage 2: phone triage model (`order_by=['metrics.val_urgent_auc DESC']`)", "",
              f"Filters: validation calibration error ≤ {params['evaluate']['max_calibration_error']}; tau fitted on "
              f"≥ {cfg['min_tau_urgent_cases']} urgent photos; servable as a registry model. Bias: among candidates "
              f"within {cfg['max_auc_drop_for_bias']} AUC of the best, the one with the fewest failing bias slices wins.", "",
              "| Candidate | Kind | Val AUC | Val calibration | Tau from (urgent) | Bias (failing slices) "
              "| Eligible | Test AUC | Test recall |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in decision["table"].itertuples():
        mark = " ✅" if r.name == decision["selected"] else ""
        lines.append(f"| {r.name}{mark} | {r.kind} | {r.val_urgent_auc:.3f} | {r.val_calibration_error:.3f} | "
                     f"{r.tau_fitted_on} ({r.tau_urgent_cases}) | {r.bias_result} ({r.bias_failed_slices}) | "
                     f"{'yes' if r.eligible else 'no'} | "
                     f"{r.test_urgent_auc:.3f} | {r.test_urgent_recall:.3f} |")
    lines += ["", f"Selected: **{decision['selected']}** — {decision['reason']}.", ""]
    return "\n".join(lines)


def run_selection(params: Dict[str, Any], device: torch.device, out_dir: Path = OUT_DIR,
                  eval_dir: Path = EVAL_DIR, rebuild_baselines: bool = True,
                  measure_speed: bool = True) -> Dict[str, Any]:
    cfg = params["selection"]
    out_dir.mkdir(parents=True, exist_ok=True)
    backbones = select_backbone(params)
    speed = (size_and_speed(params, list(backbones["backbone"])) if measure_speed
             else pd.DataFrame({"backbone": backbones["backbone"], "parameters_m": np.nan,
                                "checkpoint_mb": np.nan, "cpu_ms_per_photo": np.nan}))
    image_checkpoint = MODELS_DIR / backbones.iloc[0]["backbone"] / "best.pt"
    baselines = build_baselines(params, image_checkpoint, device, eval_dir) if rebuild_baselines else \
        [n for n in ("baseline_prevalence", "baseline_answers_lr", "baseline_image_only") if (eval_dir / n).exists()]
    missing = [n for n in cfg["candidates"] if not (eval_dir / n / "metrics.json").exists()]
    if missing:    # e.g. a CI runner only has the models it just built
        print(f"skipping candidates without an evaluation folder: {', '.join(missing)}")
    names = [n for n in cfg["candidates"] if n not in missing] + baselines
    selection_id = time.strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name="select-model") as parent:
        mlflow.set_tags({**data_version(params), "selection_id": selection_id})
        mlflow.log_params({"selected_backbone": backbones.iloc[0]["backbone"],
                           "skipped_candidates": ",".join(missing) or "none",
                           "min_tau_urgent_cases": cfg["min_tau_urgent_cases"],
                           "max_auc_drop_for_bias": cfg["max_auc_drop_for_bias"]})
        for name in names:
            s = candidate_summary(params, name, eval_dir)
            with mlflow.start_run(run_name=f"candidate-{name}", nested=True):
                mlflow.set_tags(data_version(params))
                mlflow.set_tags({"selection_id": selection_id, "selection_stage": "phone", "candidate": name,
                                 "kind": s["kind"], "bias_result": s["bias_result"],
                                 "tau_fitted_on": s["tau_fitted_on"], "checkpoint": s["checkpoint"],
                                 "eval_dir": s["eval_dir"]})
                mlflow.log_metrics({k: v for k, v in s.items() if isinstance(v, (int, float))
                                    and not isinstance(v, bool) and not math.isnan(v)})
        # the course pattern: rank the candidate runs with search_runs
        runs = mlflow.search_runs(experiment_ids=[parent.info.experiment_id],
                                  filter_string=f"tags.selection_id = '{selection_id}' "
                                                "and tags.selection_stage = 'phone'",
                                  order_by=["metrics.val_urgent_auc DESC"])
        table = pd.DataFrame({
            "name": runs["tags.candidate"], "kind": runs["tags.kind"], "bias_result": runs["tags.bias_result"],
            "tau_fitted_on": runs["tags.tau_fitted_on"],
            "tau_urgent_cases": runs["metrics.tau_urgent_cases"].astype(int),
            "val_urgent_auc": runs["metrics.val_urgent_auc"],
            "val_calibration_error": runs["metrics.val_calibration_error"],
            "bias_failed_slices": runs["metrics.bias_failed_slices"].astype(int),
            "test_urgent_auc": runs["metrics.test_urgent_auc"],
            "test_urgent_recall": runs["metrics.test_urgent_recall"],
            "checkpoint": runs["tags.checkpoint"], "eval_dir": runs["tags.eval_dir"],
            "run_id": runs["run_id"]}).reset_index(drop=True)
        table["servable"] = table["kind"] == "fusion"       # baselines are comparisons, not registry models
        decision = choose(table, params)

        save_charts(backbones, speed, decision["table"], decision["selected"], out_dir)
        (out_dir / "model_selection.md").write_text(to_markdown(backbones, speed, decision["table"], decision, params))
        chosen = decision["table"].set_index("name").loc[decision["selected"]] if decision["selected"] else None
        result = {"selection_id": selection_id, "backbone": backbones.iloc[0]["backbone"],
                  "skipped_candidates": missing,
                  "backbones": backbones.merge(speed, on="backbone").to_dict(orient="records"),
                  "selected": decision["selected"], "reason": decision["reason"],
                  "checkpoint": chosen["checkpoint"] if chosen is not None else None,
                  "eval_dir": chosen["eval_dir"] if chosen is not None else None,
                  "bias_result": chosen["bias_result"] if chosen is not None else None,
                  "candidates": json.loads(decision["table"].to_json(orient="records"))}
        (out_dir / "model_selection.json").write_text(json.dumps(result, indent=2))
        mlflow.set_tag("selected", str(decision["selected"]))
        mlflow.log_artifacts(str(out_dir))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the model after validation and bias checks")
    parser.add_argument("--no-baselines", action="store_true", help="reuse saved baselines instead of rebuilding")
    parser.add_argument("--no-speed", action="store_true", help="skip the CPU speed measurement")
    args = parser.parse_args()
    params = load_params()
    set_seed(params["seed"])
    setup_mlflow(params)
    r = run_selection(params, pick_device(), rebuild_baselines=not args.no_baselines,
                      measure_speed=not args.no_speed)
    print(f"backbone: {r['backbone']}")
    for c in r["candidates"]:
        flag = "SELECTED" if c["name"] == r["selected"] else ("eligible" if c["eligible"] else "filtered")
        print(f"  {c['name']:22s} val AUC {c['val_urgent_auc']:.3f}  cal {c['val_calibration_error']:.3f}  "
              f"tau from {c['tau_urgent_cases']:3d} urgent  bias {c['bias_result']:12s} {flag}")
    print(f"selected: {r['selected']} — {r['reason']}")


if __name__ == "__main__":
    main()
