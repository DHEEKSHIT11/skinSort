"""Extra model analysis (course Session 12): robustness and threshold trade-off.

robustness  Phone photos vary in quality. Each pad_test photo is degraded (blur, dark,
            bright, heavy JPEG, low resolution), then goes through the serving preprocessing
            (centre crop + Shades-of-Gray colour constancy) before the model sees it, and urgent AUC,
            recall at the fixed tau and flagged share are compared with the clean photos.
thresholds  Urgent recall and flagged share for every tau, out-of-fold (pooled
            pad_train + pad_val) and on pad_test, with the chosen tau marked. Shows the
            trade-off a clinic would make by moving the threshold.

    python -m src.training.analysis --checkpoint outputs/models/fusion_oof/best.pt

Writes outputs/analysis/: robustness.json/.png, thresholds.png.
"""
import argparse
import io
import json
from pathlib import Path
from typing import Callable, Dict

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

from src.data.preprocess import preprocess_photo
from src.models.fusion_model import fusion_head_from_checkpoint
from src.training.dataset import ROOT, build_transforms, data_version, load_params, load_table, resolve
from src.training.evaluate import softmax
from src.training.train import pick_device, set_seed, setup_mlflow
from src.training.train_fusion import load_image_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "outputs" / "analysis"


def jpeg(img: Image.Image, quality: int = 15) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def low_res(img: Image.Image, factor: int = 4) -> Image.Image:
    w, h = img.size
    return img.resize((w // factor, h // factor)).resize((w, h))


CORRUPTIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda img: img,
    "blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=3)),
    "dark": lambda img: ImageEnhance.Brightness(img).enhance(0.55),
    "bright": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
    "jpeg_q15": jpeg,
    "low_res_x4": low_res,
}


def urgent_scores(params, ckpt: Dict, image_model, head, df: pd.DataFrame, corrupt, device,
                  batch: int = 16) -> np.ndarray:
    """Calibrated P(urgent) for every row after applying `corrupt` to the photo."""
    from src.serving.triage_model import TriageModel   # same answer encoding as serving
    encoder = TriageModel()
    encoder.answers = params["data"]["answer_columns"]
    encoder.answer_columns = ckpt["answer_columns"]
    transform = build_transforms(params, train=False)
    images_dir = resolve(params["data"]["images_dir"])
    temperature, offset = float(ckpt["temperature"]), np.asarray(ckpt["class_offset"])
    scores = []
    for start in range(0, len(df), batch):
        part = df.iloc[start:start + batch]
        tensors = []
        for image_id in part["image_id"]:
            with Image.open(images_dir / f"{image_id}.jpg") as img:
                # degrade, then the serving preprocessing (crop + colour constancy), as in production
                tensors.append(transform(preprocess_photo(corrupt(img.convert("RGB")), params["preprocess"])))
        with torch.no_grad():
            feats = image_model.features(torch.stack(tensors).to(device)).cpu()
            logits = head(feats, encoder.answer_features(part.reset_index(drop=True))).numpy()
        scores.append(softmax(logits, (temperature, offset))[:, ckpt["classes"].index("urgent")])
    return np.concatenate(scores)


def robustness(params, checkpoint: Path, device, out_dir: Path) -> Dict[str, Dict[str, float]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    image_model = load_image_model(params, Path(ckpt["image_checkpoint"]), device)
    head = fusion_head_from_checkpoint(params, ckpt).eval()
    df = load_table(params, [params["data"]["splits"]["pad_test"]])
    is_urgent = (df["urgency"] == "urgent").to_numpy()
    tau = float(ckpt["tau"])
    results = {}
    for name, corrupt in CORRUPTIONS.items():
        p = urgent_scores(params, ckpt, image_model, head, df, corrupt, device)
        flagged = p >= tau
        results[name] = {"urgent_auc": float(roc_auc_score(is_urgent, p)),
                         "urgent_recall": float(flagged[is_urgent].mean()),
                         "flagged_share": float(flagged.mean())}
        print(f"{name:12s} AUC {results[name]['urgent_auc']:.3f}  recall {results[name]['urgent_recall']:.3f}  "
              f"flagged {results[name]['flagged_share']:.1%}")
    (out_dir / "robustness.json").write_text(json.dumps({"checkpoint": str(checkpoint), "tau": tau,
                                                         "split": "pad_test", "results": results}, indent=2))
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, key, label in zip(axes, ["urgent_auc", "urgent_recall", "flagged_share"],
                              ["urgent AUC", f"urgent recall @ tau {tau:.3f}", "flagged share"]):
        vals = [results[n][key] for n in names]
        ax.bar(names, vals, color=["#d9822b"] + ["#3b6fb6"] * (len(names) - 1))
        ax.axhline(results["clean"][key], color="#444", linestyle="--", linewidth=1)
        for x, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.suptitle("Robustness on pad_test photos (degraded before the model sees them)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness.png", dpi=130)
    plt.close(fig)
    return results


def threshold_curve(eval_dir: Path, tau: float, out_dir: Path) -> None:
    preds = pd.read_csv(eval_dir / "predictions.csv")
    taus = np.linspace(0, 0.6, 121)
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, label, style in [("val", "out-of-fold (pad_train + pad_val)", "-"), ("test", "pad_test", "--")]:
        g = preds[preds["eval_split"] == split]
        urgent = (g["label"] == 2).to_numpy()
        p = g["p_urgent"].to_numpy()
        recall = [(p[urgent] >= t).mean() for t in taus]
        flagged = [(p >= t).mean() for t in taus]
        ax.plot(taus, recall, style, color="#3b6fb6", label=f"urgent recall, {label}")
        ax.plot(taus, flagged, style, color="#d9822b", label=f"flagged share, {label}")
    ax.axvline(tau, color="#444", linewidth=1)
    ax.annotate(f"chosen tau {tau:.3f}", (tau, 0.02), xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax.axhline(0.85, color="#b8c2cc", linewidth=1, linestyle=":")
    ax.set_xlabel("threshold tau on P(urgent)")
    ax.set_ylabel("share")
    ax.set_title("Threshold trade-off: urgent cases caught vs photos flagged", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "thresholds.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness and threshold analysis")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fusion checkpoint with tau inside")
    args = parser.parse_args()
    params = load_params()
    set_seed(params["seed"])
    setup_mlflow(params)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with mlflow.start_run(run_name=f"analysis-{checkpoint.parent.name}"):
        mlflow.set_tags(data_version(params))
        results = robustness(params, checkpoint, pick_device(), out_dir)
        threshold_curve(ROOT / "outputs" / "evaluation" / checkpoint.parent.name, float(ckpt["tau"]), out_dir)
        for name, r in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in r.items()})
        mlflow.log_artifacts(str(out_dir))


if __name__ == "__main__":
    main()
