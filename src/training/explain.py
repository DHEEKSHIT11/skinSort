"""Feature sensitivity of the fusion model with SHAP: which inputs move P(urgent)?

Uses SHAP's GradientExplainer (expected gradients) on the fusion head, with the
frozen image features and the 12 answer inputs (6 answers + 6 "missing" flags) as
inputs. The image's 1,000+ features are summed into one "image" value so it can be
compared with each answer.

    python -m src.training.explain --checkpoint outputs/models/fusion_tuned/best.pt

Background = 200 pad_train photos, explained = all pad_val photos. Writes
outputs/explain/<checkpoint folder>/: shap_importance.png, shap_answers.png,
shap_summary.json, and logs them to MLflow as run explain-<folder>.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import mlflow
import numpy as np
import shap
import torch
from torch import nn

from src.models.fusion_model import fusion_head_from_checkpoint
from src.training.dataset import ROOT, data_version, load_params, load_table
from src.training.train import pick_device, set_seed, setup_mlflow
from src.training.tune import cached_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "outputs" / "explain"


class UrgentProbability(nn.Module):
    """Fusion head -> P(urgent), the score the triage queue and tau use."""

    def __init__(self, head: nn.Module, urgent_idx: int):
        super().__init__()
        self.head, self.urgent_idx = head, urgent_idx

    def forward(self, feats: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(feats, answers), dim=1)[:, self.urgent_idx:self.urgent_idx + 1]


def shap_values(model: nn.Module, background: List[torch.Tensor],
                explain: List[torch.Tensor], seed: int = 0) -> List[np.ndarray]:
    """SHAP values per input tensor, shape (n, n_features) each."""
    torch.manual_seed(seed)
    explainer = shap.GradientExplainer(model, background)
    values = explainer.shap_values(explain, nsamples=200, rseed=seed)
    return [np.asarray(v).reshape(v.shape[0], -1) for v in values]


def summarise(image_values: np.ndarray, answer_values: np.ndarray,
              answer_names: List[str]) -> Dict[str, float]:
    """Mean |SHAP| per input; image features are summed per photo first."""
    summary = {"image (all features)": float(np.abs(image_values.sum(axis=1)).mean())}
    for i, name in enumerate(answer_names):
        summary[name] = float(np.abs(answer_values[:, i]).mean())
    return dict(sorted(summary.items(), key=lambda kv: -kv[1]))


def explain(params: Dict[str, Any], checkpoint: Path, device: torch.device,
            out_dir: Path) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    splits = params["data"]["splits"]
    image_ckpt = Path(ckpt["image_checkpoint"])
    train_df = load_table(params, [splits["pad_train"]])
    val_df = load_table(params, [splits["pad_val"]])
    train_feats, train_answers, _ = cached_features(params, image_ckpt, train_df, "pad_train", device)
    val_feats, val_answers, _ = cached_features(params, image_ckpt, val_df, "pad_val", device)

    head = fusion_head_from_checkpoint(params, ckpt)
    model = UrgentProbability(head.eval(), ckpt["classes"].index("urgent"))

    rng = np.random.default_rng(params["seed"])
    idx = torch.as_tensor(rng.choice(len(train_feats), size=min(200, len(train_feats)), replace=False))
    image_values, answer_values = shap_values(model, [train_feats[idx], train_answers[idx]],
                                              [val_feats, val_answers], seed=params["seed"])
    names = list(ckpt["answer_columns"])
    importance = summarise(image_values, answer_values, names)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    labels, values = list(importance)[::-1], list(importance.values())[::-1]
    ax.barh(labels, values, color=["#d9822b" if lbl.startswith("image") else "#3b6fb6" for lbl in labels])
    ax.set_xlabel("mean |SHAP| on P(urgent), pad_val")
    ax.set_title("What moves the urgent score", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_importance.png", dpi=120)
    plt.close(fig)

    # direction: for each yes/no answer, average SHAP when the answer is yes vs no
    answers_np = val_answers.numpy()
    direction = {}
    for i, name in enumerate(names):
        if name.endswith("_missing"):
            continue
        yes, no = answers_np[:, i] > 0.5, answers_np[:, i] <= 0.5
        miss = answers_np[:, names.index(f"{name}_missing")] > 0.5 if f"{name}_missing" in names else ~yes & ~no
        direction[name] = {
            "yes": float(answer_values[yes & ~miss, i].mean()) if (yes & ~miss).any() else float("nan"),
            "no": float(answer_values[no & ~miss, i].mean()) if (no & ~miss).any() else float("nan"),
            "answered": int((~miss).sum()),
        }
    plt.figure()
    shap.summary_plot(answer_values[:, :6], answers_np[:, :6], feature_names=names[:6], show=False,
                      plot_size=(6, 3.6))
    plt.title("Answers: red = yes, blue = no", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_answers.png", dpi=120)
    plt.close("all")

    summary = {"checkpoint": str(checkpoint), "explained": "pad_val", "photos": len(val_df),
               "mean_abs_shap": importance, "answer_direction": direction}
    (out_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2))
    with mlflow.start_run(run_name=f"explain-{checkpoint.parent.name}"):
        mlflow.set_tags(data_version(params))
        mlflow.log_param("checkpoint", str(checkpoint))
        mlflow.log_metrics({f"shap_{k.split(' ')[0]}": v for k, v in importance.items()})
        mlflow.log_artifacts(str(out_dir))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP sensitivity of the fusion model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fusion checkpoint")
    args = parser.parse_args()
    params = load_params()
    set_seed(params["seed"])
    setup_mlflow(params)
    checkpoint = args.checkpoint.resolve()
    summary = explain(params, checkpoint, pick_device(), OUT_DIR / checkpoint.parent.name)
    print("mean |SHAP| on P(urgent):")
    for k, v in summary["mean_abs_shap"].items():
        print(f"  {k:22s} {v:.4f}")
    print("average SHAP when the answer is yes / no:")
    for k, d in summary["answer_direction"].items():
        print(f"  {k:12s} yes {d['yes']:+.4f}  no {d['no']:+.4f}  ({d['answered']} answered)")


if __name__ == "__main__":
    main()
