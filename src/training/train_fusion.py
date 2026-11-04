"""Train the fusion head on top of a trained image model.

    python -m src.training.train_fusion --smoke        # quick code check, tiny data
    python -m src.training.train_fusion                # uses the best backbone in outputs/models
    python -m src.training.train_fusion --backbone efficientnet_b3

Needs outputs/models/<backbone>/best.pt from src.training.train. The backbone is
frozen, so its image features are computed once and only the small head trains.
Trains on pad_train, picks the best epoch on pad_val, and also logs the image-only
scores on pad_val so MLflow shows whether the answers help.
Saves outputs/models/fusion/best.pt.
"""
import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion_model import build_fusion_head
from src.models.image_model import ImageClassifier, build_image_model, count_parameters
from src.training.dataset import ROOT, answer_feature_columns, load_params, load_table, make_loader
from src.training.train import pick_device, selection_score, set_seed, setup_mlflow, val_metrics

MODELS_DIR = ROOT / "outputs" / "models"


def find_best_image_checkpoint(models_dir: Path = MODELS_DIR) -> Path:
    """The image checkpoint with the highest saved val selection score."""
    best_path, best_score = None, -math.inf
    for path in sorted(models_dir.glob("*/best.pt")):
        if path.parent.name == "fusion":
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        score = selection_score(ckpt["val_metrics"])
        if score > best_score:
            best_path, best_score = path, score
    if best_path is None:
        raise FileNotFoundError(f"No image checkpoints in {models_dir}; run src.training.train first")
    return best_path


def load_image_model(params: Dict[str, Any], checkpoint: Path,
                     device: torch.device) -> ImageClassifier:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_image_model(params, ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def extract_features(model: ImageClassifier, df: pd.DataFrame, params: Dict[str, Any],
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen image features, answers and labels for every row (no augmentation)."""
    loader = make_loader(df, params, train=False, with_answers=True)
    features, answers, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            features.append(model.features(batch["image"].to(device)).cpu())
            answers.append(batch["answers"])
            labels.append(batch["label"])
    return torch.cat(features), torch.cat(answers), torch.cat(labels)


def train_fusion(params: Dict[str, Any], image_checkpoint: Path, train_df: pd.DataFrame,
                 val_df: pd.DataFrame, device: torch.device, epochs: Optional[int] = None,
                 out_dir: Optional[Path] = None) -> Dict[str, Any]:
    cfg = params["fusion"]
    epochs = epochs or cfg["epochs"]
    urgent_idx = params["data"]["classes"].index("urgent")
    out_dir = Path(out_dir or MODELS_DIR / "fusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    image_model = load_image_model(params, image_checkpoint, device)
    train_feats, train_answers, train_labels = extract_features(image_model, train_df, params, device)
    val_feats, val_answers, val_labels = extract_features(image_model, val_df, params, device)

    # the image model's own head on the same pad_val photos, for comparison
    with torch.no_grad():
        image_only_probs = torch.softmax(image_model.head(val_feats.to(device)), dim=1).cpu().numpy()
    image_only = val_metrics(image_only_probs, val_labels.numpy(), urgent_idx)

    head = build_fusion_head(params, image_model.num_features, train_answers.shape[1]).to(device)
    loader = DataLoader(TensorDataset(train_feats, train_answers, train_labels),
                        batch_size=params["train"]["batch_size"], shuffle=True)
    class_weights = torch.tensor(params["train"]["class_weights"], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"],
                                  weight_decay=params["train"]["weight_decay"])

    best: Dict[str, Any] = {}
    best_score, epochs_without_gain = -math.inf, 0

    with mlflow.start_run(run_name=f"fusion-{image_model.backbone_name}"):
        mlflow.log_params({
            "backbone": image_model.backbone_name, "image_checkpoint": str(image_checkpoint),
            "epochs": epochs, "lr": cfg["lr"], "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"], "answer_inputs": train_answers.shape[1],
            "train_images": len(train_df), "val_images": len(val_df),
            "parameters": count_parameters(head), "device": str(device),
        })
        mlflow.log_metrics({f"image_only_{k}": v for k, v in image_only.items() if not math.isnan(v)})
        print(f"[image only] pad_val urgent AUC {image_only['urgent_auc']:.3f}  urgent recall "
              f"{image_only['urgent_recall']:.3f}  macro F1 {image_only['macro_f1']:.3f}")

        for epoch in range(1, epochs + 1):
            head.train()
            total_loss, seen = 0.0, 0
            for feats, answers, labels in loader:
                feats, answers, labels = feats.to(device), answers.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(head(feats, answers), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                seen += len(labels)

            head.eval()
            with torch.no_grad():
                logits = head(val_feats.to(device), val_answers.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            metrics = val_metrics(probs, val_labels.numpy(), urgent_idx)
            metrics["train_loss"] = total_loss / seen
            mlflow.log_metrics({k: v for k, v in metrics.items() if not math.isnan(v)}, step=epoch)
            print(f"[fusion] epoch {epoch}/{epochs}  loss {metrics['train_loss']:.3f}  "
                  f"pad_val urgent AUC {metrics['urgent_auc']:.3f}  urgent recall "
                  f"{metrics['urgent_recall']:.3f}  macro F1 {metrics['macro_f1']:.3f}")

            score = selection_score(metrics)
            if score > best_score:
                best_score, epochs_without_gain = score, 0
                best = {"backbone": image_model.backbone_name, "epoch": epoch, **metrics}
                torch.save({"state_dict": head.state_dict(), "backbone": image_model.backbone_name,
                            "image_checkpoint": str(image_checkpoint),
                            "num_image_features": image_model.num_features,
                            "answer_columns": answer_feature_columns(params),
                            "classes": params["data"]["classes"], "epoch": epoch,
                            "val_metrics": metrics}, best_path)
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= params["train"]["early_stopping_patience"]:
                    print(f"[fusion] no improvement for {epochs_without_gain} epochs, stopping")
                    break

        mlflow.log_metrics({f"best_{k}": v for k, v in best.items()
                            if isinstance(v, float) and not math.isnan(v)})
        mlflow.log_param("best_epoch", best["epoch"])
        mlflow.log_artifact(str(best_path))
    best["checkpoint"] = str(best_path)
    best["image_only"] = image_only
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SkinSort fusion head")
    parser.add_argument("--backbone", help="image model to build on (default: best in outputs/models)")
    parser.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, to check the code")
    parser.add_argument("--epochs", type=int, help="override fusion.epochs")
    args = parser.parse_args()

    params = load_params()
    set_seed(params["seed"])
    device = pick_device()
    splits = params["data"]["splits"]
    limit = params["train"]["smoke_test_rows"] if args.smoke else None
    epochs = 1 if args.smoke else args.epochs

    checkpoint = (MODELS_DIR / args.backbone / "best.pt" if args.backbone
                  else find_best_image_checkpoint())
    train_df = load_table(params, [splits["pad_train"]], limit=limit)
    val_df = load_table(params, [splits["pad_val"]], limit=limit)
    print(f"device {device}, image model {checkpoint}, "
          f"{len(train_df)} train photos, {len(val_df)} val photos")

    setup_mlflow(params, "-smoke" if args.smoke else "")
    best = train_fusion(params, checkpoint, train_df, val_df, device, epochs=epochs)

    gain = best["urgent_auc"] - best["image_only"]["urgent_auc"]
    print(f"\nBest fusion epoch {best['epoch']}: pad_val urgent AUC {best['urgent_auc']:.3f} "
          f"({gain:+.3f} vs image only), urgent recall {best['urgent_recall']:.3f}, "
          f"macro F1 {best['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
