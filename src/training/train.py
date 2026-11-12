"""Train image models and keep the best epoch of each backbone.

    python -m src.training.train --smoke --no-pretrained   # quick code check, tiny data, 1 epoch
    python -m src.training.train                           # every backbone in params.yaml
    python -m src.training.train --backbone efficientnet_b3

    # fine-tune a trained model on the phone photos (selects on pad_val)
    python -m src.training.train --backbone efficientnet_b3 \\
        --init-checkpoint outputs/models/efficientnet_b3/best.pt \\
        --train-split pad_train --val-split pad_val --lr 0.00003 --batch-size 8 \\
        --freeze-bn --out-name efficientnet_b3_pad

Every run is logged to MLflow (see the mlflow section of params.yaml).
The best epoch of each backbone is saved to outputs/models/<backbone>/best.pt.
"""
import argparse
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch import nn

from src.models.image_model import build_image_model, count_parameters
from src.training.dataset import ROOT, load_params, load_table, make_loader, data_version


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")      # GCP GPU VM
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon laptop
    return torch.device("cpu")


def predict(model: nn.Module, loader, device: torch.device):
    """Class probabilities and true labels for every image in the loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs.append(torch.softmax(logits, dim=1).cpu())
            labels.append(batch["label"])
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def val_metrics(probs: np.ndarray, labels: np.ndarray, urgent_idx: int) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    is_urgent = (labels == urgent_idx).astype(int)
    metrics = {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "urgent_recall": recall_score(is_urgent, (preds == urgent_idx).astype(int), zero_division=0),
        "urgent_auc": float("nan"),
    }
    # AUC needs both urgent and non-urgent cases
    if 0 < is_urgent.sum() < len(is_urgent):
        metrics["urgent_auc"] = roc_auc_score(is_urgent, probs[:, urgent_idx])
    return {k: float(v) for k, v in metrics.items()}


def selection_score(metrics: Dict[str, float]) -> float:
    """Which epoch is best. Urgent AUC measures how well the model ranks urgent
    cases above the rest, which is what the queue and the threshold tau use.
    Raw urgent recall is not used here: a model that calls everything urgent
    would score 100%. Tau is chosen later in evaluate.py to reach the recall target."""
    auc = metrics["urgent_auc"]
    return metrics["macro_f1"] if math.isnan(auc) else auc


def freeze_batchnorm(model: nn.Module) -> None:
    """Keep every BatchNorm layer as trained: running stats in eval mode and scale/shift
    frozen. Used when fine-tuning with small batches, which otherwise corrupt the stats."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def train_one_backbone(params: Dict[str, Any], backbone: str, train_df: pd.DataFrame,
                       val_df: pd.DataFrame, device: torch.device, pretrained: bool = True,
                       epochs: Optional[int] = None, out_dir: Optional[Path] = None,
                       init_checkpoint: Optional[Path] = None,
                       run_name: Optional[str] = None, freeze_bn: bool = False) -> Dict[str, Any]:
    """Train one backbone. With init_checkpoint, start from that trained model's
    weights instead of ImageNet (used to fine-tune on the phone photos). With
    freeze_bn, BatchNorm layers keep their trained statistics (needed for small batches)."""
    cfg = params["train"]
    epochs = epochs or cfg["epochs"]
    urgent_idx = params["data"]["classes"].index("urgent")
    out_dir = Path(out_dir or ROOT / "outputs" / "models" / backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    model = build_image_model(params, backbone, pretrained=pretrained and not init_checkpoint)
    # splits whose photos the weights have learned from (fit_final.py refuses to pick a
    # threshold on them, because scores on photos the model trained on are too optimistic)
    trained_on = set(train_df["split"].dropna().unique()) if "split" in train_df else set()
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if ckpt["backbone"] != backbone:
            raise ValueError(f"{init_checkpoint} is a {ckpt['backbone']} model, not {backbone}")
        model.load_state_dict(ckpt["state_dict"])
        trained_on |= set(ckpt.get("trained_on", []))
    model = model.to(device)
    train_loader = make_loader(train_df, params, train=True)
    val_loader = make_loader(val_df, params, train=False)
    class_weights = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    if freeze_bn:
        freeze_batchnorm(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best: Dict[str, Any] = {}
    best_score, epochs_without_gain = -math.inf, 0

    with mlflow.start_run(run_name=run_name or backbone):
        mlflow.set_tags(data_version(params))
        mlflow.log_params({
            "backbone": backbone, "pretrained": pretrained, "epochs": epochs,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else "none",
            "freeze_bn": freeze_bn,
            "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "batch_size": cfg["batch_size"],
            "image_size": cfg["image_size"], "class_weights": str(cfg["class_weights"]),
            "train_images": len(train_df), "val_images": len(val_df),
            "parameters": count_parameters(model), "device": str(device),
        })

        for epoch in range(1, epochs + 1):
            model.train()
            if freeze_bn:
                freeze_batchnorm(model)      # model.train() switches BatchNorm back on
            start, total_loss, seen = time.time(), 0.0, 0
            for batch in train_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                seen += len(labels)
            scheduler.step()

            probs, labels = predict(model, val_loader, device)
            metrics = val_metrics(probs, labels, urgent_idx)
            metrics["train_loss"] = total_loss / seen
            metrics["epoch_seconds"] = time.time() - start
            mlflow.log_metrics({k: v for k, v in metrics.items() if not math.isnan(v)}, step=epoch)
            print(f"[{backbone}] epoch {epoch}/{epochs}  loss {metrics['train_loss']:.3f}  "
                  f"val urgent AUC {metrics['urgent_auc']:.3f}  urgent recall "
                  f"{metrics['urgent_recall']:.3f}  macro F1 {metrics['macro_f1']:.3f}")

            score = selection_score(metrics)
            if score > best_score:
                best_score, epochs_without_gain = score, 0
                best = {"backbone": backbone, "epoch": epoch, **metrics}
                torch.save({"state_dict": model.state_dict(), "backbone": backbone,
                            "classes": params["data"]["classes"], "epoch": epoch,
                            "val_metrics": metrics, "trained_on": sorted(trained_on)}, best_path)
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= cfg["early_stopping_patience"]:
                    print(f"[{backbone}] no improvement for {epochs_without_gain} epochs, stopping")
                    break

        mlflow.log_metrics({f"best_{k}": v for k, v in best.items()
                            if isinstance(v, float) and not math.isnan(v)})
        mlflow.log_param("best_epoch", best["epoch"])
        mlflow.log_artifact(str(best_path))
    best["checkpoint"] = str(best_path)
    return best


def setup_mlflow(params: Dict[str, Any], experiment_suffix: str = "") -> None:
    uri = params["mlflow"]["tracking_uri"]
    if "://" not in uri:                       # a plain folder name like "mlruns"
        uri = (ROOT / uri).as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"] + experiment_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SkinSort image models")
    parser.add_argument("--backbone", help="train only this backbone (default: all in params.yaml)")
    parser.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, to check the code")
    parser.add_argument("--no-pretrained", action="store_true", help="skip downloading ImageNet weights")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="start from this trained model instead of ImageNet weights")
    parser.add_argument("--train-split", default="train", help="key under data.splits to train on")
    parser.add_argument("--val-split", default="val", help="key under data.splits to pick the best epoch on")
    parser.add_argument("--lr", type=float, help="override train.lr")
    parser.add_argument("--batch-size", type=int, help="override train.batch_size")
    parser.add_argument("--out-name", help="folder under outputs/models (default: the backbone name)")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="keep BatchNorm statistics fixed (use with small fine-tuning batches)")
    args = parser.parse_args()
    if args.out_name and not args.backbone:
        parser.error("--out-name needs --backbone")
    if args.init_checkpoint and not args.backbone:
        parser.error("--init-checkpoint needs --backbone")

    params = load_params()
    if args.lr:
        params["train"]["lr"] = args.lr
    if args.batch_size:
        params["train"]["batch_size"] = args.batch_size
    set_seed(params["seed"])
    device = pick_device()
    splits = params["data"]["splits"]
    limit = params["train"]["smoke_test_rows"] if args.smoke else None
    epochs = 1 if args.smoke else args.epochs

    train_df = load_table(params, [splits[args.train_split]], limit=limit)
    val_df = load_table(params, [splits[args.val_split]], limit=limit)
    print(f"device {device}, {len(train_df)} train images, {len(val_df)} val images")

    setup_mlflow(params, "-smoke" if args.smoke else "")
    backbones = [args.backbone] if args.backbone else params["train"]["backbones"]
    out_dir = ROOT / "outputs" / "models" / args.out_name if args.out_name else None
    results = [train_one_backbone(params, name, train_df, val_df, device,
                                  pretrained=not args.no_pretrained, epochs=epochs,
                                  out_dir=out_dir, init_checkpoint=args.init_checkpoint,
                                  run_name=args.out_name, freeze_bn=args.freeze_bn)
               for name in backbones]

    print("\nBest epoch per backbone (sorted by val urgent AUC):")
    for r in sorted(results, key=selection_score, reverse=True):
        print(f"  {r['backbone']:18s} epoch {r['epoch']:2d}  urgent AUC {r['urgent_auc']:.3f}  "
              f"urgent recall {r['urgent_recall']:.3f}  macro F1 {r['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
