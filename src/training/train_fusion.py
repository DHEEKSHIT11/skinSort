"""Train the fusion head on top of a trained image model.

    python -m src.training.train_fusion --smoke        # quick code check, tiny data
    python -m src.training.train_fusion                # uses the best backbone in outputs/models
    python -m src.training.train_fusion --backbone efficientnet_b3
    python -m src.training.train_fusion --image-checkpoint outputs/models/efficientnet_b3_pad/best.pt

Needs outputs/models/<backbone>/best.pt from src.training.train. The backbone is
frozen, so its image features are computed once and only the small head trains.
Trains on pad_train, picks the best epoch on pad_val, and also logs the image-only
scores on pad_val so MLflow shows whether the answers help.
Saves outputs/models/fusion/best.pt.
"""
import argparse
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion_model import build_fusion_head
from src.models.image_model import ImageClassifier, build_image_model, count_parameters
from src.training.dataset import ROOT, answer_feature_columns, load_params, load_table, make_loader, data_version
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


def fit_head(head: nn.Module, train_data: Tuple[torch.Tensor, ...], val_data: Tuple[torch.Tensor, ...],
             device: torch.device, urgent_idx: int, lr: float, weight_decay: float,
             class_weights: List[float], batch_size: int, epochs: int, patience: int,
             on_epoch: Optional[Callable[[int, Dict[str, float], bool], None]] = None,
             seed: Optional[int] = None,
             select_best: bool = True,
             sample_weights: Optional[torch.Tensor] = None) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """Train a fusion head on precomputed (features, answers, labels) tensors.

    Picks the best epoch on val_data with selection_score and early-stops after
    `patience` epochs without gain. With select_best=False it trains exactly `epochs`
    epochs and returns the last one (val_data is then only reported, never used to
    choose), which is what fit_final.py needs for honest out-of-fold scores.
    sample_weights (one per training row, e.g. Kamiran & Calders reweighing) multiply each
    row's loss on top of the class weights. on_epoch(epoch, metrics, is_best) is called after
    every epoch (logging / saving). Returns the best epoch's metrics and weights.
    """
    head = head.to(device)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    weights = sample_weights if sample_weights is not None else torch.ones(len(train_data[2]))
    loader = DataLoader(TensorDataset(*train_data, weights.float()), batch_size=batch_size, shuffle=True,
                        generator=generator)
    class_w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    per_row = nn.CrossEntropyLoss(weight=class_w, reduction="none")
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    val_feats, val_answers, val_labels = (t.to(device) for t in val_data)

    best_metrics: Dict[str, float] = {}
    best_state: Dict[str, torch.Tensor] = {}
    best_score, epochs_without_gain = -math.inf, 0
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss, seen = 0.0, 0
        for feats, answers, labels, w in loader:
            feats, answers, labels, w = feats.to(device), answers.to(device), labels.to(device), w.to(device)
            optimizer.zero_grad()
            # weighted mean; with all weights 1 this equals CrossEntropyLoss(weight=class_w)
            loss = (per_row(head(feats, answers), labels) * w).sum() / (class_w[labels] * w).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)

        head.eval()
        with torch.no_grad():
            probs = torch.softmax(head(val_feats, val_answers), dim=1).cpu().numpy()
        metrics = val_metrics(probs, val_labels.cpu().numpy(), urgent_idx)
        metrics["train_loss"] = total_loss / seen
        metrics["epoch"] = epoch

        score = selection_score(metrics) if select_best else float(epoch)
        is_best = score > best_score
        if is_best:
            best_score, epochs_without_gain = score, 0
            best_metrics = dict(metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            epochs_without_gain += 1
        if on_epoch:
            on_epoch(epoch, metrics, is_best)
        if select_best and not is_best and epochs_without_gain >= patience:
            break
    return best_metrics, best_state


def train_fusion(params: Dict[str, Any], image_checkpoint: Path, train_df: pd.DataFrame,
                 val_df: pd.DataFrame, device: torch.device, epochs: Optional[int] = None,
                 out_dir: Optional[Path] = None) -> Dict[str, Any]:
    cfg = params["fusion"]
    epochs = epochs or cfg["epochs"]
    urgent_idx = params["data"]["classes"].index("urgent")
    out_dir = Path(out_dir or MODELS_DIR / "fusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    weight_decay = cfg.get("weight_decay", params["train"]["weight_decay"])
    class_weights = cfg.get("class_weights", params["train"]["class_weights"])

    image_model = load_image_model(params, image_checkpoint, device)
    train_data = extract_features(image_model, train_df, params, device)
    val_data = extract_features(image_model, val_df, params, device)
    val_feats, _, val_labels = val_data

    # the image model's own head on the same pad_val photos, for comparison
    with torch.no_grad():
        image_only_probs = torch.softmax(image_model.head(val_feats.to(device)), dim=1).cpu().numpy()
    image_only = val_metrics(image_only_probs, val_labels.numpy(), urgent_idx)

    head = build_fusion_head(params, image_model.num_features, train_data[1].shape[1])

    def save(state: Dict[str, torch.Tensor], metrics: Dict[str, float]) -> None:
        torch.save({"state_dict": state, "backbone": image_model.backbone_name,
                    "image_checkpoint": str(image_checkpoint),
                    "num_image_features": image_model.num_features,
                    "answer_columns": answer_feature_columns(params),
                    "hidden_dim": cfg["hidden_dim"], "dropout": cfg["dropout"],
                    "classes": params["data"]["classes"], "epoch": int(metrics["epoch"]),
                    "val_metrics": metrics}, best_path)

    def on_epoch(epoch: int, metrics: Dict[str, float], is_best: bool) -> None:
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if k != "epoch" and not math.isnan(v)}, step=epoch)
        print(f"[fusion] epoch {epoch}/{epochs}  loss {metrics['train_loss']:.3f}  "
              f"pad_val urgent AUC {metrics['urgent_auc']:.3f}  urgent recall "
              f"{metrics['urgent_recall']:.3f}  macro F1 {metrics['macro_f1']:.3f}")
        if is_best:
            save({k: v.detach().cpu() for k, v in head.state_dict().items()}, metrics)

    # nested when called from inside another run (e.g. the final retrain in tune.py)
    with mlflow.start_run(run_name=f"fusion-{image_model.backbone_name}",
                          nested=mlflow.active_run() is not None):
        mlflow.set_tags(data_version(params))
        mlflow.log_params({
            "backbone": image_model.backbone_name, "image_checkpoint": str(image_checkpoint),
            "epochs": epochs, "lr": cfg["lr"], "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"], "weight_decay": weight_decay,
            "class_weights": str(class_weights), "answer_inputs": train_data[1].shape[1],
            "train_images": len(train_df), "val_images": len(val_df),
            "parameters": count_parameters(head), "device": str(device),
        })
        mlflow.log_metrics({f"image_only_{k}": v for k, v in image_only.items() if not math.isnan(v)})
        print(f"[image only] pad_val urgent AUC {image_only['urgent_auc']:.3f}  urgent recall "
              f"{image_only['urgent_recall']:.3f}  macro F1 {image_only['macro_f1']:.3f}")

        best_metrics, _ = fit_head(head, train_data, val_data, device, urgent_idx, lr=cfg["lr"],
                                   weight_decay=weight_decay, class_weights=class_weights,
                                   batch_size=params["train"]["batch_size"], epochs=epochs,
                                   patience=params["train"]["early_stopping_patience"],
                                   on_epoch=on_epoch)
        best = {"backbone": image_model.backbone_name, **best_metrics,
                "epoch": int(best_metrics["epoch"])}
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()
                            if k != "epoch" and not math.isnan(v)})
        mlflow.log_param("best_epoch", best["epoch"])
        mlflow.log_artifact(str(best_path))
    best["checkpoint"] = str(best_path)
    best["image_only"] = image_only
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SkinSort fusion head")
    parser.add_argument("--backbone", help="image model to build on (default: best in outputs/models)")
    parser.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, to check the code")
    parser.add_argument("--image-checkpoint", type=Path, help="exact image model checkpoint to build on")
    parser.add_argument("--epochs", type=int, help="override fusion.epochs")
    args = parser.parse_args()

    params = load_params()
    set_seed(params["seed"])
    device = pick_device()
    splits = params["data"]["splits"]
    limit = params["train"]["smoke_test_rows"] if args.smoke else None
    epochs = 1 if args.smoke else args.epochs

    if args.image_checkpoint:
        checkpoint = args.image_checkpoint.resolve()
    elif args.backbone:
        checkpoint = MODELS_DIR / args.backbone / "best.pt"
    else:
        checkpoint = find_best_image_checkpoint()
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
