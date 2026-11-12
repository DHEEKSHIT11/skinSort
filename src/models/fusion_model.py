"""Fusion model: trained image features + the patient's answers -> urgency.

The image backbone comes from a trained ImageClassifier and stays frozen; only
the small fusion head is trained (on the phone-photo splits, which have answers).

Quick check (prints the head size for each backbone in params.yaml):
    python -m src.models.fusion_model
"""
from typing import Any, Dict

import torch
from torch import nn

from src.models.image_model import ImageClassifier


class FusionHead(nn.Module):
    """Answers go through a small layer first, then join the image features."""

    def __init__(self, num_image_features: int, num_answers: int, num_classes: int = 3,
                 hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.answer_branch = nn.Sequential(nn.Linear(num_answers, hidden_dim), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_image_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_features: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([image_features, self.answer_branch(answers)], dim=1))


class FusionModel(nn.Module):
    """Full model for serving: photo + answers in, class scores out."""

    def __init__(self, image_model: ImageClassifier, head: FusionHead):
        super().__init__()
        self.backbone = image_model.backbone
        self.backbone_name = image_model.backbone_name
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features, answers)


def build_fusion_head(params: Dict[str, Any], num_image_features: int,
                      num_answers: int) -> FusionHead:
    cfg = params["fusion"]
    return FusionHead(num_image_features, num_answers,
                      num_classes=len(params["data"]["classes"]),
                      hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"])


def fusion_head_from_checkpoint(params: Dict[str, Any], ckpt: Dict[str, Any]) -> FusionHead:
    """Rebuild a saved head with the size it was trained with (tuning may change hidden_dim)."""
    hidden = ckpt.get("hidden_dim", ckpt["state_dict"]["answer_branch.0.weight"].shape[0])
    p = {**params, "fusion": {**params["fusion"], "hidden_dim": int(hidden)}}
    head = build_fusion_head(p, ckpt["num_image_features"], len(ckpt["answer_columns"]))
    head.load_state_dict(ckpt["state_dict"])
    return head


def main() -> None:
    from src.models.image_model import build_image_model, count_parameters
    from src.training.dataset import answer_feature_columns, load_params
    params = load_params()
    num_answers = len(answer_feature_columns(params))
    for name in params["train"]["backbones"]:
        image_model = build_image_model(params, name, pretrained=False)
        head = build_fusion_head(params, image_model.num_features, num_answers)
        print(f"{name:18s} fusion head {count_parameters(head) / 1e3:6.1f}K parameters "
              f"({image_model.num_features} image features + {num_answers} answer inputs)")


if __name__ == "__main__":
    main()
