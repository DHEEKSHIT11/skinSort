"""Image model: a pretrained timm backbone with a small classification head.

The backbone and head are kept separate so the fusion model can reuse the
trained backbone's image features later.

Quick check (prints the size of each backbone in params.yaml):
    python -m src.models.image_model
"""
from typing import Any, Dict

import timm
import torch
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 3,
                 pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = backbone_name
        # num_classes=0 makes timm return pooled image features instead of class scores
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.num_features, num_classes))

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


def build_image_model(params: Dict[str, Any], backbone_name: str,
                      pretrained: bool = True) -> ImageClassifier:
    return ImageClassifier(backbone_name,
                           num_classes=len(params["data"]["classes"]),
                           pretrained=pretrained,
                           dropout=params["train"].get("dropout", 0.0))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    from src.training.dataset import load_params
    params = load_params()
    for name in params["train"]["backbones"]:
        model = build_image_model(params, name, pretrained=False)
        print(f"{name:18s} {count_parameters(model) / 1e6:6.1f}M parameters, "
              f"{model.num_features} image features")


if __name__ == "__main__":
    main()