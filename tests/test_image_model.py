"""Unit tests for src/models/image_model.py (small random inputs, no downloads)."""
import pytest
import timm
import torch

from src.models.image_model import build_image_model
from src.training.dataset import load_params


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
def test_output_shapes(name):
    params = load_params()
    model = build_image_model(params, name, pretrained=False).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(images)
        feats = model.features(images)
    assert logits.shape == (2, len(params["data"]["classes"]))
    assert feats.shape == (2, model.num_features)


def test_backbones_in_params_exist_in_timm():
    available = set(timm.list_models())
    for name in load_params()["train"]["backbones"]:
        assert name in available, f"{name} is not a timm model name"