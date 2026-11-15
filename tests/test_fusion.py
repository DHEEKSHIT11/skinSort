"""Unit tests for the fusion model and src/training/train_fusion.py on fake data."""
import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.fusion_model import FusionHead, FusionModel, build_fusion_head
from src.models.image_model import build_image_model
from src.training.dataset import answer_feature_columns, load_params, load_table
from src.training.train_fusion import find_best_image_checkpoint, train_fusion


@pytest.fixture
def params(tmp_path):
    p = load_params()
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(12):
        image_id = f"img{i}"
        Image.fromarray(np.full((64, 64, 3), i * 20, dtype=np.uint8)).save(images / f"{image_id}.jpg")
        row = {"image_id": image_id, "split": "pad_train" if i < 9 else "pad_val",
               "urgency": ["routine", "priority", "urgent"][i % 3]}
        for c in p["data"]["answer_columns"]:
            row[c] = np.nan if i % 2 else 1
            row[f"{c}_missing"] = i % 2
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "dataset.csv", index=False)
    p["data"]["dataset_csv"] = str(tmp_path / "dataset.csv")
    p["data"]["images_dir"] = str(images)
    p["train"].update(image_size=32, batch_size=4, num_workers=0)
    return p


def save_image_checkpoint(params, path, urgent_auc):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_image_model(params, "resnet18", pretrained=False)
    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18",
                "classes": params["data"]["classes"], "epoch": 1,
                "val_metrics": {"urgent_auc": urgent_auc, "macro_f1": 0.5}}, path)


def test_fusion_head_output_shape():
    head = FusionHead(num_image_features=16, num_answers=12, num_classes=3, hidden_dim=8)
    out = head(torch.randn(5, 16), torch.randn(5, 12))
    assert out.shape == (5, 3)


def test_fusion_model_freezes_backbone(params):
    image_model = build_image_model(params, "resnet18", pretrained=False)
    head = build_fusion_head(params, image_model.num_features, len(answer_feature_columns(params)))
    model = FusionModel(image_model, head)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    out = model(torch.randn(2, 3, 32, 32), torch.randn(2, 12))
    assert out.shape == (2, 3)


def test_find_best_image_checkpoint_skips_fusion(params, tmp_path):
    save_image_checkpoint(params, tmp_path / "a" / "best.pt", urgent_auc=0.7)
    save_image_checkpoint(params, tmp_path / "b" / "best.pt", urgent_auc=0.8)
    save_image_checkpoint(params, tmp_path / "fusion" / "best.pt", urgent_auc=0.99)
    assert find_best_image_checkpoint(tmp_path) == tmp_path / "b" / "best.pt"


def test_find_best_image_checkpoint_needs_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_best_image_checkpoint(tmp_path)


def test_train_fusion_saves_checkpoint(params, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    image_ckpt = tmp_path / "resnet18" / "best.pt"
    save_image_checkpoint(params, image_ckpt, urgent_auc=0.5)

    best = train_fusion(params, image_ckpt, load_table(params, ["pad_train"]),
                        load_table(params, ["pad_val"]), torch.device("cpu"),
                        epochs=2, out_dir=tmp_path / "fusion")
    saved = torch.load(tmp_path / "fusion" / "best.pt", weights_only=False)
    assert saved["backbone"] == "resnet18"
    assert saved["answer_columns"] == answer_feature_columns(params)
    assert best["epoch"] in (1, 2)
    assert 0.0 <= best["macro_f1"] <= 1.0
    assert "urgent_auc" in best["image_only"]
