"""Unit test for src/training/train.py: trains a tiny model for 2 epochs on fake data."""
import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.training.dataset import load_params, load_table
from src.training.train import train_one_backbone, val_metrics


@pytest.fixture
def params(tmp_path):
    p = load_params()
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(12):
        image_id = f"img{i}"
        Image.fromarray(np.full((64, 64, 3), i * 20, dtype=np.uint8)).save(images / f"{image_id}.jpg")
        rows.append({"image_id": image_id, "split": "train" if i < 9 else "val",
                     "urgency": ["routine", "priority", "urgent"][i % 3]})
    pd.DataFrame(rows).to_csv(tmp_path / "dataset.csv", index=False)
    p["data"]["dataset_csv"] = str(tmp_path / "dataset.csv")
    p["data"]["images_dir"] = str(images)
    p["train"].update(image_size=32, batch_size=4, num_workers=0)
    return p


def test_val_metrics_perfect_model():
    labels = np.array([0, 1, 2, 2])
    probs = np.eye(3)[labels]
    m = val_metrics(probs, labels, urgent_idx=2)
    assert m["urgent_recall"] == 1.0 and m["urgent_auc"] == 1.0 and m["macro_f1"] == 1.0


def test_train_one_backbone_saves_checkpoint(params, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    best = train_one_backbone(params, "resnet18", load_table(params, ["train"]),
                              load_table(params, ["val"]), torch.device("cpu"),
                              pretrained=False, epochs=2, out_dir=tmp_path / "model")
    assert (tmp_path / "model" / "best.pt").exists()
    assert 0.0 <= best["macro_f1"] <= 1.0
    assert best["epoch"] in (1, 2)