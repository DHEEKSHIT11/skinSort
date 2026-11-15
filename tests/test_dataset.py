"""Unit tests for src/training/dataset.py using a tiny fake dataset (no GCS needed)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.training.dataset import load_params, load_table, make_loader, answer_feature_columns


@pytest.fixture
def params(tmp_path):
    p = load_params()
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(12):
        image_id = f"img{i}"
        Image.fromarray(np.full((64, 64, 3), i * 20, dtype=np.uint8)).save(images / f"{image_id}.jpg")
        row = {"image_id": image_id, "split": "pad_train" if i < 9 else "val",
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


def test_load_table_filters_split(params):
    assert len(load_table(params, ["pad_train"])) == 9
    assert len(load_table(params, ["val"])) == 3


def test_unknown_split_fails(params):
    with pytest.raises(ValueError):
        load_table(params, ["no_such_split"])


def test_batch_shapes_and_labels(params):
    df = load_table(params, ["pad_train"])
    batch = next(iter(make_loader(df, params, train=True, with_answers=True)))
    assert tuple(batch["image"].shape) == (4, 3, 32, 32)
    assert set(batch["label"].tolist()) <= {0, 1, 2}
    assert tuple(batch["answers"].shape) == (4, len(answer_feature_columns(params)))


def test_missing_answers_become_zero(params):
    df = load_table(params, ["pad_train"])
    loader = make_loader(df, params, train=False, with_answers=True)
    answers = loader.dataset.answers
    assert not np.isnan(answers).any()


def test_data_version_reads_dvc_lock():
    from src.training.dataset import data_version
    info = data_version(load_params())
    assert len(info["data_dataset_md5"]) == 32
    assert info["data_images_md5"].endswith(".dir")
