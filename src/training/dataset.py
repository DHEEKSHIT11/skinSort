"""Load SkinSort images and labels from the data pipeline output.

Reads Data-Pipeline/data/processed/dataset.csv (made by the `split` stage) and
the 384x384 images next to it. Every path and setting comes from params.yaml.

Quick check on real data:
    python -m src.training.dataset
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]

# timm's pretrained backbones expect ImageNet normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_params(path: Path = ROOT / "params.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Paths in params.yaml are relative to the repo root (absolute ones are kept)."""
    return ROOT / path


def data_version(params: Dict[str, Any]) -> Dict[str, str]:
    """DVC version of the data this run uses, read from Data-Pipeline/dvc.lock.

    Logged on every MLflow run so each result can be traced to the exact data:
    data_dataset_md5 / data_images_md5 are the DVC hashes, data_matches_lock says
    whether the dataset.csv on disk is that version (False = local edits or not pulled).
    """
    import hashlib

    csv = resolve(params["data"]["dataset_csv"])
    lock = ROOT / "Data-Pipeline" / "dvc.lock"
    info = {"data_dataset_md5": "unknown", "data_images_md5": "unknown"}
    if lock.exists():
        stages = yaml.safe_load(lock.read_text()).get("stages", {})
        for stage in stages.values():
            for out in stage.get("outs", []):
                if out["path"].endswith("processed/dataset.csv"):
                    info["data_dataset_md5"] = out["md5"]
                elif out["path"].endswith("processed/images"):
                    info["data_images_md5"] = out["md5"]
    if csv.exists():
        info["data_matches_lock"] = str(hashlib.md5(csv.read_bytes()).hexdigest()
                                        == info["data_dataset_md5"])
    return info


class RandomBlur:
    """Gaussian blur with probability p, to mimic soft phone photos."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return img


def build_transforms(params: Dict[str, Any], train: bool) -> transforms.Compose:
    size = params["train"]["image_size"]
    aug = params["augment"]
    if train:
        steps = [transforms.RandomResizedCrop(size, scale=tuple(aug["random_crop_scale"]))]
        if aug["horizontal_flip"]:
            steps.append(transforms.RandomHorizontalFlip())
        if aug["vertical_flip"]:
            steps.append(transforms.RandomVerticalFlip())
        steps.append(transforms.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]))
        steps.append(RandomBlur(aug["blur_prob"]))
    else:
        # images are already square from the pipeline; this only matters if image_size changes
        steps = [transforms.Resize((size, size))]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def answer_feature_columns(params: Dict[str, Any]) -> List[str]:
    """The 6 patient answers plus their 6 missing flags (12 inputs for the fusion model)."""
    answers = params["data"]["answer_columns"]
    return answers + [f"{c}_missing" for c in answers]


def load_table(params: Dict[str, Any], splits: List[str],
               limit: Optional[int] = None) -> pd.DataFrame:
    """Rows of dataset.csv for the given splits, e.g. ["train"] or ["test", "pad_test"]."""
    cfg = params["data"]
    df = pd.read_csv(resolve(cfg["dataset_csv"]), low_memory=False)
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for splits {splits}; check the split names in params.yaml")

    unknown = set(df[cfg["label_column"]]) - set(cfg["classes"])
    if unknown:
        raise ValueError(f"Unknown urgency labels: {unknown}")

    if limit and len(df) > limit:
        # keep the urgency mix when taking a small sample for a smoke test
        df = (df.groupby(cfg["label_column"])
                .sample(frac=limit / len(df), random_state=params["seed"])
                .reset_index(drop=True))
    return df


class SkinDataset(Dataset):
    """One item = processed image tensor, urgency label, and (optionally) patient answers."""

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any], train: bool,
                 with_answers: bool = False):
        cfg = params["data"]
        images_dir = resolve(cfg["images_dir"])
        self.paths = [images_dir / f"{image_id}.jpg" for image_id in df["image_id"]]
        class_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
        self.labels = df[cfg["label_column"]].map(class_to_idx).to_numpy()
        self.transform = build_transforms(params, train)
        self.answers = None
        if with_answers:
            cols = answer_feature_columns(params)
            # raw answers are empty when missing; the _missing flag already says so
            self.answers = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            self.answers = self.answers.to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with Image.open(self.paths[idx]) as img:
            image = self.transform(img.convert("RGB"))
        item = {"image": image, "label": torch.tensor(self.labels[idx], dtype=torch.long),
                "index": idx}
        if self.answers is not None:
            item["answers"] = torch.from_numpy(self.answers[idx])
        return item


def make_loader(df: pd.DataFrame, params: Dict[str, Any], train: bool,
                with_answers: bool = False) -> DataLoader:
    dataset = SkinDataset(df, params, train=train, with_answers=with_answers)
    batch_size = params["train"]["batch_size"]
    # In training, drop a leftover batch of 1 image: batch norm layers fail on it.
    # Only when there is more than one full batch, so tiny datasets still train.
    drop_last = train and len(dataset) > batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=params["train"]["num_workers"],
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last)


def main() -> None:
    params = load_params()
    for split in params["data"]["splits"].values():
        df = load_table(params, [split])
        print(f"{split:14s} {len(df):6d} images  {df['urgency'].value_counts().to_dict()}")

    df = load_table(params, ["pad_train"], limit=params["train"]["smoke_test_rows"])
    batch = next(iter(make_loader(df, params, train=True, with_answers=True)))
    print("image batch:", tuple(batch["image"].shape), " labels:", batch["label"][:8].tolist(),
          " answers:", tuple(batch["answers"].shape))


if __name__ == "__main__":
    main()
