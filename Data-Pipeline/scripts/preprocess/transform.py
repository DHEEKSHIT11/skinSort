"""Standardise images and encode the tabular features.

Images: centre-crop to a square, resize to image_size, apply Shades-of-Gray
colour constancy, save as JPEG -> data/processed/images/<image_id>.jpg

Features (data/interim/features.csv), one row per image:
  age (median-filled) + age_missing, age_bin_*
  sex_female / sex_male / sex_unknown
  site_<body_site> one-hot
  q_<answer> as 1/0 with q_<answer>_missing. Only PAD-UFES-20 has answers;
  for every other image they are all marked missing rather than invented.

Training-time augmentation (blur, crop, brightness) is NOT done here; it is
applied on the fly during training so each epoch sees different variants.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from scripts.common.config import load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger
from scripts.preprocess.harmonize import ANSWER_COLUMNS

logger = get_logger("preprocess.transform")

SEXES = ["female", "male", "unknown"]
BODY_SITES = ["head_neck", "trunk", "upper_extremity", "lower_extremity", "acral",
              "genital", "unknown"]


# --- images ---------------------------------------------------------------------

def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Crop the largest centred square, then resize to size x size."""
    return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)


def shades_of_gray(pixels: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy on an HxWx3 uint8 array.

    Estimates the light colour with the Minkowski p-norm of each channel and
    rescales channels so that light becomes neutral grey.
    """
    rgb = pixels.astype(np.float32)
    illuminant = np.power(np.mean(np.power(rgb, power), axis=(0, 1)), 1.0 / power)
    illuminant = np.maximum(illuminant, 1e-6)
    gains = illuminant.mean() / illuminant
    return np.clip(rgb * gains, 0, 255).astype(np.uint8)


def transform_image(paths: tuple, size: int, power: int, quality: int) -> str:
    """Process one image; returns '' on success or an error message."""
    src, dest = paths
    try:
        with Image.open(src) as img:
            img.draft("RGB", (2 * size, 2 * size))  # fast reduced-scale JPEG decode
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = center_crop_resize(img, size)
            out = Image.fromarray(shades_of_gray(np.asarray(img), power))
            out.save(dest, "JPEG", quality=quality)
        return ""
    except Exception as exc:  # reported per image, the step fails if any image fails
        return f"{type(exc).__name__}: {exc}"


def transform_images(df: pd.DataFrame, raw_dir: Path, out_dir: Path,
                     cfg: Dict[str, Any], workers: int = 0) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(raw_dir / p), str(out_dir / f"{i}.jpg"))
            for p, i in zip(df["image_path"], df["image_id"])]
    fn = partial(transform_image, size=cfg["image_size"],
                 power=cfg["color_constancy_power"], quality=cfg["jpeg_quality"])
    n_workers = workers if workers and workers > 0 else (os.cpu_count() or 1)
    logger.info("Transforming %d images with %d processes", len(jobs), n_workers)
    if n_workers == 1:
        return [fn(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(fn, jobs, chunksize=64))


# --- tabular features ---------------------------------------------------------------

def encode_features(df: pd.DataFrame, age_bins: List[float]) -> pd.DataFrame:
    """Encode age, sex, body site and patient answers into numeric columns."""
    out = pd.DataFrame({"image_id": df["image_id"]})
    age = pd.to_numeric(df["age"], errors="coerce")
    out["age_missing"] = age.isna().astype(int)
    out["age"] = age.fillna(age.median() if age.notna().any() else 0.0)
    labels = [f"age_bin_{int(lo)}_{int(hi)}" for lo, hi in zip(age_bins[:-1], age_bins[1:])]
    bins = pd.cut(age, bins=age_bins, labels=labels, right=False)
    for label in labels:
        out[label] = (bins == label).astype(int)
    for sex in SEXES:
        out[f"sex_{sex}"] = (df["sex"].fillna("unknown") == sex).astype(int)
    for site in BODY_SITES:
        out[f"site_{site}"] = (df["body_site"].fillna("unknown") == site).astype(int)
    for column in ANSWER_COLUMNS:
        answer = pd.to_numeric(df[column], errors="coerce")
        out[f"{column}_missing"] = answer.isna().astype(int)
        out[column] = answer.fillna(0).astype(int)
    return out


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    interim, processed = resolve_path(params, "interim"), resolve_path(params, "processed")
    cfg = params["preprocess"]["transform"]
    df = pd.read_csv(interim / "deduped.csv", low_memory=False)

    errors = transform_images(df, resolve_path(params, "raw"), processed / "images", cfg,
                              params["preprocess"]["workers"])
    failed = [(i, e) for i, e in zip(df["image_id"], errors) if e]
    if failed:
        logger.error("%d images failed to transform, e.g. %s", len(failed), failed[:3])
        raise RuntimeError(f"{len(failed)} images failed to transform")

    features = encode_features(df, cfg["age_bins"])
    features.to_csv(interim / "features.csv", index=False)
    summary = {"images": int(len(df)), "feature_columns": int(features.shape[1] - 1),
               "image_size": cfg["image_size"]}
    logger.info("Transform complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize/normalise images, encode features")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
