"""Measure every image once: size, brightness, contrast, sharpness and perceptual hash.

Input : data/interim/harmonized.csv + raw images
Output: data/interim/image_stats.csv (one row per image_id)

These numbers feed the quality filter, the duplicate finder and the data
statistics/drift checks, so images are opened only once. The work is spread
over a process pool because it is CPU-bound.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from scripts.common.config import load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("preprocess.image_stats")

STATS_COLUMNS = ["image_id", "readable", "error", "width", "height", "brightness",
                 "contrast", "sharpness", "phash"]
SHARPNESS_SIDE = 512  # measure sharpness on a fixed-size copy so scores are comparable
DRAFT_SIDE = 1024     # JPEG decode target; must stay well above SHARPNESS_SIDE


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: low values mean a blurry image."""
    g = gray.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def measure_image(path: str) -> Dict[str, Any]:
    """Return quality statistics for one image file; never raises."""
    try:
        with Image.open(path) as img:
            width, height = img.size  # original size, before any reduced decoding
            # Optimisation: JPEGs are decoded directly at reduced scale (>= 1024 px);
            # every statistic below is computed on a 512 px copy anyway.
            img.draft("RGB", (DRAFT_SIDE, DRAFT_SIDE))
            img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
            img.load()
            gray = img.convert("L")
            small = gray.copy()
            small.thumbnail((SHARPNESS_SIDE, SHARPNESS_SIDE))
            pixels = np.asarray(small)
            return {
                "readable": True, "error": "",
                "width": width, "height": height,
                "brightness": float(pixels.mean()),
                "contrast": float(pixels.std()),
                "sharpness": laplacian_variance(pixels),
                "phash": str(imagehash.phash(gray)),
            }
    except Exception as exc:  # any decoding problem marks the image unreadable
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"[:200],
                "width": np.nan, "height": np.nan, "brightness": np.nan,
                "contrast": np.nan, "sharpness": np.nan, "phash": ""}


def _worker_count(setting: int) -> int:
    return setting if setting and setting > 0 else (os.cpu_count() or 1)


def compute_stats(df: pd.DataFrame, raw_dir: Path, workers: int = 0) -> pd.DataFrame:
    """Measure every image listed in df (needs image_id and image_path)."""
    paths = [str(raw_dir / p) for p in df["image_path"]]
    n_workers = _worker_count(workers)
    logger.info("Measuring %d images with %d processes", len(paths), n_workers)
    if n_workers == 1:
        results = [measure_image(p) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(measure_image, paths, chunksize=64))
    stats = pd.DataFrame(results)
    stats.insert(0, "image_id", df["image_id"].to_numpy())
    return stats[STATS_COLUMNS]


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    interim = resolve_path(params, "interim")
    interim.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(interim / "harmonized.csv", low_memory=False)
    stats = compute_stats(df, resolve_path(params, "raw"), params["preprocess"]["workers"])
    stats.to_csv(interim / "image_stats.csv", index=False)
    summary = {
        "images": int(len(stats)),
        "unreadable": int((~stats["readable"]).sum()),
        "median_brightness": round(float(stats["brightness"].median()), 1),
        "median_sharpness": round(float(stats["sharpness"].median()), 1),
    }
    logger.info("Image statistics: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-image quality statistics")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
