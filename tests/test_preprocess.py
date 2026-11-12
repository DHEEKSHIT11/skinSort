"""Serving preprocessing must match the data pipeline's image transform."""
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data.preprocess import center_crop_resize, preprocess_photo, shades_of_gray
from src.training.dataset import ROOT, load_params

DATA_PIPELINE = ROOT / "Data-Pipeline"


def photo(w=300, h=200):
    """Smooth skin-like image with a reddish colour cast (noise would be a JPEG worst case)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 60 * np.sin(xx / 40.0)[..., None] * np.cos(yy / 30.0)[..., None]
    base = base * np.array([1.25, 0.9, 0.7])
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


@pytest.mark.skipif(not (DATA_PIPELINE / "params.yaml").exists(), reason="data pipeline not present")
def test_settings_match_data_pipeline():
    ours = load_params()["preprocess"]
    theirs = yaml.safe_load((DATA_PIPELINE / "params.yaml").read_text())["preprocess"]["transform"]
    for key in ("image_size", "color_constancy_power", "jpeg_quality"):
        assert ours[key] == theirs[key], key


@pytest.mark.skipif(not (DATA_PIPELINE / "scripts").exists(), reason="data pipeline code not present")
def test_same_pixels_as_data_pipeline():
    sys.path.insert(0, str(DATA_PIPELINE))
    try:
        from scripts.preprocess.transform import center_crop_resize as dp_crop
        from scripts.preprocess.transform import shades_of_gray as dp_sog
    finally:
        sys.path.remove(str(DATA_PIPELINE))
    img = photo()
    a = shades_of_gray(np.asarray(center_crop_resize(img, 384)), 6)  # same code path as serving
    b = dp_sog(np.asarray(dp_crop(img, 384)), 6)
    assert np.array_equal(a, b)


def test_preprocess_output_and_near_idempotent():
    cfg = load_params()["preprocess"]
    once = preprocess_photo(photo(), cfg)
    assert once.size == (384, 384) and once.mode == "RGB"
    twice = preprocess_photo(once, cfg)
    diff = np.abs(np.asarray(once, dtype=float) - np.asarray(twice, dtype=float)).mean()
    assert diff < 2.0          # processed images are (almost) unchanged by a second pass
    colour_cast = np.asarray(photo(), dtype=float).mean(axis=(0, 1))
    balanced = np.asarray(once, dtype=float).mean(axis=(0, 1))
    assert balanced.std() < colour_cast.std()   # the red cast is reduced
