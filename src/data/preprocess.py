"""Photo preprocessing for serving — the same steps the data pipeline applied to every
training image (Data-Pipeline/scripts/preprocess/transform.py):

    EXIF rotation -> largest centred square, resized to image_size (Lanczos)
    -> Shades-of-Gray colour constancy (p = color_constancy_power) -> JPEG at jpeg_quality

The settings live in params.yaml (preprocess section) and must match
Data-Pipeline/params.yaml (preprocess.transform); tests/test_preprocess.py checks both.
Serving inputs are raw photos, so this runs once, like in training. Running it again on an
already processed photo changes pixels by ~1/255 on average (JPEG re-encoding) and P(urgent)
by ~0.005, so evaluation scripts feed processed images to the model directly.
"""
import io
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageOps


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)


def shades_of_gray(pixels: np.ndarray, power: int = 6) -> np.ndarray:
    rgb = pixels.astype(np.float32)
    illuminant = np.power(np.mean(np.power(rgb, power), axis=(0, 1)), 1.0 / power)
    illuminant = np.maximum(illuminant, 1e-6)
    gains = illuminant.mean() / illuminant
    return np.clip(rgb * gains, 0, 255).astype(np.uint8)


def preprocess_photo(img: Image.Image, cfg: Dict[str, Any]) -> Image.Image:
    """A raw upload -> the same kind of image the model was trained on."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = center_crop_resize(img, cfg["image_size"])
    out = Image.fromarray(shades_of_gray(np.asarray(img), cfg["color_constancy_power"]))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=cfg["jpeg_quality"])      # same compression as training images
    buf.seek(0)
    return Image.open(buf).convert("RGB")
