"""Unit tests for src/training/analysis.py image corruptions."""
import numpy as np
from PIL import Image

from src.training.analysis import CORRUPTIONS


def test_corruptions_keep_size_and_change_pixels():
    img = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))
    base = np.asarray(img, dtype=float)
    for name, corrupt in CORRUPTIONS.items():
        out = corrupt(img)
        assert out.size == img.size and out.mode == "RGB"
        diff = np.abs(np.asarray(out, dtype=float) - base).mean()
        assert (diff == 0) == (name == "clean"), name
