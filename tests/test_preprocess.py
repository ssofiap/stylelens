from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from common.preprocess import preprocess

FIX = Path(__file__).parent / "fixtures"
GOLDEN = np.load(FIX / "golden_preprocess.npz")


def load(name: str) -> Image.Image:
    return Image.open(FIX / f"{name}.png").convert("RGB")


def test_shape():
    assert preprocess(load("img0")).shape == (3, 224, 224)


@pytest.mark.parametrize("name", GOLDEN.files)
def test_matches_open_clip(name):
    got = preprocess(load(name)).astype(np.float32)
    np.testing.assert_allclose(got, GOLDEN[name][0], atol=1e-4)  # golden is (1, C, H, W)
