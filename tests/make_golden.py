"""Regenerate golden preprocessing fixtures. Local only (needs torch + open_clip).
Run after any deliberate change to preprocessing or model family:
    python tests/make_golden.py
"""

from pathlib import Path

import numpy as np
import open_clip
from PIL import Image

from common.model_config import CLIP_ARCH, CLIP_PRETRAINED

OUT = Path(__file__).parent / "fixtures"
SIZES = [(300, 400), (640, 480), (224, 224)]  # portrait, landscape, exact size


def synthetic(w: int, h: int) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // w, y * 255 // h, ((x // 16 + y // 16) % 2) * 255], -1)
    return Image.fromarray(img.astype(np.uint8))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _, _, tf = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    expected = {}
    for i, (w, h) in enumerate(SIZES):
        path = OUT / f"img{i}.png"
        synthetic(w, h).save(path)
        img = Image.open(path).convert("RGB")  # reload: test reads the same bytes
        expected[f"img{i}"] = tf(img).unsqueeze(0).numpy()
    np.savez_compressed(OUT / "golden_preprocess.npz", **expected)


if __name__ == "__main__":
    main()
