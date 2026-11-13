import numpy as np
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# reimplements CLIP’s preprocessing with PIL and numpy only, so the serving image won’t need Pytorch
def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    nw, nh = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    img = img.resize((nw, nh), Image.BICUBIC)
    l, t = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    img = img.crop((l, t, l + size, t + size))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1) # channels-first, shape (3, 224, 224)