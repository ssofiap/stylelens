from pathlib import Path

import numpy as np
import pytest

from common.model_config import CLIP_ARCH, CLIP_PRETRAINED

MODEL = Path("export/clip_vitb32_image.onnx")
GOLDEN = np.load(Path(__file__).parent / "fixtures" / "golden_preprocess.npz")


@pytest.mark.skipif(not MODEL.is_file(), reason="model artifact not in repo")
def test_onnx_matches_pytorch():
    torch = pytest.importorskip("torch")
    open_clip = pytest.importorskip("open_clip")
    import onnxruntime as ort

    model, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    model.eval()
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for key in GOLDEN.files:
        x = GOLDEN[key]
        with torch.no_grad():
            ref = model.encode_image(torch.from_numpy(x)).numpy()[0]
        out = sess.run(None, {name: x})[0][0]
        cos = ref @ out / (np.linalg.norm(ref) * np.linalg.norm(out))
        assert cos > 0.9999, f"{key}: cos={cos:.6f}"
