import numpy as np
import onnxruntime as ort
import open_clip
import pandas as pd
import torch
from PIL import Image

from common.preprocess import preprocess
from common.model_config import CLIP_ARCH, CLIP_PRETRAINED

OUT = "export/clip_vitb32_image.onnx"
model, _, ref_preprocess = open_clip.create_model_and_transforms(
    CLIP_ARCH, pretrained=CLIP_PRETRAINED
)
visual = model.visual.eval()  # image encoder only

torch.onnx.export(
    visual,
    torch.randn(1, 3, 224, 224),
    OUT,
    input_names=["image"],
    output_names=["embedding"],
    dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)  # if the exporter errors, add dynamo=False

img = Image.open(pd.read_csv("data/catalog.csv")["path"].iloc[0])
a, b = ref_preprocess(img).numpy(), preprocess(img)
print("preprocess max abs diff:", np.abs(a - b).max())

with torch.no_grad():
    t = visual(torch.from_numpy(b)[None]).numpy()

o = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"]).run(None, {"image": b[None]})[0]
print("onnx vs torch max abs diff:", np.abs(t - o).max())
