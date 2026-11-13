import numpy as np, onnxruntime as ort
from common.preprocess import preprocess

# wraps the ONNX model and is reused by the API 
class Embedder:
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def embed(self, images):               # list of PIL images -> (n, 512), unit length
        x = np.stack([preprocess(i) for i in images]).astype(np.float32)
        e = self.sess.run(None, {"image": x})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)