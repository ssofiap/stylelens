import io
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Info, generate_latest
from qdrant_client import QdrantClient

from common.preprocess import preprocess  # SAME function the indexer uses
from common.registry import MODEL_NAME, resolve

MODEL_PATH = os.getenv("MODEL_PATH")  # dev override: skip the registry
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_VERSION = os.getenv("MODEL_VERSION")  # if set, pins a version, ignores alias
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "catalog")  # an alias, later
MODEL_INFO = Info("stylelens_model", "Loaded embedding model")

LAT = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5)
EMBED_S = Histogram("stylelens_embed_seconds", "Image -> vector", buckets=LAT)
SEARCH_S = Histogram("stylelens_search_seconds", "Vector -> top-k", buckets=LAT)
REQ_S = Histogram("stylelens_request_seconds", "End-to-end /search", buckets=LAT)
REQS = Counter("stylelens_requests_total", "Requests", ["status"])
TOP1 = Histogram(
    "stylelens_top1_similarity", "Cosine sim of best hit", buckets=[i / 20 for i in range(1, 21)]
)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH:
        path, version = MODEL_PATH, "local"
    else:
        path, version = resolve(MODEL_ALIAS, MODEL_VERSION)
    state["version"] = str(version)
    MODEL_INFO.info({"name": MODEL_NAME, "version": state["version"]})
    state["sess"] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    state["input"] = state["sess"].get_inputs()[0].name
    state["qdrant"] = QdrantClient(url=QDRANT_URL)
    state["sess"].run(None, {state["input"]: np.zeros((1, 3, 224, 224), np.float32)})  # warm-up
    yield


app = FastAPI(title="StyleLens", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": state.get("version")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
def search(
    file: Annotated[UploadFile, File()],
    k: Annotated[int, Query(ge=1, le=50)] = 5,
):
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except (OSError, Image.DecompressionBombError):
        REQS.labels("400").inc()
        raise HTTPException(400, "not a valid image") from None

    t1 = time.perf_counter()
    x = preprocess(img).astype(np.float32)[None, ...]
    v = state["sess"].run(None, {state["input"]: x})[0][0]
    v = v / np.linalg.norm(v)  # must match the indexer
    t2 = time.perf_counter()
    hits = state["qdrant"].query_points(COLLECTION, query=v.tolist(), limit=k).points
    t3 = time.perf_counter()

    EMBED_S.observe(t2 - t1)
    SEARCH_S.observe(t3 - t2)
    REQ_S.observe(t3 - t0)
    if hits:
        TOP1.observe(hits[0].score)

    REQS.labels("200").inc()

    return {
        "results": [
            {"id": h.id, "score": h.score, "articleType": h.payload.get("articleType")}
            for h in hits
        ]
    }
