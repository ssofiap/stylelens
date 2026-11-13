import sys, time, numpy as np, pandas as pd
from PIL import Image
from qdrant_client import QdrantClient
from common.embed import Embedder

K = 5
emb, coll = Embedder(sys.argv[1]), sys.argv[2]       # onnx path, collection or alias
client = QdrantClient(url="http://localhost:6333")
prec, lat = [], []

for r in pd.read_csv("data/queries.csv").itertuples():
    img = Image.open(r.path)
    t0 = time.perf_counter(); v = emb.embed([img])[0]
    t1 = time.perf_counter()
    hits = client.query_points(coll, query=v.tolist(), limit=K, with_payload=True).points
    t2 = time.perf_counter()
    prec.append(np.mean([h.payload["articleType"] == r.articleType for h in hits]))
    lat.append((t1 - t0, t2 - t1))

prec, lat = np.array(prec), np.array(lat)[10:] * 1000   # drop 10 warm-up queries
print(f"precision@{K}: {prec.mean():.3f}")
print("embed  ms p50/p99:", np.percentile(lat[:, 0], [50, 99]).round(1))
print("search ms p50/p99:", np.percentile(lat[:, 1], [50, 99]).round(1))

# precision@5: 0.834
# embed  ms p50/p99: [29.3 59.9]
# search ms p50/p99: [ 6.8 44.1]