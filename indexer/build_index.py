import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams, Distance, PointStruct, CreateAlias, CreateAliasOperation)
from common.embed import Embedder

version, onnx_path = sys.argv[1], sys.argv[2]        # e.g. v1 export/clip_vitb32_image.onnx
coll = f"catalog_{version}"
client, emb = QdrantClient(url="http://localhost:6333"), Embedder(onnx_path)
df = pd.read_csv("data/catalog.csv")

if client.collection_exists(coll):
    client.delete_collection(coll)

client.create_collection(coll, vectors_config=VectorParams(size=512, distance=Distance.COSINE))

for i in tqdm(range(0, len(df), 64)):
    b = df.iloc[i:i + 64]
    vecs = emb.embed([Image.open(p) for p in b["path"]])
    client.upsert(coll, points=[
        PointStruct(id=int(r.id), vector=v.tolist(),payload={"articleType": r.articleType, "path": r.path})
        for r, v in zip(b.itertuples(), vecs)]
    )

client.update_collection_aliases(change_aliases_operations=[
    CreateAliasOperation(create_alias=CreateAlias(collection_name=coll, alias_name="catalog"))]
)

print(coll, "points:", client.count(coll).count)