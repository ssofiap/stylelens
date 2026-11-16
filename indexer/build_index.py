"""Build catalog_v{N} with registry model version N, verify it, optionally promote.

Promotion = atomically repointing the alias. The previous collection is kept for rollback.
"""

import csv
import os
import sys

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PointStruct,
    VectorParams,
)

from common.embed import Embedder
from common.registry import resolve

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CATALOG_CSV = os.getenv("CATALOG_CSV", "data/catalog.csv")
ALIAS = os.getenv("ALIAS", "catalog")
BATCH = int(os.getenv("BATCH", "64"))
PROMOTE = os.getenv("PROMOTE", "false").lower() == "true"
MODEL_VERSION = os.environ["MODEL_VERSION"]  # required: registry version N
MODEL_PATH = os.getenv("MODEL_PATH")  # dev override: skip the registry


def alias_target(client: QdrantClient, alias: str) -> str | None:
    for a in client.get_aliases().aliases:
        if a.alias_name == alias:
            return a.collection_name
    return None


def point_alias(client: QdrantClient, alias: str, coll: str, live: str | None) -> None:
    ops = []
    if live:
        ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
    ops.append(
        CreateAliasOperation(create_alias=CreateAlias(collection_name=coll, alias_name=alias))
    )
    client.update_collection_aliases(change_aliases_operations=ops)  # one request = atomic


def main() -> None:
    coll = f"catalog_v{MODEL_VERSION}"
    client = QdrantClient(url=QDRANT_URL)
    live = alias_target(client, ALIAS)
    if live == coll:
        sys.exit(f"{coll} is live behind '{ALIAS}'; refusing to rebuild it in place")

    path = MODEL_PATH or resolve(version=MODEL_VERSION)[0]
    emb = Embedder(path)
    with open(CATALOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if client.collection_exists(coll):
        client.delete_collection(coll)  # safe: not live, checked above
    client.create_collection(coll, vectors_config=VectorParams(size=512, distance=Distance.COSINE))

    for i in range(0, len(rows), BATCH):
        b = rows[i : i + BATCH]
        imgs = [Image.open(r["path"].replace("\\", "/")).convert("RGB") for r in b]
        vecs = emb.embed(imgs)
        client.upsert(
            coll,
            points=[
                PointStruct(
                    id=int(r["id"]),
                    vector=v.tolist(),
                    payload={"articleType": r["articleType"], "path": r["path"]},
                )
                for r, v in zip(b, vecs, strict=True)
            ],
            wait=True,
        )
        if (i // BATCH) % 10 == 0:
            print(f"{i + len(b)}/{len(rows)}", flush=True)

    n = client.count(coll, exact=True).count
    if n != len(rows):
        sys.exit(f"{coll}: {n} points, expected {len(rows)}; alias not moved")
    print(f"{coll}: {n} points OK", flush=True)

    if live is None or PROMOTE:
        point_alias(client, ALIAS, coll, live)
        print(f"alias '{ALIAS}': {live} -> {coll}")
    else:
        print(f"alias '{ALIAS}' left on {live}; set PROMOTE=true to switch")


if __name__ == "__main__":
    main()
