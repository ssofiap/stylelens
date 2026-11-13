# stylelens

Fashion visual search with a production-style ML lifecycle: model registry, CI evaluation gate, blue/green vector-index swap, and drift monitoring on Kubernetes 

> Status: work in progress ⋆˚࿔

## what it does ⭑.ᐟ

Upload a clothing photo and get the most visually similar catalog items back.

The focus is the MLOps around the model, not the model itself:

- Versioned embedding models in an MLflow registry (`champion` / `candidate` aliases)
- CI that blocks promotion if retrieval quality or latency regresses
- Zero-downtime embedding-model upgrades via Qdrant collection aliases, with rollback
- Prometheus/Grafana monitoring of latency and unlabeled drift signals

## architecture ⭑.ᐟ

```
                 +-----------------+
 git push -----> | GitHub Actions  |--- lint, tests, eval gate, build image ---> GHCR
                 +-----------------+
                          |
                          v
+-----------+    +------------------+     +--------------------+
|  Client   |--->|  FastAPI service |---->|  Qdrant            |
| (curl/UI) |    |  /search         |     |  alias: catalog -> |
+-----------+    |  /healthz        |     |  catalog_v{N}      |
                 |  /metrics        |     +--------------------+
                 |  ONNX CLIP       |               ^
                 +--------+---------+               |
                          |                +--------+---------+
                          |                |  Indexer Job     |
              resolve     |                |  (K8s Job):      |
              `champion`  v                |  embed catalog   |
                 +------------------+      |  with model v{N} |
                 |  MLflow registry |----->+------------------+
                 |  ONNX artifacts, |
                 |  eval metrics    |
                 +------------------+

 Prometheus <--- /metrics ---  FastAPI        Grafana <--- Prometheus
```

## stack ⭑.ᐟ

| Layer | Choice |
|---|---|
| Embedding model | CLIP ViT-B/32 exported to ONNX (512-d), CPU inference |
| Vector DB | Qdrant (collection aliases for index swaps) |
| Registry / tracking | MLflow |
| API | FastAPI, Uvicorn |
| Packaging | Multi-stage Docker, pinned dependencies |
| CI/CD | GitHub Actions, GHCR |
| Orchestration | Kubernetes (kind), Helm |
| Observability | Prometheus, Grafana |
| Load testing | k6 or Locust |

## repository layout ⭑.ᐟ

```
stylelens/
  app/                 FastAPI service (inference, search, metrics)
  indexer/             Catalog embedding + Qdrant upsert job
  eval/                Offline evaluation (precision@k, latency)
  export/              PyTorch -> ONNX export and registration
  deploy/
    helm/              Charts: api, qdrant, indexer
    kind/              Local cluster config
    grafana/           Dashboards (JSON)
    prometheus/        Alert rules
  .github/workflows/   CI pipeline
  Makefile
```

## quickstart ⭑.ᐟ
**TBD.**

## model lifecycle ⭑.ᐟ

1. **Export**: the embedding model is exported to ONNX and logged to MLflow with its eval metrics.
2. **Candidate**: a new version receives the `candidate` alias.
3. **Index**: the indexer Job embeds the catalog with that version into a new collection `catalog_vN`.
4. **Gate**: CI runs offline evaluation against the holdout set and compares with `champion`.
5. **Promote**: if the gate passes, the Qdrant alias `catalog` is repointed to `catalog_vN` and MLflow `champion` moves to the new version.
6. **Rollback**: repoint the alias back to the previous collection. No reindex is needed, because the old collection is retained until explicitly deleted.

A new embedding model changes the vector space, so queries embedded with model N cannot search an index built with model M. The API therefore resolves the model version and the collection together; they are never swapped independently.

## evaluation ⭑.ᐟ

- **Data:** [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). Catalog subset (up to 5k images) and a fixed holdout of about 500 queries. MIT license.
- **Metric:** precision@k, where a result is relevant if it shares the query's article type. This is a coarse proxy for visual similarity; it does not capture color, pattern or style.
- **Gate:** CI fails if precision@k falls below the current champion's, or if p99 latency regresses beyond a set margin. 

| Model | Format | precision@5 | embed p50 (ms) | embed p99 (ms) | search p99 (ms) |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | PyTorch | TBD | TBD | TBD | TBD |
| CLIP ViT-B/32 | ONNX | TBD | TBD | TBD | TBD |
| CLIP ViT-B/16 (candidate) | ONNX | TBD | TBD | TBD | TBD |

Benchmark setup (**TBD PROGRESSIVLEY**): 
- CPU model: Intel(R) Core(TM) Ultra 7 258V
- Number of queries: 500 
- More: (concurrency, warm-up)

## monitoring ⭑.ᐟ

Exposed at `/metrics`:

| Metric | Type | Purpose |
|---|---|---|
| `request_latency_seconds` | histogram | End-to-end latency |
| `embed_latency_seconds` | histogram | Model inference time |
| `search_latency_seconds` | histogram | Vector search time |
| `top1_similarity` | histogram | Distribution of best-match cosine similarity |
| `input_brightness`, `input_pixels` | histogram | Input image statistics |
| `drift_psi{feature}` | gauge | PSI against a reference window |
| `model_version_info` | gauge | Serving model/collection version |

**Drift without labels.** Retrieval has no ground truth in production, so drift is tracked on proxies: the top-1 similarity distribution and input image statistics, each compared with a reference window using the Population Stability Index:

```
PSI = sum_i (p_i - q_i) * ln(p_i / q_i)
```

where `p_i` and `q_i` are the reference and current fractions in bin `i`. A common rule of thumb treats PSI above 0.2 as a significant shift; this project uses a threshold of **TBD** and alerts through a Prometheus rule. PSI on proxies indicates that inputs or similarity behavior changed, not that quality dropped.

## design decisions ⭑.ᐟ

| Decision | Rationale | Alternative considered |
|---|---|---|
| ONNX Runtime on CPU | Lower latency and smaller image than PyTorch; no GPU needed at this scale | PyTorch eager, Triton (heavier than needed) |
| Qdrant collection aliases | Atomic index swap and cheap rollback | Overwriting one collection (no rollback) |
| Offline eval as a CI gate | Prevents promoting a worse model automatically | Manual review |
| Label-based precision@k | Cheap, reproducible, automatable | Human relevance judgments (not feasible here) |
| Reindex as a Job | Reproducible, observable, retryable | Reindex inside the API at startup |

**TBD**: add measured trade-offs after benchmarking.

## limitations ⭑.ᐟ

- Precision@k with category labels does not measure fine-grained visual similarity.
- The catalog is a small subset; behavior at millions of items (ANN recall, sharding, index build time) is untested.
- Single-node kind cluster; no multi-zone availability, autoscaling under real load, or cloud deployment.
- Drift signals are proxies and have not been validated against real quality degradation.
- CPU-only inference.

## future work ⭑.ᐟ

- ArgoCD for GitOps sync of the Helm releases
- Terraform and managed Kubernetes deployment
- Canary traffic split between two model versions with online comparison
- Fine-tuning on catalog data and re-running the promotion gate

## license and data ⭑.ᐟ

Code: MIT. Images are not redistributed in this repository.