# Hanzo Chat RAG API

FastAPI RAG service for Hanzo Chat (LibreChat fork): document ingest,
embedding, chunking, and vector similarity retrieval. Mounted by chat via
`RAG_API_URL`.

**Repo**: `github.com/hanzoai/chat-rag-api`
**Upstream**: `danny-avila/rag_api` (MIT)
**Image**: `ghcr.io/hanzoai/chat-rag-api`
**Runtime**: Python 3.10 / FastAPI / LangChain 0.3

## Vector backends (`VECTOR_DB_TYPE`)

| Value | Store | Notes |
|-------|-------|-------|
| `pgvector` (default) | Postgres + pgvector | legacy |
| `atlas-mongo` | MongoDB Atlas vector | |
| `qdrant` | **hanzoai/vector (Qdrant)** | native Hanzo vector DB |

### Qdrant backend (`hanzoai/vector`)

`app/services/vector_store/qdrant_vector.py` — `QdrantVector(QdrantVectorStore)`.
Targets the deployed `hanzoai/vector` (Qdrant) service. Satisfies the exact
vector_store contract `app/routes/document_routes.py` drives:

- `add_documents(docs, ids=[file_id]*n)` / `aadd_documents` — LibreChat inserts
  every chunk of a file under the same `file_id`. Qdrant point ids must be
  unique, so `file_id` is stamped into each point's payload `metadata.file_id`
  and Qdrant assigns unique point ids.
- `similarity_search_with_score_by_vector(emb, k, filter={"file_id": ...})`
  (+ async) — the route's Mongo-style filter (`{"file_id": x}` /
  `{"file_id": {"$in": [...]}}`) is translated to a native Qdrant filter on
  `metadata.file_id`.
- `get_all_ids` / `get_filtered_ids(ids)` / `get_documents_by_ids(ids)` —
  Qdrant scroll over `metadata.file_id`.
- `delete(ids, collection_only=False)` — FilterSelector on `metadata.file_id`.

Config: `QDRANT_URL` (default `http://vector:6333`), `QDRANT_API_KEY`,
`QDRANT_DISTANCE` (cosine), `EMBEDDINGS_DIM` (1536 for text-embedding-3-small,
matching the live hanzoai/vector collection). The collection is created on
startup with a KEYWORD payload index on `metadata.file_id`.

Health: `qdrant_health_check` pings `client.get_collections()`.

## Test

```bash
pytest tests/test_qdrant_vector.py   # in-memory Qdrant, 4 tests
```

## Deploy (operator CR, no GHA)

Image built on-cluster via arcd → `ghcr.io/hanzoai/chat-rag-api:<tag>`.
Operator CR: `hanzoai/universe/infra/k8s/operator/crs/rag-api-v1.yaml`
(service `rag-api`). For Qdrant set on the CR:
`VECTOR_DB_TYPE=qdrant`, `QDRANT_URL=http://vector.hanzo.svc.cluster.local:6333`,
`EMBEDDINGS_DIM=1536`.
