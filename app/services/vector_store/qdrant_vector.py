"""Qdrant vector store backend for the Hanzo RAG API.

Targets `hanzoai/vector` (Qdrant) — the native Hanzo vector database — as a
drop-in replacement for the pgvector backend. It satisfies exactly the
vector_store contract that `app/routes/document_routes.py` drives:

    add_documents(docs, ids=[file_id]*n)        / aadd_documents(...)
    similarity_search_with_score_by_vector(emb, k, filter={"file_id": ...})
        and the async variant (run via run_in_executor)
    get_all_ids()           -> list[str]   (distinct file_ids)
    get_filtered_ids(ids)   -> list[str]   (subset that exist)
    get_documents_by_ids(ids) -> list[Document]
    delete(ids, collection_only=False)

LibreChat's RAG API addresses everything by `file_id`: every chunk of an
uploaded file is inserted with `ids=[file_id]*n` and later queried/deleted by
that id. pgvector stored `file_id` as the non-unique `custom_id` column. Qdrant
point ids must be unique (UUID/uint), so we let langchain_qdrant assign unique
point ids and instead persist `file_id` inside each point's payload metadata —
then filter/scroll on `metadata.file_id` for get_*/delete. The route-level
Mongo-style filter (`{"file_id": x}` / `{"file_id": {"$in": [...]}}`) is
translated to a native Qdrant filter here.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# langchain_qdrant stores the original LangChain Document metadata under this
# payload key and the page content under the content key. We read/scroll the
# nested `metadata.file_id` field accordingly.
METADATA_KEY = "metadata"
CONTENT_KEY = "page_content"
FILE_ID_PATH = f"{METADATA_KEY}.file_id"


def _ensure_collection(
    client: QdrantClient, collection_name: str, dim: int, distance: models.Distance
) -> None:
    """Create the collection if it doesn't exist and index the file_id payload
    field so filter/scroll on metadata.file_id is fast and reliable."""
    try:
        client.get_collection(collection_name)
    except (UnexpectedResponse, ValueError):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dim, distance=distance),
        )
    # Payload index is idempotent; ignore "already exists".
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=FILE_ID_PATH,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:  # noqa: BLE001 - index may already exist
        pass


def _file_id_filter(value: Any) -> models.Filter:
    """Translate the route-level Mongo-style file_id filter into a native
    Qdrant filter. Accepts {"file_id": "x"}, {"file_id": {"$in": [...]}},
    {"file_id": {"$eq": "x"}}, or a bare string/list."""
    # Unwrap {"file_id": ...}
    if isinstance(value, dict) and "file_id" in value:
        value = value["file_id"]
    if isinstance(value, dict):
        if "$in" in value:
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key=FILE_ID_PATH,
                        match=models.MatchAny(any=list(value["$in"])),
                    )
                ]
            )
        if "$eq" in value:
            value = value["$eq"]
    if isinstance(value, (list, tuple)):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=FILE_ID_PATH, match=models.MatchAny(any=list(value))
                )
            ]
        )
    return models.Filter(
        must=[
            models.FieldCondition(
                key=FILE_ID_PATH, match=models.MatchValue(value=value)
            )
        ]
    )


class QdrantVector(QdrantVectorStore):
    """QdrantVectorStore extended with the id helpers + Mongo-filter shim that
    the RAG API routes expect. Construct via :meth:`from_config`."""

    @classmethod
    def from_config(
        cls,
        *,
        url: str,
        api_key: Optional[str],
        collection_name: str,
        embeddings: Embeddings,
        dim: int,
        distance: str = "cosine",
        prefer_grpc: bool = False,
    ) -> "QdrantVector":
        client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        dist = {
            "cosine": models.Distance.COSINE,
            "euclid": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }.get(distance.lower(), models.Distance.COSINE)
        _ensure_collection(client, collection_name, dim, dist)
        return cls(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            content_payload_key=CONTENT_KEY,
            metadata_payload_key=METADATA_KEY,
        )

    # ---- id-based helpers (file_id lives in metadata.file_id) --------------

    def add_documents(self, documents: list[Document], ids=None, **kwargs) -> list[str]:
        """LibreChat passes ids=[file_id]*n. Stamp file_id into each document's
        metadata so it can be filtered/deleted by file_id, and let Qdrant assign
        unique point ids. Returns the file_ids (one per chunk) to preserve the
        route's expectation that len(returned) == len(documents)."""
        file_ids = self._stamp_file_ids(documents, ids)
        super().add_documents(documents, **kwargs)
        return file_ids

    async def aadd_documents(
        self, documents: list[Document], ids=None, **kwargs
    ) -> list[str]:
        file_ids = self._stamp_file_ids(documents, ids)
        await run_in_executor(None, lambda: super(QdrantVector, self).add_documents(documents, **kwargs))
        return file_ids

    @staticmethod
    def _stamp_file_ids(documents: list[Document], ids) -> list[str]:
        file_ids: list[str] = []
        for i, doc in enumerate(documents):
            fid = None
            if ids and i < len(ids):
                fid = ids[i]
            if fid is None:
                fid = doc.metadata.get("file_id")
            if fid is not None:
                doc.metadata["file_id"] = fid
            file_ids.append(fid)
        return file_ids

    def get_all_ids(self) -> list[str]:
        return sorted(self._scroll_file_ids(None))

    def get_filtered_ids(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        present = self._scroll_file_ids(_file_id_filter(list(ids)))
        wanted = set(ids)
        return [i for i in sorted(present) if i in wanted]

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        docs: list[Document] = []
        points = self._scroll_points(_file_id_filter(list(ids)), with_vectors=False)
        for p in points:
            payload = p.payload or {}
            meta = payload.get(METADATA_KEY, {}) or {}
            docs.append(
                Document(page_content=payload.get(CONTENT_KEY, ""), metadata=meta)
            )
        return docs

    def delete(
        self, ids: Optional[list[str]] = None, collection_only: bool = False, **kwargs
    ) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=_file_id_filter(list(ids))),
            wait=True,
        )

    # ---- similarity search with the route's Mongo-style filter ------------

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, filter: Any = None, **kwargs
    ):
        qfilter = _file_id_filter(filter) if filter else None
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=qfilter,
            limit=k,
            with_payload=True,
        )
        out = []
        for r in results:
            payload = r.payload or {}
            meta = payload.get(METADATA_KEY, {}) or {}
            out.append(
                (Document(page_content=payload.get(CONTENT_KEY, ""), metadata=meta), r.score)
            )
        return out

    async def asimilarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, filter: Any = None, **kwargs
    ):
        return await run_in_executor(
            None, self.similarity_search_with_score_by_vector, embedding, k, filter
        )

    # ---- internal scroll helpers -----------------------------------------

    def _scroll_points(self, qfilter, with_vectors: bool = False) -> list[Any]:
        points: list[Any] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qfilter,
                with_payload=True,
                with_vectors=with_vectors,
                limit=256,
                offset=offset,
            )
            points.extend(batch)
            if offset is None:
                break
        return points

    def _scroll_file_ids(self, qfilter) -> set[str]:
        ids: set[str] = set()
        for p in self._scroll_points(qfilter, with_vectors=False):
            meta = (p.payload or {}).get(METADATA_KEY, {}) or {}
            fid = meta.get("file_id")
            if fid is not None:
                ids.add(fid)
        return ids
