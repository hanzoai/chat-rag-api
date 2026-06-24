"""Tests for the Qdrant (hanzoai/vector) backend.

Runs against an in-memory Qdrant (`QdrantClient(":memory:")`) with a tiny
deterministic fake embedding, exercising exactly the vector_store contract that
`app/routes/document_routes.py` drives: add by file_id, similarity search with
the route's Mongo-style file_id filter, the id helpers, and delete by file_id.
"""

import os
import sys

# Allow `import app...` when run from the repo root or tests/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.services.vector_store.qdrant_vector import (
    QdrantVector,
    CONTENT_KEY,
    METADATA_KEY,
    FILE_ID_PATH,
    _file_id_filter,
)

DIM = 8


class FakeEmbeddings(Embeddings):
    """Deterministic embedding: hashes each token into an 8-dim vector so that
    documents sharing words land near each other. Good enough to assert that
    filtering + retrieval wiring is correct."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in text.lower().split():
            vec[hash(tok) % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def make_store() -> QdrantVector:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="rag",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name="rag",
        field_name=FILE_ID_PATH,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    return QdrantVector(
        client=client,
        collection_name="rag",
        embedding=FakeEmbeddings(),
        content_payload_key=CONTENT_KEY,
        metadata_payload_key=METADATA_KEY,
    )


def _docs(file_id: str, n: int, word: str):
    return [
        Document(page_content=f"{word} chunk {i}", metadata={"file_id": file_id})
        for i in range(n)
    ]


def test_filter_translation():
    f = _file_id_filter({"file_id": "abc"})
    assert f.must[0].key == FILE_ID_PATH
    f2 = _file_id_filter({"file_id": {"$in": ["a", "b"]}})
    assert set(f2.must[0].match.any) == {"a", "b"}
    f3 = _file_id_filter(["x", "y"])
    assert set(f3.must[0].match.any) == {"x", "y"}


def test_add_and_id_helpers():
    s = make_store()
    # LibreChat inserts every chunk of a file with ids=[file_id]*n
    returned = s.add_documents(_docs("file-A", 3, "kubernetes"), ids=["file-A"] * 3)
    assert returned == ["file-A", "file-A", "file-A"]
    s.add_documents(_docs("file-B", 2, "grocery"), ids=["file-B"] * 2)

    # get_all_ids -> distinct file_ids
    assert sorted(s.get_all_ids()) == ["file-A", "file-B"]
    # get_filtered_ids -> subset that exists
    assert s.get_filtered_ids(["file-A", "missing"]) == ["file-A"]
    assert s.get_filtered_ids([]) == []
    # get_documents_by_ids -> all chunks for those file_ids
    docs = s.get_documents_by_ids(["file-A"])
    assert len(docs) == 3
    assert all(d.metadata["file_id"] == "file-A" for d in docs)
    assert all(d.page_content for d in docs)


def test_similarity_search_with_file_id_filter():
    s = make_store()
    s.add_documents(_docs("file-A", 2, "kubernetes"), ids=["file-A"] * 2)
    s.add_documents(_docs("file-B", 2, "kubernetes"), ids=["file-B"] * 2)

    emb = FakeEmbeddings().embed_query("kubernetes")
    # route call form: filter={"file_id": body.file_id}
    res = s.similarity_search_with_score_by_vector(emb, k=10, filter={"file_id": "file-A"})
    assert len(res) == 2
    assert all(doc.metadata["file_id"] == "file-A" for doc, _ in res)
    assert all(isinstance(score, float) for _, score in res)

    # multi: filter={"file_id": {"$in": [...]}}
    res2 = s.similarity_search_with_score_by_vector(
        emb, k=10, filter={"file_id": {"$in": ["file-A", "file-B"]}}
    )
    assert len(res2) == 4


def test_delete_by_file_id():
    s = make_store()
    s.add_documents(_docs("file-A", 3, "alpha"), ids=["file-A"] * 3)
    s.add_documents(_docs("file-B", 2, "beta"), ids=["file-B"] * 2)
    assert sorted(s.get_all_ids()) == ["file-A", "file-B"]

    s.delete(ids=["file-A"])
    assert s.get_all_ids() == ["file-B"]
    assert s.get_documents_by_ids(["file-A"]) == []
    # deleting nothing is a no-op
    s.delete(ids=[])
    assert s.get_all_ids() == ["file-B"]
