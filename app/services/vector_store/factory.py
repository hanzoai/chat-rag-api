from typing import Optional
from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None,
    # qdrant-only options (hanzoai/vector)
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_dim: Optional[int] = None,
    qdrant_distance: str = "cosine",
    qdrant_prefer_grpc: bool = False,
):
    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    elif mode == "qdrant":
        # Imported lazily so pgvector/atlas deployments don't need qdrant deps.
        from .qdrant_vector import QdrantVector

        return QdrantVector.from_config(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            embeddings=embeddings,
            dim=qdrant_dim,
            distance=qdrant_distance,
            prefer_grpc=qdrant_prefer_grpc,
        )
    else:
        raise ValueError(
            "Invalid mode specified. Choose 'sync', 'async', 'atlas-mongo', or 'qdrant'."
        )