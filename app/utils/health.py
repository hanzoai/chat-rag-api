# app/utils/health.py
from app.config import VECTOR_DB_TYPE, VectorDBType
from app.services.database import pg_health_check
from app.services.mongo_client import mongo_health_check


def qdrant_health_check() -> bool:
    """Liveness against hanzoai/vector (Qdrant): the shared vector_store client
    can list collections."""
    try:
        from app.config import vector_store

        vector_store.client.get_collections()
        return True
    except Exception:  # noqa: BLE001
        return False


async def is_health_ok() -> bool:
    # The route awaits this. pg/mongo checks are coroutines; the Qdrant check is
    # a sync bool. Await the async backends and return the sync one directly so
    # the result is a uniform awaitable bool (await on a plain bool raises
    # TypeError: "object bool can't be used in 'await' expression").
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        return await pg_health_check()
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return await mongo_health_check()
    if VECTOR_DB_TYPE == VectorDBType.QDRANT:
        return qdrant_health_check()
    return True