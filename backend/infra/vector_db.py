import os
import logging
import time
import shutil
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import NamedVector, ScoredPoint
from backend.config.settings import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_DIMENSION, PROCESSED_PATH

logger = logging.getLogger(__name__)


# Qdrant connection check and fallback
def is_qdrant_available():
    try:
        test_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        test_client.get_collections()
        return True
    except Exception as e:
        logger.warning(f"Qdrant unavailable: {e}")
        return False

QDRANT_AVAILABLE = is_qdrant_available()
if QDRANT_AVAILABLE:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = None

COLLECTION_NAME = "documents"

def get_client() -> QdrantClient:
    if not QDRANT_AVAILABLE:
        raise RuntimeError("Qdrant is not available.")
    return client

def ensure_collection_exists():
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant unavailable: skipping collection check.")
        return
    try:
        client = get_client()
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
        for field, schema in [("source", "text"), ("session_id", "keyword"), ("created_at", "integer")]:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=schema,
                )
                logger.info(f"Created payload index for '{field}'")
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to ensure collection exists: {e}")
        raise

def upsert_vectors(points: List[dict], session_id: str = "default"):
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant unavailable: skipping upsert.")
        return
    ensure_collection_exists()
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload={
                        "source": p["payload"].get("display_name") or p["payload"].get("source") or p["payload"].get("file_name", "Unknown"),
                        "page": p["payload"].get("page", "N/A"),
                        "text": p["payload"].get("page_content") or p["payload"].get("text", ""),
                        "session_id": session_id,
                        "created_at": time.time(),
                        **{k: v for k, v in p["payload"].items() if k not in ["source", "page", "text", "page_content", "session_id", "created_at"]}
                    }
                ) for p in points
            ]
        )
        logger.info(f"Successfully upserted {len(points)} points to Qdrant for session {session_id}")
    except Exception as exc:
        logger.error(f"Qdrant upsert failed for session {session_id}: {exc}")
        raise exc

def search_vectors(query_vector: List[float], limit: int = 5, session_id: Optional[str] = None) -> List[dict]:
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant unavailable: returning empty search result.")
        return []
    ensure_collection_exists()
    search_filter = None
    if session_id:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            ]
        )
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        ).points
        logger.info(f"Qdrant search for session {session_id} returned {len(results)} results")
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
                "content": r.payload.get("text") or r.payload.get("page_content", "")
            } for r in results
        ]
    except Exception as exc:
        logger.error(f"Qdrant search failed for session {session_id}: {exc}")
        return []

def delete_vectors_by_doc_id(doc_id: str):
    """Deletes all vector points associated with a specific document ID."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Successfully deleted vectors for doc_id {doc_id} from Qdrant")
    except Exception as exc:
        logger.error(f"Qdrant deletion failed for doc_id {doc_id}: {exc}")
        raise exc

def is_indexed_in_qdrant(file_name: str, session_id: str = "default") -> bool:
    """Check if a document is already indexed for this session."""
    ensure_collection_exists()
    try:
        client = get_client()
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchText(text=file_name),
                    ),
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )[0]
        return len(results) > 0
    except Exception as e:
        logger.warning(f"is_indexed_in_qdrant check failed for {file_name} in session {session_id}: {e}")
        return False

def get_session_document_count(session_id: str) -> int:
    """Returns the unique document count for a specific session by querying Qdrant."""
    ensure_collection_exists()
    try:
        # Use scroll to find unique sources for this session
        # For small limits (default 10, here we use 1000 since max is 5), this is efficient enough
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )[0]
        unique_files = {r.payload.get("source") for r in results if r.payload.get("source")}
        return len(unique_files)
    except Exception as e:
        logger.warning(f"Could not count session documents for {session_id}: {e}")
        return 0

def delete_session_vectors(session_id: str):
    """Delete all vectors belonging to a session and clean up local scratch files."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted vectors for session {session_id} from Qdrant")
        
        # Cleanup local scratch files for this session
        session_scratch_dir = os.path.join(PROCESSED_PATH, session_id)
        if os.path.exists(session_scratch_dir):
            shutil.rmtree(session_scratch_dir)
            logger.info(f"Deleted local scratch files for session {session_id}")
            
    except Exception as exc:
        logger.error(f"Failed to delete session vectors for {session_id}: {exc}")
        raise exc

def delete_vectors_older_than(cutoff_timestamp: float):
    """Purge vectors older than a specific timestamp for TTL cleanup."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="created_at",
                            range=models.Range(lt=cutoff_timestamp)
                        )
                    ]
                )
            )
        )
        logger.info(f"Purged stale vectors older than {cutoff_timestamp}")
    except Exception as exc:
        logger.error(f"Failed to purge stale vectors: {exc}")

def get_collection_count() -> int:
    """Returns the total number of points in the Qdrant collection."""
    ensure_collection_exists()
    try:
        res = client.get_collection(COLLECTION_NAME)
        return res.points_count
    except Exception as exc:
        logger.error(f"Failed to get Qdrant collection count: {exc}")
        return 0

def reset_collection():
    """Deletes and recreates the Qdrant collection."""
    try:
        logger.info(f"Deleting collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)
        ensure_collection_exists()
        logger.info(f"Collection '{COLLECTION_NAME}' reset successfully")
    except Exception as exc:
        logger.error(f"Failed to reset Qdrant collection: {exc}")
        # If it doesn't exist, ensure_collection_exists will create it anyway
        ensure_collection_exists()

# Removed top-level init call to prevent startup blocking.
# Operations will lazy-initialize the collection as needed.
