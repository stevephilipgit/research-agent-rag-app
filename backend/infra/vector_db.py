import os
import logging
import time
import shutil
import builtins
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient, models
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from config.settings import (
    ENVIRONMENT,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_DIMENSION,
    PROCESSED_PATH,
)

logger = logging.getLogger(__name__)


_IS_DEVELOPMENT = (ENVIRONMENT or "development").strip().lower() == "development"
_IS_PRODUCTION = (ENVIRONMENT or "development").strip().lower() == "production"
_QDRANT_TIMEOUT_SECONDS = 60
_LOCAL_QDRANT_PATH = "./qdrant_local_storage"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _connect_cloud_qdrant() -> QdrantClient:
    logger.info("Attempting cloud Qdrant connection | url=%s | timeout=%ss", QDRANT_URL, _QDRANT_TIMEOUT_SECONDS)
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=_QDRANT_TIMEOUT_SECONDS,
    )
    client.get_collections()
    return client


def _make_client() -> QdrantClient:
    existing = getattr(builtins, "_RA_QDRANT_CLIENT_SINGLETON", None)
    if existing is not None:
        return existing

    if _IS_DEVELOPMENT:
        # Development mode always uses embedded Qdrant to preserve local/offline flow.
        logger.info("Using embedded Qdrant (development) | path=%s", _LOCAL_QDRANT_PATH)
        c = QdrantClient(path=_LOCAL_QDRANT_PATH, timeout=_QDRANT_TIMEOUT_SECONDS)
        setattr(builtins, "_RA_QDRANT_CLIENT_SINGLETON", c)
        return c

    if _IS_PRODUCTION:
        if not QDRANT_URL or not QDRANT_API_KEY:
            msg = "Production requires QDRANT_URL and QDRANT_API_KEY; refusing embedded fallback."
            logger.error(msg)
            raise RuntimeError(msg)
        try:
            c = _connect_cloud_qdrant()
            logger.info("Connected to cloud Qdrant | url=%s", QDRANT_URL)
            setattr(builtins, "_RA_QDRANT_CLIENT_SINGLETON", c)
            return c
        except Exception:
            logger.exception("Cloud Qdrant unavailable after retries")
            raise

    msg = f"Unsupported ENVIRONMENT={ENVIRONMENT!r}. Allowed values: development, production."
    logger.error(msg)
    raise RuntimeError(msg)


import threading

_client_lock = threading.Lock()
_client_instance: Optional[QdrantClient] = None

def get_client() -> QdrantClient:
    """Lazily initialize and return the Qdrant client. Thread-safe."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
        
    with _client_lock:
        if _client_instance is None:
            # Add retry logic for local storage locks (Task: Fix locking error)
            max_retries = 3
            last_exc = None
            for attempt in range(max_retries):
                try:
                    _client_instance = _make_client()
                    break
                except RuntimeError as e:
                    if "already accessed by another instance" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Qdrant storage locked (attempt {attempt+1}/{max_retries}). Waiting...")
                        time.sleep(1.5)
                        last_exc = e
                        continue
                    raise
                except Exception:
                    raise
            
            if _client_instance is None and last_exc:
                raise last_exc
                
    return _client_instance

# QDRANT_AVAILABLE is now a property or check
def is_qdrant_available():
    try:
        return get_client() is not None
    except Exception:
        return False

# For backward compatibility
client = None # Will be initialized via get_client() where used


COLLECTION_NAME = "documents"


def _keyword_filter(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any_filter(key: str, values: List[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def _build_query_filter(
    source_filters: Optional[List[str]] = None,
    topic_filters: Optional[List[str]] = None,
    document_type: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[models.Filter]:
    must = []

    if user_id:
        must.append(_keyword_filter("user_id", user_id))

    if source_filters:
        normalized_sources = [os.path.basename((s or "").strip()) for s in source_filters if s and s.strip()]
        if len(normalized_sources) == 1:
            must.append(_keyword_filter("source", normalized_sources[0]))
        elif len(normalized_sources) > 1:
            must.append(_match_any_filter("source", normalized_sources))

    if topic_filters:
        normalized_topics = [(t or "").strip().lower() for t in topic_filters if t and t.strip()]
        if len(normalized_topics) == 1:
            must.append(_keyword_filter("topic", normalized_topics[0]))
        elif len(normalized_topics) > 1:
            must.append(_match_any_filter("topic", normalized_topics))

    if document_type and document_type.strip():
        must.append(_keyword_filter("document_type", document_type.strip().lower()))

    if not must:
        return None
    return models.Filter(must=must)


def ensure_collection_exists():
    if not is_qdrant_available():
        logger.warning("Qdrant unavailable: skipping collection check.")
        return

    try:
        c = get_client()
        existing = [col.name for col in c.get_collections().collections]
        if COLLECTION_NAME not in existing:
            from qdrant_client.models import Distance, VectorParams

            c.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection | collection=%s | dimension=%s", COLLECTION_NAME, EMBEDDING_DIMENSION)
        else:
            logger.info("Qdrant collection exists | collection=%s", COLLECTION_NAME)

        indexes = [
            ("doc_id", "keyword"),
            ("session_id", "keyword"),
            ("user_id", "keyword"),
            ("source", "keyword"),
            ("topic", "keyword"),
            ("created_at", "integer"),
            ("document_type", "keyword"),
        ]
        for field, schema in indexes:
            try:
                c.create_payload_index(collection_name=COLLECTION_NAME, field_name=field, field_schema=schema)
                logger.info("Created payload index | collection=%s | field=%s", COLLECTION_NAME, field)
            except Exception as exc:
                logger.debug("Payload index already exists or failed | field=%s | reason=%s", field, exc)
    except Exception as e:
        logger.error(f"Failed to ensure collection exists: {e}")
        raise


def upsert_vectors(points: List[dict], session_id: str = "default"):
    if not is_qdrant_available():
        logger.warning("Qdrant unavailable: skipping upsert.")
        return

    ensure_collection_exists()
    now_ts = int(time.time())

    try:
        normalized_points = []
        for p in points:
            payload = dict(p.get("payload") or {})
            source = os.path.basename((payload.get("display_name") or payload.get("source") or payload.get("file_name") or "Unknown").strip())
            final_payload = {
                "source": source,
                "page": payload.get("page", payload.get("page_number", "N/A")),
                "chunk_index": payload.get("chunk_index", 0),
                "text": payload.get("page_content") or payload.get("text", ""),
                "session_id": payload.get("session_id") or session_id or "default",
                "user_id": payload.get("user_id") or session_id or "default",
                "document_type": payload.get("document_type", "general"),
                "topic": payload.get("topic", "general"),
                "created_at": payload.get("created_at", now_ts),
                "file_hash": payload.get("file_hash", ""),
                "doc_id": payload.get("doc_id", ""),
                "embedding_model": payload.get("embedding_model", "unknown"),
                **{k: v for k, v in payload.items() if k not in {"text", "page_content", "source", "display_name"}},
            }
            normalized_points.append(models.PointStruct(id=p["id"], vector=p["vector"], payload=final_payload))

        get_client().upsert(collection_name=COLLECTION_NAME, points=normalized_points)
        logger.info(
            "Vector upsert completed | collection=%s | session_id=%s | points=%d",
            COLLECTION_NAME,
            session_id,
            len(points),
        )
    except Exception as exc:
        logger.error(f"Qdrant upsert failed for session {session_id}: {exc}")
        raise


def search_vectors(
    query_vector: List[float],
    limit: int = 5,
    session_id: Optional[str] = None,
    source_filters: Optional[List[str]] = None,
    topic_filters: Optional[List[str]] = None,
    document_type: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[dict]:
    if not is_qdrant_available():
        logger.warning("Qdrant unavailable: returning empty search result.")
        return []

    ensure_collection_exists()
    started = time.perf_counter()
    query_filter = _build_query_filter(
        source_filters=source_filters,
        topic_filters=topic_filters,
        document_type=document_type,
        user_id=user_id,
    )

    logger.info(
        "Retrieval filters | user_id=%s | sources=%s | topics=%s | document_type=%s",
        user_id,
        source_filters,
        topic_filters,
        document_type,
    )

    try:
        points = get_client().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sources = {p.payload.get("source", "unknown") for p in points if p.payload}
        logger.info(
            "Retrieval results | chunks=%d | unique_sources=%d | latency_ms=%.2f",
            len(points),
            len(sources),
            elapsed_ms,
        )

        return [
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload,
                "content": p.payload.get("text") or p.payload.get("page_content", ""),
            }
            for p in points
        ]
    except Exception as exc:
        logger.error(f"Qdrant search failed for session {session_id}: {exc}")
        return []


def delete_vectors_by_doc_id(doc_id: str):
    try:
        get_client().delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
            ),
        )
        logger.info(f"Successfully deleted vectors for doc_id {doc_id} from Qdrant")
    except Exception as exc:
        logger.error(f"Qdrant deletion failed for doc_id {doc_id}: {exc}")
        raise


def is_indexed_in_qdrant(file_name: str, session_id: str = "default") -> bool:
    ensure_collection_exists()
    try:
        c = get_client()
        source = os.path.basename((file_name or "").strip())
        results = c.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=session_id)),
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


def is_file_hash_indexed_in_qdrant(file_hash: str, session_id: str = "default") -> bool:
    if not file_hash:
        return False
    ensure_collection_exists()
    try:
        c = get_client()
        results = c.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash)),
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=session_id)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )[0]
        return len(results) > 0
    except Exception as e:
        logger.warning(f"is_file_hash_indexed_in_qdrant check failed for hash in session {session_id}: {e}")
        return False


def is_doc_id_indexed_in_qdrant(doc_id: str) -> bool:
    if not doc_id:
        return False
    ensure_collection_exists()
    try:
        c = get_client()
        results = c.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )[0]
        return len(results) > 0
    except Exception as e:
        logger.warning(f"is_doc_id_indexed_in_qdrant check failed for doc_id {doc_id}: {e}")
        return False


def get_session_document_count(session_id: str) -> int:
    ensure_collection_exists()
    try:
        results = get_client().scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=session_id))]),
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
    try:
        get_client().delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=session_id))])
            ),
        )
        logger.info(f"Deleted vectors for session {session_id} from Qdrant")

        session_scratch_dir = os.path.join(PROCESSED_PATH, session_id)
        if os.path.exists(session_scratch_dir):
            shutil.rmtree(session_scratch_dir)
            logger.info(f"Deleted local scratch files for session {session_id}")
    except Exception as exc:
        logger.error(f"Failed to delete session vectors for {session_id}: {exc}")
        raise


def delete_vectors_older_than(cutoff_timestamp: float):
    try:
        get_client().delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="created_at", range=models.Range(lt=cutoff_timestamp))]
                )
            ),
        )
        logger.info(f"Purged stale vectors older than {cutoff_timestamp}")
    except Exception as exc:
        logger.error(f"Failed to purge stale vectors: {exc}")


def get_collection_count() -> int:
    ensure_collection_exists()
    try:
        res = get_client().get_collection(COLLECTION_NAME)
        return res.points_count
    except Exception as exc:
        logger.error(f"Failed to get Qdrant collection count: {exc}")
        return 0


def reset_collection():
    try:
        logger.info(f"Deleting collection '{COLLECTION_NAME}'")
        get_client().delete_collection(COLLECTION_NAME)
        ensure_collection_exists()
        logger.info(f"Collection '{COLLECTION_NAME}' reset successfully")
    except Exception as exc:
        logger.error(f"Failed to reset Qdrant collection: {exc}")
        ensure_collection_exists()


async def cleanup_orphan_vectors():
    """Phase 5: Cleanup Qdrant vectors that have no matching document in Postgres"""
    if not is_qdrant_available():
        return
        
    from infra.db import db
    from qdrant_client.models import PointIdsList

    try:
        registered_ids = await db.get_all_document_ids()
        registered_set = {str(r) for r in registered_ids}

        orphans = []
        offset = None
        while True:
            res = get_client().scroll(
                collection_name=COLLECTION_NAME, 
                offset=offset, 
                limit=100, 
                with_payload=True,
                with_vectors=False
            )
            points = res[0]
            offset = res[1]
            
            for p in points:
                if p.payload and p.payload.get("doc_id"):
                    doc_id = p.payload.get("doc_id")
                    if doc_id not in registered_set:
                        orphans.append(p.id)
                        
            if offset is None:
                break

        if orphans:
            get_client().delete(COLLECTION_NAME, PointIdsList(points=orphans))
            logger.info(f"Cleaned up {len(orphans)} orphan vectors")
    except Exception as exc:
        logger.error(f"Failed to cleanup orphan vectors: {exc}")
