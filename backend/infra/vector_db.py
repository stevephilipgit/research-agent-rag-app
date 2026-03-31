import os
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config.settings import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Initialize Qdrant Client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "documents"

def init_collection():
    """Initializes the Qdrant collection for semantic search."""
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            logger.info(f"Creating collection '{COLLECTION_NAME}' in Qdrant")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION, 
                    distance=models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists in Qdrant")
    except Exception as exc:
        logger.error(f"Failed to initialize Qdrant collection: {exc}")
        # We don't raise here as it might be a temporary connection issue
        pass

def upsert_vectors(points: List[dict]):
    """Upserts a list of vector points into Qdrant."""
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p["payload"]
                ) for p in points
            ]
        )
        logger.info(f"Successfully upserted {len(points)} points to Qdrant")
    except Exception as exc:
        logger.error(f"Qdrant upsert failed: {exc}")
        raise exc

def search_vectors(query_vector: List[float], limit: int = 5) -> List[dict]:
    """Performs semantic search in Qdrant based on a query vector."""
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        logger.info(f"Qdrant search returned {len(results)} results")
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
                "content": r.payload.get("page_content", "")
            } for r in results
        ]
    except Exception as exc:
        logger.error(f"Qdrant search failed: {exc}")
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

def get_collection_count() -> int:
    """Returns the total number of points in the Qdrant collection."""
    try:
        res = client.get_collection(COLLECTION_NAME)
        return res.points_count
    except Exception as exc:
        logger.error(f"Failed to get Qdrant collection count: {exc}")
        return 0

# Optional initialization on module load
try:
    init_collection()
except:
    pass
