import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from infra.vector_db import client as qdrant_client, COLLECTION_NAME

def reset():
    print(f"Attempting to reset Qdrant collection: '{COLLECTION_NAME}'")
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if exists:
            qdrant_client.delete_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted successfully")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist, nothing to delete.")
            
        # Re-check to verify
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            print("Verification: Collection is GONE.")
        else:
            print("Verification FAILED: Collection still exists.")
            
    except Exception as e:
        print(f"Delete failed: {e}")

if __name__ == "__main__":
    reset()
