import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from infra.vector_db import get_client, COLLECTION_NAME

def reset():
    print(f"Attempting to reset Qdrant collection: '{COLLECTION_NAME}'")
    try:
        client = get_client()
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if exists:
            client.delete_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted successfully")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist, nothing to delete.")
            
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            print("Verification: Collection is GONE.")
        else:
            print("Verification FAILED: Collection still exists.")
            
    except Exception as e:
        print(f"Delete failed: {e}")

if __name__ == "__main__":
    reset()
