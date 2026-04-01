import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.vector_db import reset_collection

if __name__ == "__main__":
    print("Resetting Qdrant collection...")
    reset_collection()
    print("Done.")
