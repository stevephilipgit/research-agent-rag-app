import os
import sys
from pathlib import Path
import asyncio

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import logging
logging.basicConfig(level=logging.DEBUG)

from infra.storage import upload_file, get_file_url, delete_file, _get_client
from config.settings import BUCKET_NAME

async def main():
    print("=== ISOLATED SUPABASE UPLOAD TEST ===")
    
    # Check client
    client = _get_client()
    if client is None:
        print("ERROR: Could not initialize Supabase client.")
        return
        
    print(f"Client initialized. Target bucket: {BUCKET_NAME}")
    
    # Create mock files
    txt_content = b"Hello, this is a test text file."
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n"
    docx_content = b"PK\x03\x04\x14\x00\x00\x00\x00\x00" # minimal zip header
    
    files_to_test = [
        ("test_text.txt", txt_content),
        ("test_pdf.pdf", pdf_content),
        ("test_word.docx", docx_content)
    ]
    
    session_id = "test-session-isolated"
    
    for filename, content in files_to_test:
        print(f"\n--- Testing upload for: {filename} ---")
        try:
            storage_path = upload_file(content, filename, session_id=session_id)
            if storage_path:
                print(f"SUCCESS: Uploaded to {storage_path}")
                url = get_file_url(storage_path)
                print(f"URL: {url}")
                
                # Cleanup
                print(f"Cleaning up {storage_path}...")
                delete_file(storage_path)
            else:
                print(f"FAILED: upload_file returned empty path for {filename}")
        except Exception as e:
            print(f"EXCEPTION during upload: {e}")

if __name__ == "__main__":
    asyncio.run(main())
