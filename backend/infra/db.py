import os
import logging
import asyncio
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_supabase_client = None

def _get_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        logger.error(f"Supabase client initialization failed: {exc}")
        return None


class DB:
    @staticmethod
    async def get_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return None
        def _call():
            res = client.table("documents").select("*").eq("file_hash", file_hash).execute()
            return res.data[0] if res.data else None
        return await asyncio.to_thread(_call)

    @staticmethod
    async def insert(table: str, data: Dict[str, Any]) -> str:
        client = _get_client()
        if not client:
            raise Exception("DB not available")
        def _call():
            res = client.table(table).insert(data).execute()
            if not res.data:
                raise Exception("Insert failed")
            return res.data[0]["id"]
        return await asyncio.to_thread(_call)

    @staticmethod
    async def update(table: str, doc_id: str, data: Dict[str, Any]):
        client = _get_client()
        if not client:
            raise Exception("DB not available")
        def _call():
            client.table(table).update(data).eq("id", doc_id).execute()
        await asyncio.to_thread(_call)

    @staticmethod
    async def delete(table: str, doc_id: str):
        client = _get_client()
        if not client:
            raise Exception("DB not available")
        def _call():
            client.table(table).delete().eq("id", doc_id).execute()
        await asyncio.to_thread(_call)

    @staticmethod
    async def query_documents(user_id: str) -> List[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return []
        def _call():
            res = client.table("documents").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return res.data
        return await asyncio.to_thread(_call)
    
    @staticmethod
    async def get_document(doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return None
        def _call():
            res = client.table("documents").select("*").eq("id", doc_id).eq("user_id", user_id).execute()
            return res.data[0] if res.data else None
        return await asyncio.to_thread(_call)

    @staticmethod
    async def get_all_document_ids() -> List[str]:
        client = _get_client()
        if not client:
            return []
        def _call():
            res = client.table("documents").select("id").execute()
            return [d["id"] for d in res.data]
        return await asyncio.to_thread(_call)


# Backwards compatibility wrappers that use sync, where needed for other modules
def load_registry() -> List[dict]:
    client = _get_client()
    if not client:
        return []
    try:
        res = client.table("documents").select("*").execute()
        return res.data
    except Exception:
        return []

def save_doc_to_registry(entry: dict) -> None:
    client = _get_client()
    if not client:
        return
    client.table("documents").insert(entry).execute()

def remove_from_registry(doc_id: str):
    client = _get_client()
    if not client:
        return
    client.table("documents").delete().eq("id", doc_id).execute()

db = DB()
