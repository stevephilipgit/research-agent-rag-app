import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from config.settings import SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY

logger = logging.getLogger(__name__)

_supabase_client = None

def _get_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = SUPABASE_URL
    key = SUPABASE_SERVICE_ROLE_KEY
    if not key:
        logger.warning("AUDIT: DB - SUPABASE_SERVICE_ROLE_KEY missing. Falling back to SUPABASE_KEY.")
        key = SUPABASE_KEY

    if not url or not key:
        logger.error("AUDIT: DB - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment.")
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        logger.exception("AUDIT: DB - Supabase client initialization failed")
        return None


class DB:
    @staticmethod
    async def get_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return None
        def _call():
            try:
                res = client.table("documents").select("*").eq("file_hash", file_hash).execute()
                return res.data[0] if res.data else None
            except Exception as exc:
                logger.exception(f"AUDIT: DB - get_by_hash failed for {file_hash}")
                return None
        return await asyncio.to_thread(_call)

    @staticmethod
    async def insert(table: str, data: Dict[str, Any]) -> str:
        client = _get_client()
        if not client:
            raise Exception("DB not available")
        logger.info(f"AUDIT: DB - Inserting into {table} | Data: {data}")
        def _call():
            try:
                res = client.table(table).insert(data).execute()
                if not res.data:
                    # Check if error is hidden in response
                    if hasattr(res, "error") and res.error:
                        logger.error(f"AUDIT: DB - Insert error in response: {res.error}")
                    raise Exception(f"Insert into {table} failed - no data returned. RLS Policy likely blocked it.")
                return res.data[0]["id"]
            except Exception as exc:
                error_str = str(exc)
                if "Could not find the" in error_str and "column" in error_str:
                    logger.error(f"AUDIT: Schema Error | Table: {table} | Operation: INSERT | Payload Keys: {list(data.keys())} | Error: {error_str} | Hint: Run schema migrations to sync backend.")
                else:
                    logger.exception(f"AUDIT: DB - Insert into {table} failed | Payload Keys: {list(data.keys())}")
                raise exc
        return await asyncio.to_thread(_call)

    @staticmethod
    async def update(table: str, doc_id: str, data: Dict[str, Any]):
        client = _get_client()
        if not client:
            raise Exception("DB not available")
        logger.info(f"AUDIT: DB - Updating {table} | ID: {doc_id} | Data: {data}")
        def _call():
            try:
                client.table(table).update(data).eq("id", doc_id).execute()
            except Exception as exc:
                logger.exception(f"AUDIT: DB - Update of {table} ID {doc_id} failed")
                raise exc
        await asyncio.to_thread(_call)

    @staticmethod
    async def delete(table: str, doc_id: str):
        client = _get_client()
        if not client:
            raise Exception("DB not available")
        logger.info(f"AUDIT: DB - Deleting from {table} | ID: {doc_id}")
        def _call():
            try:
                client.table(table).delete().eq("id", doc_id).execute()
            except Exception as exc:
                logger.exception(f"AUDIT: DB - Delete from {table} ID {doc_id} failed")
                raise exc
        await asyncio.to_thread(_call)

    @staticmethod
    async def query_documents(user_id: str) -> List[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return []
        def _call():
            try:
                res = client.table("documents").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
                return res.data
            except Exception as exc:
                logger.exception(f"AUDIT: DB - query_documents failed for user {user_id}")
                return []
        return await asyncio.to_thread(_call)
    
    @staticmethod
    async def get_document(doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return None
        def _call():
            try:
                res = client.table("documents").select("*").eq("id", doc_id).eq("user_id", user_id).execute()
                return res.data[0] if res.data else None
            except Exception as exc:
                logger.exception(f"AUDIT: DB - get_document failed for ID {doc_id}")
                return None
        return await asyncio.to_thread(_call)

    @staticmethod
    async def get_all_document_ids() -> List[str]:
        client = _get_client()
        if not client:
            return []
        def _call():
            try:
                res = client.table("documents").select("id").execute()
                return [d["id"] for d in res.data]
            except Exception as exc:
                logger.exception("AUDIT: DB - get_all_document_ids failed")
                return []
        return await asyncio.to_thread(_call)


# Backwards compatibility wrappers
def load_registry() -> List[dict]:
    client = _get_client()
    if not client:
        return []
    try:
        res = client.table("documents").select("*").execute()
        return res.data
    except Exception as exc:
        logger.exception("AUDIT: DB - load_registry failed")
        return []

def save_doc_to_registry(entry: dict) -> None:
    client = _get_client()
    if not client:
        return
    try:
        client.table("documents").insert(entry).execute()
    except Exception as exc:
        logger.exception("AUDIT: DB - save_doc_to_registry failed")

def remove_from_registry(doc_id: str):
    client = _get_client()
    if not client:
        return
    try:
        client.table("documents").delete().eq("id", doc_id).execute()
    except Exception as exc:
        logger.exception(f"AUDIT: DB - remove_from_registry failed for {doc_id}")

db = DB()
