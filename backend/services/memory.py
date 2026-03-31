import logging
import time
from typing import List, Dict, Any, Tuple
from threading import Lock

from backend.config import ENABLE_MEMORY
from backend.core.telemetry import emit_log

logger = logging.getLogger(__name__)

_memory_store: Dict[str, List[Tuple[float, str, str]]] = {}
_lock = Lock()
DEFAULT_MEMORY_LIMIT = 5
MEMORY_TTL_HOURS = 24

def save_memory(session_id: str, query: str, answer: str) -> None:
    if not ENABLE_MEMORY or not session_id or not query:
        return

    with _lock:
        if session_id not in _memory_store:
            _memory_store[session_id] = []
        
        # Pruning by TTL and Limit
        now = time.time()
        _memory_store[session_id] = [
            (ts, q, a) for (ts, q, a) in _memory_store[session_id]
            if (now - ts) < (MEMORY_TTL_HOURS * 3600)
        ]
        
        _memory_store[session_id].append((now, query, answer))
        
        # Limit to last N
        if len(_memory_store[session_id]) > DEFAULT_MEMORY_LIMIT * 2:
            _memory_store[session_id] = _memory_store[session_id][-DEFAULT_MEMORY_LIMIT:]

    emit_log("Memory Injection", "success", f"Saved to session {session_id}", "query")

def get_memory(session_id: str, limit: int = DEFAULT_MEMORY_LIMIT) -> List[Tuple[float, str, str]]:
    if not ENABLE_MEMORY or not session_id:
        return []

    with _lock:
        memories = _memory_store.get(session_id, [])
        return memories[-limit:]

def build_prompt_with_memory(query: str, docs_formatted: str, session_id: str) -> str:
    """Format memory context to the Prompt."""
    if not ENABLE_MEMORY or not session_id:
        return f"Context:\n{docs_formatted}\n\nQuestion:\n{query}"

    memories = get_memory(session_id)
    if not memories:
        return f"Context:\n{docs_formatted}\n\nQuestion:\n{query}"

    memory_text = "\n".join([f"User: {q}\nAI: {a}" for (ts, q, a) in memories])
    emit_log("Memory Injection", "success", f"Injected {len(memories)} history messages", "query")

    return f"""Previous Conversation History:
{memory_text}

New Document Context:
{docs_formatted}

New Question:
{query}
"""
