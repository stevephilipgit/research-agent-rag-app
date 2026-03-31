"""Shared structured telemetry for backend pipeline and UI log streaming."""

from collections import deque
from datetime import datetime
from queue import Empty, Queue
from threading import Lock
from typing import Deque, Dict, List

_entries: Deque[dict] = deque(maxlen=500)
_subscribers: set[Queue] = set()
_lock = Lock()
_next_id = 1


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def emit_log(step: str, status: str, detail: str = "", scope: str = "system") -> dict:
    global _next_id
    with _lock:
        entry = {
            "id": _next_id,
            "time": _timestamp(),
            "step": step,
            "status": status,
            "detail": detail,
            "scope": scope,
        }
        _next_id += 1
        _entries.append(entry)
        subscribers = list(_subscribers)

    for subscriber in subscribers:
        subscriber.put(entry)

    return entry


def get_logs(limit: int = 200) -> List[dict]:
    with _lock:
        return list(_entries)[-limit:]


def subscribe() -> Queue:
    queue: Queue = Queue()
    with _lock:
        _subscribers.add(queue)
    return queue


def unsubscribe(queue: Queue) -> None:
    with _lock:
        _subscribers.discard(queue)


def wait_for_log(queue: Queue, timeout: float = 15.0) -> Dict | None:
    try:
        return queue.get(timeout=timeout)
    except Empty:
        return None
