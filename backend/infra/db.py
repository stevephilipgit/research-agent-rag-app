import os
import json
import logging
from typing import List
from config.settings import ROOT_DIR

DOC_REGISTRY_PATH = ROOT_DIR / "data" / "doc_registry.json"
logger = logging.getLogger(__name__)

def load_registry() -> List[dict]:
    DOC_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DOC_REGISTRY_PATH.exists():
        DOC_REGISTRY_PATH.write_text("[]", encoding="utf-8")
        return []
    try:
        return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_doc_to_registry(entry: dict) -> None:
    data = load_registry()
    data.append(entry)
    DOC_REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def remove_from_registry(doc_id: str):
    data = load_registry()
    data = [entry for entry in data if entry.get("doc_id") != doc_id]
    DOC_REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
