import hashlib
import math
import re
from typing import List, Optional, Callable
from langchain_core.embeddings import Embeddings
from config.settings import EMBEDDING_DIMENSION

class LocalHashEmbeddings(Embeddings):
    """Deterministic offline embeddings so ingestion works without network access."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

_embeddings: Optional[LocalHashEmbeddings] = None

def get_embeddings(callback: Optional[Callable[[str], None]] = None) -> LocalHashEmbeddings:
    global _embeddings
    if _embeddings is None:
        if callback:
            callback("Initializing local hash embeddings...")
        _embeddings = LocalHashEmbeddings()
    return _embeddings
