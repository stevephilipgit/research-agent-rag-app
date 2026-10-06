"""Application-level exceptions shared across layers.

These types cross module boundaries (infrastructure raises them, the API
boundary maps them to HTTP), so they live in `core` where every layer may
depend on them without inverting the dependency direction.
"""


class VectorServiceUnavailable(RuntimeError):
    """Raised when the vector store is known to be unavailable.

    Callers at the API boundary translate this into HTTP 503 so an
    unavailable vector store is never reported as an empty, successful
    retrieval.
    """


VECTOR_UNAVAILABLE_DETAIL = "Vector search service is temporarily unavailable"
