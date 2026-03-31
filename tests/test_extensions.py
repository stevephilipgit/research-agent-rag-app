from langchain.docstore.document import Document

from services.hybrid_retriever import _merge_results, hybrid_retrieve
from services.query_rewriter import rewrite_query
from utils.cache import get_embedding_cache, get_query_cache, set_embedding_cache, set_query_cache


def test_query_cache_round_trip():
    set_query_cache("capital of france", ["Paris"])
    assert get_query_cache("capital of france") == ["Paris"]


def test_embedding_cache_round_trip():
    vector = [0.1, 0.2, 0.3]
    set_embedding_cache("hybrid query", vector)
    assert get_embedding_cache("hybrid query") == vector


def test_merge_results_deduplicates_documents():
    doc_a = Document(page_content="Paris is the capital of France.", metadata={"source": "a"})
    doc_b = Document(page_content="Paris is the capital of France.", metadata={"source": "b"})
    doc_c = Document(page_content="The Eiffel Tower is in Paris.", metadata={"source": "c"})

    merged = _merge_results([doc_a, doc_c], [doc_b], top_k=2)

    assert len(merged) == 2
    assert merged[0].page_content == doc_a.page_content
    assert merged[1].page_content == doc_c.page_content


def test_hybrid_retrieve_uses_cache_when_present():
    cached_docs = [Document(page_content="Cached result", metadata={"source": "cache"})]
    set_query_cache("cached hybrid query", cached_docs)

    result = hybrid_retrieve("cached hybrid query", top_k=2)

    assert result == cached_docs


def test_query_rewriter_never_returns_empty_string():
    result = rewrite_query("Explain the Eiffel Tower")
    assert isinstance(result, str)
    assert result.strip()
