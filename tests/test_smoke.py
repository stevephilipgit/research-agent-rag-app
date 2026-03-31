import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.agent import run_research_agent
from src.config import GROQ_API_KEY, TAVILY_API_KEY, GROQ_MODEL
from src.document_loader import ingest_documents
from src.rag_pipeline import retrieve_context
from src.tools import document_search, summarize_text


class SmokeSkip(Exception):
    pass


def skip_if_env_issue(message: str):
    markers = [
        "Cannot send a request, as the client has been closed",
        "access a socket in a way forbidden by its access permissions",
        "Retrieval failed due to an internal error",
        "Connection error.",
    ]
    if any(marker in message for marker in markers):
        raise SmokeSkip(message)


def test_config():
    assert GROQ_API_KEY.startswith("gsk_"), "Invalid GROQ_API_KEY"
    assert TAVILY_API_KEY, "TAVILY_API_KEY is missing"
    assert GROQ_MODEL == "llama-3.1-8b-instant", "Incorrect GROQ_MODEL"
    print(f"Config OK - LLM: Groq/{GROQ_MODEL} | Embeddings: HuggingFace all-MiniLM-L6-v2")


def test_ingestion():
    os.makedirs("data/documents", exist_ok=True)
    with open("data/documents/test_smoke.txt", "w", encoding="utf-8") as f:
        f.write("The capital of France is Paris. The Eiffel Tower was built in 1889.")
    result = ingest_documents()
    if result["status"] != "success":
        skip_if_env_issue(result.get("message", ""))
    assert result["status"] == "success", "Ingestion failed"
    assert result["chunks_created"] > 0, "No chunks created"
    print(f"Ingestion OK - {result['chunks_created']} chunks (embedded via HuggingFace)")


def test_rag():
    result = retrieve_context("What is the capital of France?")
    skip_if_env_issue(result)
    assert "Paris" in result, "RAG retrieval failed"
    print("RAG Retrieval OK")


def test_document_search_tool():
    result = document_search("Eiffel Tower")
    assert len(result) > 0, "Document search tool failed"
    print("document_search tool OK")


def test_summarize_tool():
    result = summarize_text(
        "The Eiffel Tower is a wrought-iron lattice tower. It was designed by Gustave Eiffel. "
        "It stands 330 metres tall. It is located on the Champ de Mars in Paris."
    )
    assert len(result) > 10, "Summarize tool failed"
    print("summarize_text tool OK (via ChatGroq llama-3.1-8b-instant)")


def test_full_agent():
    result = run_research_agent("What is mentioned in the documents about France?")
    if not result.get("steps") and "not available" in result.get("answer", "").lower():
        raise SmokeSkip("Agent could not reach its external services in this environment.")
    assert result["answer"] and len(result["answer"]) > 20, "Agent failed to generate an answer"
    assert len(result.get("steps", [])) >= 1, "Agent did not take enough steps"
    print(
        f"Full agent OK - model=Groq/{GROQ_MODEL} | "
        f"steps={len(result.get('steps', []))} | answer_length={len(result['answer'])}"
    )


def test_cleanup():
    os.remove("data/documents/test_smoke.txt")
    shutil.rmtree("vector_store", ignore_errors=True)
    print("Cleanup OK")


if __name__ == "__main__":
    tests = [
        test_config,
        test_ingestion,
        test_rag,
        test_document_search_tool,
        test_summarize_tool,
        test_full_agent,
        test_cleanup,
    ]
    passed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
        except SmokeSkip as e:
            skipped += 1
            print(f"{t.__name__} SKIPPED: {e}")
        except Exception as e:
            print(f"{t.__name__} FAILED: {e}")
    print(f"\n{passed} passed, {skipped} skipped, {len(tests) - passed - skipped} failed.")
