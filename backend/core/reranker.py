import logging
import os
import threading
import time

from core.telemetry import emit_log
from config.settings import ENABLE_RERANKER

logger = logging.getLogger(__name__)

_reranker_model = None
_reranker_lock = threading.Lock()
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_BATCH_SIZE = 16
RERANKER_MAX_CHARS = 512
RERANKER_PREFILTER_MULTIPLIER = 3
RERANKER_CPU_THREADS = int(os.getenv("RERANKER_CPU_THREADS", "4"))


def get_reranker_model():
    if not ENABLE_RERANKER:
        logger.info("Reranker disabled via ENABLE_RERANKER=false")
        return None
    global _reranker_model
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                try:
                    import torch
                    from sentence_transformers import CrossEncoder

                    # Authenticate with HuggingFace if token is available
                    hf_token = os.environ.get("HF_TOKEN")
                    if hf_token:
                        try:
                            from huggingface_hub import login
                            login(token=hf_token, add_to_git_credential=False)
                            logger.info("HuggingFace login successful")
                        except Exception as login_exc:
                            logger.warning(f"HuggingFace login failed: {login_exc}")

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cpu":
                        torch.set_num_threads(RERANKER_CPU_THREADS)

                    try:
                        _reranker_model = CrossEncoder(
                            RERANKER_MODEL_NAME,
                            device=device,
                            local_files_only=True,
                        )
                        logger.info(
                            "Reranker model loaded successfully from local cache on %s.",
                            device,
                        )
                        emit_log(
                            "Reranking",
                            "success",
                            f"Cross-encoder model loaded from local cache on {device}",
                            "query",
                        )
                    except Exception as cache_exc:
                        logger.info(
                            "Local reranker cache unavailable on %s, falling back to remote load: %s",
                            device,
                            cache_exc,
                        )
                        emit_log(
                            "Reranking",
                            "in_progress",
                            f"Local reranker cache unavailable on {device}, trying remote load",
                            "query",
                        )
                        _reranker_model = CrossEncoder(
                            RERANKER_MODEL_NAME,
                            device=device,
                        )
                        logger.info(
                            "Reranker model loaded successfully from remote source on %s.",
                            device,
                        )
                        emit_log(
                            "Reranking",
                            "success",
                            f"Cross-encoder model loaded from remote source on {device}",
                            "query",
                        )
                except Exception as exc:
                    logger.error(f"Failed to load reranker model: {exc}")
                    emit_log("Reranking", "failure", f"Cross-encoder unavailable: {exc}", "query")
                    _reranker_model = None
    return _reranker_model


def warmup_reranker() -> None:
    model = get_reranker_model()
    if model is None:
        return

    try:
        start = time.time()
        model.predict(
            [("test", "test")],
            batch_size=1,
            show_progress_bar=False,
        )
        logger.info("Reranker warmup completed in %.2fs", time.time() - start)
    except Exception as exc:
        logger.warning("[reranker] Warmup failed: %s", exc)


def is_good_chunk(text: str) -> bool:
    if not text:
        return False

    text = text.strip()
    if len(text) < 100:
        return False
    if text.count(",") > 8:
        return False
    if "." not in text:
        return False
    if len(text.split()) < 10:
        return False
    return True


def rerank(query: str, docs: list, top_k: int = 3) -> list:
    if not docs:
        emit_log("Reranking", "success", "No documents to rerank", "query")
        return []

    if not query or not query.strip():
        logger.warning("[reranker] Empty query passed - returning docs in original order.")
        emit_log("Reranking", "failure", "Empty query passed to reranker", "query")
        return docs[:top_k]

    good_docs = [doc for doc in docs if is_good_chunk(doc.page_content)]
    if not good_docs:
        logger.warning("[reranker] All docs failed quality filter - returning original docs unfiltered.")
        emit_log("Reranking", "failure", "All chunks failed quality filter", "query")
        return docs[:top_k]

    rerank_limit = max(top_k, top_k * RERANKER_PREFILTER_MULTIPLIER)
    candidate_docs = good_docs[:rerank_limit]

    model = get_reranker_model()
    if model is None:
        logger.warning("[reranker] Model unavailable - returning quality-filtered docs without reranking.")
        emit_log("Reranking", "success", "Model unavailable; using original ordering", "query")
        return candidate_docs[:top_k]

    try:
        emit_log("Reranking", "in_progress", f"Reranking {len(candidate_docs)} documents", "query")
        pairs = [
            (
                query,
                doc.page_content.replace("\n", " ").strip()[:RERANKER_MAX_CHARS],
            )
            for doc in candidate_docs
        ]
        start = time.time()
        scores = model.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
        logger.info(
            "[reranker] Reranking took %.2fs for %d documents on model %s",
            time.time() - start,
            len(candidate_docs),
            RERANKER_MODEL_NAME,
        )
        ranked_docs = sorted(zip(candidate_docs, scores), key=lambda item: item[1], reverse=True)
        emit_log(
            "Reranking",
            "success",
            f"Reranked to top {min(top_k, len(ranked_docs))} documents",
            "query",
        )
        return [doc for doc, _ in ranked_docs[:top_k]]
    except Exception as exc:
        logger.error(f"[reranker] Reranking failed: {exc}", exc_info=True)
        emit_log("Reranking", "failure", str(exc), "query")
        return candidate_docs[:top_k]
