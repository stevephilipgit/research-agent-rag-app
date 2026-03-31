import logging
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import tool

from core.rag import retrieve_context_with_extensions
from services.tools import advanced_tools

logger = logging.getLogger(__name__)

# -------------------------------
# DOCUMENT SEARCH TOOL (RAG)
# -------------------------------
@tool
def document_search(query: str) -> str:
    """
    Search the uploaded documents knowledge base for relevant information.
    Use when the user asks about content from uploaded PDFs or text files.
    """
    if not query or not query.strip():
        return "No query provided."

    logger.info(f"[TOOL] document_search called | query: '{query[:80]}'")
    try:
        result = retrieve_context_with_extensions(query)
        logger.info(f"[TOOL] document_search returned {len(result) if result else 0} chars")
        return str(result) if result else "No relevant content found in documents."
    except Exception as e:
        logger.error(f"Error in document_search: {str(e)}", exc_info=True)
        return "An error occurred while searching documents."


# -------------------------------
# READ URL TOOL
# -------------------------------
@tool
def read_url(url: str) -> str:
    """
    Fetch and read the full text content of a webpage URL.
    """
    if not url or not url.strip():
        return "No URL provided."

    url = url.strip()

    # Basic sanity check — must look like a URL
    if not url.startswith(("http://", "https://")):
        return "Invalid URL. Must start with http:// or https://"

    logger.info(f"[TOOL] read_url called | url: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()

        if not docs:
            return "No content found at the URL."

        content = docs[0].page_content
        if not content or not content.strip():
            return "Page loaded but contained no readable text."

        return content.strip()[:3000]

    except Exception as e:
        logger.error(f"[TOOL] read_url failed for {url}: {str(e)}", exc_info=True)
        return f"Failed to read URL: {str(e)}"


# -------------------------------
# SUMMARIZE TEXT TOOL
# -------------------------------
@tool
def summarize_text(text: str) -> str:
    """
    Summarize the given text into concise key points.
    """
    if not text or not text.strip():
        return "No text provided to summarize."

    logger.info("[TOOL] summarize_text called")

    try:
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return "Could not extract sentences from the provided text."

        summary = " ".join(sentences[:4])
        if len(sentences) > 4:
            summary += "..."

        return summary

    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}", exc_info=True)
        return "An error occurred while summarizing text."


# -------------------------------
# TOOL LIST (ONLY VALID TOOLS)
# -------------------------------
all_tools = [
    document_search,
    read_url,
    summarize_text,
] + advanced_tools
