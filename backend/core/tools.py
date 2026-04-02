import logging
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import tool

from core.rag import retrieve_context_with_extensions
from services.tools import advanced_tools

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# -------------------------------
# DOCUMENT SEARCH TOOL (RAG)
# -------------------------------
@tool
def document_search(query: str, config: RunnableConfig) -> str:
    """
    Search the uploaded documents knowledge base for relevant information.
    Use when the user asks about content from uploaded PDFs or text files.
    """
    if not query or not query.strip():
        return "No query provided."

    session_id = config.get("configurable", {}).get("session_id", "default")
    logger.info(f"[TOOL] document_search called | session: {session_id} | query: '{query[:80]}'")
    try:
        result = retrieve_context_with_extensions(query, session_id=session_id)
        logger.info(f"[TOOL] document_search returned {len(result) if result else 0} chars")
        return str(result) if result else "No relevant content found in documents."
    except Exception as e:
        logger.error(f"Error in document_search: {str(e)}", exc_info=True)
        return "An error occurred while searching documents."


# -------------------------------
# TOOL LIST (ONLY VALID TOOLS)
# -------------------------------
all_tools = [
    document_search
]
