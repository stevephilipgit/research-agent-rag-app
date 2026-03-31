from backend.config import ENABLE_TOOL_GUARD

ALLOWED_TOOLS = ["calculator", "document_search", "read_url", "summarize_text"]

def is_tool_allowed(tool_name: str):
    if not ENABLE_TOOL_GUARD:
        return True
    return tool_name in ALLOWED_TOOLS
