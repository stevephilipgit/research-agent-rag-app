import logging
import operator
import re
from typing import Annotated, Iterator, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from utils.cache_db import get_cached_response, set_cached_response
try:
    from core.telemetry import emit_log
except ImportError:
    def emit_log(step: str, status: str = "success", detail: str = "", scope: str = "system"):
        pass

from core.tools import all_tools
from config.settings import (
    ENABLE_CACHE,
    ENABLE_MEMORY,
    ENABLE_SECURITY,
    ENABLE_VALIDATION,
    MAX_CONTEXT_CHARS,
    ENABLE_TOOL_GUARD,
    ENABLE_RETRY,
)
from services.security import sanitize_input, validate_input, detect_injection
from services.memory import save_memory, get_memory, build_prompt_with_memory
from services.grounding_validator import validate_answer
from services.tool_guard import is_tool_allowed
from utils.cache import get_cache, set_cache
from utils.retry import retry_call
from utils.sanitize import safe_llm_call, safe_tool_call
from utils.streaming import safe_stream

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


SYSTEM_PROMPT = """You are a document-based RAG assistant.

Your job:
1. When a user asks a question, call 'document_search' to retrieve information FROM UPLOADED DOCUMENTS ONLY.
2. After the tool returns results, READ them carefully. They are your ONLY source of truth.
3. Synthesize a clear answer ONLY from these results.
4. Always cite your sources: [filename, page X]
5. If 'document_search' returns no relevant content, say: "Answer not found in uploaded documents."

IMPORTANT: Never use external knowledge or invent facts. If the information isn't in the context, say it's not found.
"""

SYNTHESIS_PROMPT = """You are a document-based assistant.

Rules:
- Answer ONLY from the provided context
- Include all relevant details
- Be clear and concise
- Preserve source grounding
"""

from config.llm import get_llm
llm = get_llm()
llm_with_tools = llm.bind_tools(all_tools)
tool_node = ToolNode(tools=all_tools)


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def agent_node(state: AgentState):
    messages = state["messages"]

    if not any(isinstance(message, SystemMessage) for message in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    # Fix 6: Timeout Safety (10s)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(safe_llm_call, llm_with_tools, messages, retries=1)
        try:
            response = future.result(timeout=10)
        except TimeoutError:
            emit_log("LLM", "failure", "Request timed out after 10s", "query")
            return {"messages": [AIMessage(content="Request timed out. Please try again.")]}
        except Exception as e:
            emit_log("LLM", "failure", f"LLM error: {str(e)}", "query")
            return {"messages": [AIMessage(content="LLM execution failed.")]}

    if not response or not getattr(response, "content", None) and not getattr(response, "tool_calls", None):
         return {"messages": [AIMessage(content="LLM execution failed.")]}
    return {"messages": [response]}


def _parse_citations(tool_content: str) -> list:
    citations = []
    if not tool_content:
        return citations

    seen = set()
    rag_pattern = re.compile(
        r"\[Source:\s*(?P<source>[^\],]+),\s*Page:\s*(?P<page>[^\]]+)\]",
        re.IGNORECASE,
    )
    for match in rag_pattern.finditer(tool_content):
        source = match.group("source").strip()
        page = match.group("page").strip()
        doc_name = source.replace("\\", "/").split("/")[-1]
        key = (doc_name, page)
        if key not in seen:
            seen.add(key)
            citations.append({"type": "document", "document": doc_name, "page": page})

    url_pattern = re.compile(r"https?://[^\s\)\]\"\']+")
    for url in url_pattern.findall(tool_content):
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            citations.append({"type": "web", "label": url})

    return citations


def clean_context(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.strip()


def is_valid_answer(answer: str, context: str) -> bool:
    """Checks if the answer has meaningful overlap with the source context."""
    if not answer or len(answer) < 10:
        return False
    if not context:
        return False
    
    # Simple word overlap check
    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_words = set(re.findall(r"\w+", context.lower()))
    
    # Filter common stop words for better validation
    stop_words = {"the", "a", "an", "is", "are", "and", "or", "in", "on", "at", "to", "for", "with", "this", "that"}
    overlap = (answer_words & context_words) - stop_words
    
    return len(overlap) > 3


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for citation in citations:
        key = citation.get("document", "") + citation.get("page", "") + citation.get("label", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _collect_agent_execution(query: str, history_messages=None) -> dict:
    agent = build_agent()

    if history_messages is None:
        history_messages = []

    initial_state = {"messages": [*history_messages, HumanMessage(content=query)]}

    tool_context = ""
    rich_steps = []
    citations = []
    direct_answer = ""

    for step in agent.stream(initial_state):
        if "tools" in step:
            for message in step["tools"]["messages"]:
                if not isinstance(message, ToolMessage):
                    continue

                tool_name = getattr(message, "name", "tool")
                if ENABLE_TOOL_GUARD and not is_tool_allowed(tool_name):
                    emit_log("Tool Guard", "failure", f"Blocked unauthorized tool: {tool_name}", "query")
                    continue

                raw_content = str(message.content or "")
                tool_name = getattr(message, "name", "tool")
                if raw_content:
                    tool_context = f"{tool_context}\n\n{raw_content}".strip()

                citations.extend(_parse_citations(raw_content))
                preview = clean_context(raw_content)[:400]
                rich_steps.append(
                    {
                        "type": "tool_result",
                        "label": f"Result from: {tool_name}",
                        "detail": preview,
                    }
                )

        if "agent" in step:
            for message in step["agent"]["messages"]:
                if not isinstance(message, AIMessage):
                    continue

                tool_calls = bool(getattr(message, "tool_calls", None))
                raw_content = message.content
                content = raw_content.strip() if isinstance(raw_content, str) else str(raw_content or "").strip()

                if tool_calls:
                    for tool_call in message.tool_calls:
                        name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                        args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                        query_sent = args.get("query") or args.get("url") or args.get("text") or str(args)
                        rich_steps.append(
                            {
                                "type": "tool_call",
                                "label": f"Calling tool: {name}",
                                "detail": f"Input: {str(query_sent)[:200]}",
                            }
                        )
                        emit_log("Agent Execution", "in_progress", f"Calling tool: {name}", "query")
                elif content and len(content) > 10:
                    direct_answer = content
                    rich_steps.append(
                        {
                            "type": "answer",
                            "label": "Final answer generated",
                            "detail": content[:300],
                        }
                    )

    return {
        "tool_context": tool_context,
        "steps": rich_steps,
        "citations": _dedupe_citations(citations),
        "direct_answer": direct_answer,
    }


def _synthesis_messages(query: str, tool_context: str) -> list[BaseMessage]:
    return [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(
            content=f"""
Context:
{tool_context}

Question:
{query}

Answer clearly.
""".strip()
        ),
    ]


def run_research_agent(query: str, history_messages=None, session_id: str = "default"):
    # [NEW] Input Validation
    if ENABLE_SECURITY:
        if not validate_input(query):
            return {"answer": "Invalid query", "steps": [], "citations": []}
        if detect_injection(query):
            return {"answer": "Unsafe query detected", "steps": [], "citations": []}

    # [NEW] Cache check
    if ENABLE_CACHE:
        cached = get_cache(query)
        if cached:
            return cached

    cached_static = get_cached_response(query)
    if cached_static:
        return {
            "answer": cached_static,
            "steps": [{"type": "answer", "label": "Cache hit", "detail": "Served from response cache"}],
            "citations": [],
        }

    try:
        collected = _collect_agent_execution(query, history_messages)
    except Exception as exc:
        logger.error(f"Agent stream error: {exc}", exc_info=True)
        emit_log("Agent Execution", "failure", f"Agent stream error: {exc}", "query")
        return {
            "answer": "Something went wrong. Please try again.",
            "steps": [],
            "citations": [],
        }

    tool_context = collected["tool_context"]
    
    # [NEW] Limit context size to 4000 chars
    if len(tool_context) > 4000:
        tool_context = tool_context[:4000]
        
    final_answer = collected["direct_answer"]

    if tool_context:
        try:
            # [NEW] Memory Injection
            synthesis_prompt = _synthesis_messages(query, tool_context)
            if ENABLE_MEMORY:
                enriched_content = build_prompt_with_memory(query, tool_context, session_id)
                synthesis_prompt[-1].content = enriched_content

            if ENABLE_RETRY:
                final_response = safe_llm_call(llm, synthesis_prompt, retries=2)
            else:
                final_response = safe_llm_call(llm, synthesis_prompt, retries=1)
                
            final_answer = str(getattr(final_response, "content", "") or "").strip()
            
            # Fix 8: Safe Response Validation
            if not is_valid_answer(final_answer, tool_context):
                emit_log("Grounding", "failure", "Answer failed overlap validation", "query")
                final_answer = "Answer not found in uploaded documents."

            # [NEW] Grounding Validator (optional/extra check)
            if ENABLE_VALIDATION:
                # Create dummy doc list for validator
                from langchain_core.documents import Document as LC_Document
                v_docs = [LC_Document(page_content=tool_context)]
                if not validate_answer(final_answer, v_docs):
                    final_answer = "The generated answer could not be verified against the provided documents. [Grounding check failed]"

        except Exception as exc:
            logger.error(f"Synthesis failed: {exc}", exc_info=True)
            emit_log("Agent Execution", "failure", f"Synthesis failed: {exc}", "query")

    if not final_answer or len(final_answer.strip()) < 10:
        final_answer = "Answer not found in uploaded documents."

    # [NEW] Save Memory
    if ENABLE_MEMORY:
        save_memory(session_id, query, final_answer)

    set_cached_response(query, final_answer)
    emit_log(
        "Agent Execution",
        "success",
        f"run_research_agent done | steps={len(collected['steps'])} | citations={len(collected['citations'])}",
        "query",
    )

    # [NEW] Cache result
    res = {
        "answer": final_answer,
        "steps": collected["steps"],
        "citations": collected["citations"],
    }
    if ENABLE_CACHE:
        set_cache(query, res)

    return res


def run_research_agent_stream(query: str, history_messages=None, session_id: str = "default") -> Iterator[dict]:
    # [NEW] Security Layer
    if ENABLE_SECURITY:
        if not validate_input(query):
            err = "Invalid query"
            yield {"type": "token", "data": err}
            yield {"type": "done", "data": {"answer": err, "steps": [], "citations": []}}
            return
        if detect_injection(query):
            err = "Unsafe query detected"
            yield {"type": "token", "data": err}
            yield {"type": "done", "data": {"answer": err, "steps": [], "citations": []}}
            return

    # [NEW] Cache check
    if ENABLE_CACHE:
        cached = get_cache(query)
        if cached:
            yield {"type": "token", "data": cached.get("answer", "")}
            yield {"type": "done", "data": cached}
            return

    cached_static = get_cached_response(query)
    if cached_static:
        yield {"type": "token", "data": cached_static}
        yield {
            "type": "done",
            "data": {
                "answer": cached,
                "steps": [{"type": "answer", "label": "Cache hit", "detail": "Served from response cache"}],
                "citations": [],
            },
        }
        return

    try:
        collected = _collect_agent_execution(query, history_messages)
    except Exception as exc:
        logger.error(f"Agent stream error: {exc}", exc_info=True)
        emit_log("Agent Execution", "failure", f"Agent stream error: {exc}", "query")
        yield {"type": "error", "data": {"detail": "Something went wrong. Please try again."}}
        return

    tool_context = collected["tool_context"]
    
    # [NEW] Limit context size to 4000 chars
    if len(tool_context) > 4000:
        tool_context = tool_context[:4000]
        
    final_answer = collected["direct_answer"]
    streamed_chunks = []

    try:
        if tool_context:
            # [NEW] Memory Injection
            synthesis_prompt = _synthesis_messages(query, tool_context)
            if ENABLE_MEMORY:
                enriched_content = build_prompt_with_memory(query, tool_context, session_id)
                synthesis_prompt[-1].content = enriched_content

            emit_log("LLM", "in_progress", "Streaming started", "query")

            try:
                for chunk in safe_stream(llm, synthesis_prompt):
                    text = getattr(chunk, "content", "") or ""
                    if not text:
                        continue
                    streamed_chunks.append(text)
                    yield {"type": "token", "data": text}
            except Exception as stream_exc:
                logger.error(f"Streaming failed: {stream_exc}")
                yield {"type": "error", "data": {"detail": "Streaming failed. Please try again."}}
                return
            final_answer = "".join(streamed_chunks).strip()

            # Fix 8: Safe Response Validation for streaming
            if not is_valid_answer(final_answer, tool_context):
                emit_log("Grounding", "failure", "Streamed answer failed overlap validation", "query")
                # We can't really "undo" the stream, but we can append a warning or override the final_answer for the record
                final_answer = "Answer not found in uploaded documents."

            # [NEW] Grounding Validator
            if ENABLE_VALIDATION:
                from langchain_core.documents import Document as LC_Document
                v_docs = [LC_Document(page_content=tool_context)]
                if not validate_answer(final_answer, v_docs):
                    emit_log("Grounding", "failure", "Generated answer rejected by grounding validator.", "query")
                    final_answer = "Answer not found in uploaded documents. [Grounding check failed]"

        elif final_answer:
            # Check direct answer too
            if not is_valid_answer(final_answer, tool_context):
                 final_answer = "Answer not found in uploaded documents."
            for part in [final_answer]:
                yield {"type": "token", "data": part}
        else:
            fallback = "Answer not found in uploaded documents."
            yield {"type": "token", "data": fallback}
            final_answer = fallback
    except Exception as exc:
        logger.error(f"Streaming synthesis failed: {exc}", exc_info=True)
        emit_log("Agent Execution", "failure", f"Streaming synthesis failed: {exc}", "query")
        yield {"type": "error", "data": {"detail": "Something went wrong. Please try again."}}
        return

    if not final_answer or len(final_answer.strip()) < 1:
        final_answer = "The information is not available in the provided documents."

    # [NEW] Save Memory
    if ENABLE_MEMORY:
        save_memory(session_id, query, final_answer)

    set_cached_response(query, final_answer)
    emit_log("Agent Execution", "success", "Final answer generated", "query")

    final_res = {
        "answer": final_answer,
        "steps": collected["steps"],
        "citations": collected["citations"],
    }
    if ENABLE_CACHE:
        set_cache(query, final_res)

    yield {
        "type": "done",
        "data": final_res,
    }
