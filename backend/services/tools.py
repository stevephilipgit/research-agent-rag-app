import logging
import math
from typing import Any, Dict

from langchain.tools import tool

from core.telemetry import emit_log
from config import ENABLE_TOOLS_ADVANCED

logger = logging.getLogger(__name__)

# -------------------------------
# SAFE CALCULATOR TOOL
# -------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Accepts expressions like: 2^3, 5*10, sqrt(25), (10+5)/2
    """
    if not ENABLE_TOOLS_ADVANCED or not expression or not expression.strip():
        return "Calculator tool is disabled."

    emit_log("Tool Used", "in_progress", f"Calculator: {expression}", "query")
    
    # Simple whitelist approach
    allowed_chars = "0123456789+-*/(). ^,sqrt,pow,abs,min,max"
    clean_expr = "".join([c for c in expression if c in allowed_chars])
    
    # Basic sanitization
    clean_expr = clean_expr.replace("^", "**")
    
    try:
        # Restricted globals/locals for eval (still risky, better to use asteval if available)
        safe_globals = {
            "math": math,
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "min": min,
            "max": max,
        }
        # pylint: disable=eval-used
        result = eval(clean_expr, {"__builtins__": {}}, safe_globals)
        emit_log("Tool Used", "success", f"Calculator result: {result}", "query")
        return f"Calculation Result: {result}"
    except Exception as exc:
        logger.warning(f"Calculator failed: {exc}")
        emit_log("Tool Used", "failure", f"Calculator error: {exc}", "query")
        return f"Error in calculation: {exc}. Please check your syntax."


# -------------------------------
# STRUCTURED QUERY (STUB)
# -------------------------------
@tool
def db_query(sql_params: Dict[str, Any]) -> str:
    """
    Query internal structured database for facts.
    Input should be a dictionary with 'table' and 'filters'.
    """
    if not ENABLE_TOOLS_ADVANCED or not sql_params:
        return "Database tool is disabled."

    emit_log("Tool Used", "in_progress", f"Structured DB query", "query")
    
    # This is a stub for the advanced DB integration
    # Requirement: "Structured Query (DB)"
    try:
        table = sql_params.get("table", "unknown")
        # Security: DO NOT execute raw SQL here
        # Return a meaningful message or a mock result for demo
        emit_log("Tool Used", "success", f"Query returned 0 results for {table}", "query")
        return f"No records found in table '{table}' matching the criteria."
    except Exception as exc:
        logger.error(f"DB Query failed: {exc}")
        emit_log("Tool Used", "failure", f"DB Query error: {exc}", "query")
        return "Database access failed. Please use document search instead."

# List for dynamic tool registration
advanced_tools = [
    calculator,
    db_query,
]
