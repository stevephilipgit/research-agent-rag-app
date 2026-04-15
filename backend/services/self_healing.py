"""Self-Healing Flow Orchestrator

Wraps response generation with evaluation, decision, and adaptive retry.
Maintains separation from core logic - can be optionally enabled.
"""

import os
from typing import Callable, Tuple, Dict, Optional
from services.eval_engine import EvaluationEngine
from services.decision_engine import DecisionEngine
from services.strategy_manager import StrategyManager

# Module-level instances
eval_engine = EvaluationEngine()
decision_engine = DecisionEngine()
strategy_manager = StrategyManager()

MAX_RETRIES = 2
USE_LLM_EVAL = os.getenv("USE_LLM_EVAL", "false").lower() == "true"

# State management for adaptive AI features
_llm_client = None
_current_retrieval_params = {"top_k": 5}
_current_model = None
_previous_scores = []  # Track scores for early stop detection


def _get_llm_client():
    """Lazy load LLM client on first use."""
    global _llm_client
    if _llm_client is None:
        try:
            from config.llm import get_llm
            _llm_client = get_llm()
        except Exception as e:
            import logging
            logging.warning(f"Failed to load LLM client for evaluation: {e}. Falling back to heuristics.")
            _llm_client = None
    return _llm_client


def _create_llm_wrapper(llm_client) -> Optional[Callable[[str], str]]:
    """
    Create a wrapper function that converts string prompts to LLM calls.
    
    Args:
        llm_client: LangChain LLM client (ChatGroq, ChatOpenAI, etc.)
        
    Returns:
        Callable that takes a string prompt and returns a string response
    """
    if llm_client is None:
        return None
    
    def prompt_to_response(prompt: str) -> str:
        """Convert string prompt to LLM response."""
        try:
            from langchain_core.messages import HumanMessage
            
            # Call LLM with prompt
            message = HumanMessage(content=prompt)
            response = llm_client.invoke([message])
            
            # Extract text from response
            return str(getattr(response, "content", "") or "")
        except Exception as e:
            import logging
            logging.debug(f"LLM evaluation call failed: {e}")
            return ""
    
    return prompt_to_response


def set_retrieval_params(params: Dict) -> None:
    """Set retrieval parameters for adaptive retrieval (UPGRADE 2)."""
    global _current_retrieval_params
    _current_retrieval_params = params.copy()


def get_retrieval_params() -> Dict:
    """Get current retrieval parameters."""
    return _current_retrieval_params.copy()


def set_model(model_id: str) -> None:
    """Set model for adaptive model fallback (UPGRADE 3)."""
    global _current_model
    _current_model = model_id


def get_model() -> Optional[str]:
    """Get current model if set."""
    return _current_model


def reset_adaptive_state() -> None:
    """Reset adaptive state between queries."""
    global _current_retrieval_params, _current_model, _previous_scores
    _current_retrieval_params = {"top_k": 5}
    _current_model = None
    _previous_scores = []


def self_healing_flow(
    query: str,
    generate_fn: Callable[[str], str],
    context: str = ""
) -> Tuple[str, float, int]:
    """
    Execute self-healing flow with adaptive AI enhancements.

    Implements UPGRADES:
    - UPGRADE 1: Tracks all attempts → returns best (max score)
    - UPGRADE 2: Adapts retrieval (top_k ↑ if score < 0.5)
    - UPGRADE 3: Adapts model (fallback if score < 0.4)
    - UPGRADE 4: Optimizes evaluation (LLM for short/uncertain)
    - UPGRADE 6: Early stop (score ↓ twice)
    """
    global _previous_scores, _current_retrieval_params
    
    retries = 0
    current_query = query
    response: str = ""
    final_score: float = 0.0
    best_response: str = ""
    best_score: float = 0.0
    attempts_history: list = []
    consecutive_decreases = 0
    
    llm_wrapper = None
    if USE_LLM_EVAL:
        llm_client = _get_llm_client()
        llm_wrapper = _create_llm_wrapper(llm_client)

    local_retrieval_params = _current_retrieval_params.copy()

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = generate_fn(current_query)
        except Exception as e:
            import logging
            logging.error(f"Response generation failed on attempt {attempt}: {str(e)}")
            # If generation fails, use empty response to allow fallback
            if attempt > 0:
                continue  # Try again if not first attempt
            response = ""  # Fallback to empty response

        # UPGRADE 4: Evaluation optimization
        response_word_count = len(response.split())
        score_history_len = len(_previous_scores)
        recent_score = _previous_scores[-1] if score_history_len > 0 else 0.5
        uncertain_score = 0.4 <= recent_score <= 0.7
        use_llm_eval = USE_LLM_EVAL and (response_word_count < 50 or uncertain_score)
        
        if use_llm_eval and llm_wrapper is not None:
            scores = eval_engine.evaluate_with_fallback(
                current_query, response, context, llm_fn=llm_wrapper
            )
        else:
            scores = eval_engine.evaluate(current_query, response, context)
        
        final_score = eval_engine.final_score(scores)
        
        # UPGRADE 1: Track all attempts
        attempts_history.append((response, final_score))
        _previous_scores.append(final_score)
        
        if final_score > best_score:
            best_score = final_score
            best_response = response
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1

        # UPGRADE 6: Early stop if score ↓ twice
        if consecutive_decreases >= 2:
            break

        decision = decision_engine.decide(final_score)

        # UPGRADE 1: Return best response - don't exit early except for confirmed accept
        # Accept on attempt 1 is risky; require 2+ attempts to confirm
        if decision == "accept" and attempt >= 1:
            break

        if attempt < MAX_RETRIES:
            if decision == "improve":
                current_query = strategy_manager.improve_prompt(current_query)
            elif decision == "retry":
                current_query = strategy_manager.rewrite_query(current_query)
                
                # UPGRADE 2: Retrieval adaptation
                if final_score < 0.5:
                    local_retrieval_params = strategy_manager.expand_retrieval(
                        local_retrieval_params
                    )
                    set_retrieval_params(local_retrieval_params)
                
                # UPGRADE 3: Model fallback
                if final_score < 0.4:
                    set_model("fallback_model")

            retries += 1

    # UPGRADE 1: Return best response
    if attempts_history:
        best_response, best_score = max(
            attempts_history,
            key=lambda x: x[1] if isinstance(x, tuple) else 0.0
        )
        return best_response, best_score, retries
    
    return response, final_score, retries
