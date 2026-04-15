"""
Regression test to ensure baseline system still works without self-healing.
"""

import sys
import os
from pathlib import Path

BACKEND_PATH = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# Disable self-healing
os.environ["ENABLE_SELF_HEALING"] = "false"
os.environ["USE_LLM_EVAL"] = "false"

from core.agent import run_research_agent
from models.schema import QueryRequest
from services.rag_service import query_agent
from langchain_core.messages import HumanMessage

def test_query_agent_baseline():
    """
    Regression test: query_agent should work without self-healing.
    This ensures we haven't broken the existing flow.
    """
    try:
        # Mock history
        history_messages = []
        
        # Call query_agent with self-healing disabled
        result = query_agent(query="What is artificial intelligence?", session_id="test-regression")
        
        # Verify response structure
        assert "answer" in result, "Missing 'answer' in response"
        assert "steps" in result, "Missing 'steps' in response"
        assert "citations" in result, "Missing 'citations' in response"
        assert "messages" in result, "Missing 'messages' in response"
        assert "logs" in result, "Missing 'logs' in response"
        assert "debug" in result, "Missing 'debug' in response"
        
        # Verify non-empty answer
        assert result["answer"], "Answer is empty"
        
        print("✓ REGRESSION TEST: Baseline query_agent works without self-healing")
        return True
        
    except Exception as e:
        print(f"✗ REGRESSION TEST FAILED: {e}")
        return False


def test_eval_engine_compatibility():
    """Regression: Evaluation engine should work independently."""
    from services.eval_engine import EvaluationEngine
    
    try:
        engine = EvaluationEngine()
        
        # Test heuristic evaluation (no LLM)
        scores = engine.evaluate(
            query="What is AI?",
            response="AI is artificial intelligence that enables machines to learn and perform tasks.",
            context="AI context document"
        )
        
        assert "relevance" in scores
        assert "grounding" in scores
        assert "completeness" in scores
        
        final = engine.final_score(scores)
        assert 0 <= final <= 1.0
        
        # Test fallback when LLM disabled
        scores2 = engine.evaluate_with_fallback(
            query="What is AI?",
            response="AI is artificial intelligence.",
            context="",
            llm_fn=None  # No LLM
        )
        
        assert "relevance" in scores2
        
        print("✓ REGRESSION TEST: Evaluation engine compatible")
        return True
        
    except Exception as e:
        print(f"✗ REGRESSION TEST FAILED: {e}")
        return False


def test_decision_threshold_logic():
    """Regression: Decision engine thresholds unchanged."""
    from services.decision_engine import DecisionEngine
    
    try:
        engine = DecisionEngine(accept_threshold=0.75, improve_threshold=0.5)
        
        # Test exact thresholds
        assert engine.decide(0.75) == "accept", "0.75 should be accept"
        assert engine.decide(0.74) == "improve", "0.74 should be improve"
        assert engine.decide(0.5) == "improve", "0.5 should be improve"
        assert engine.decide(0.49) == "retry", "0.49 should be retry"
        
        print("✓ REGRESSION TEST: Decision thresholds unchanged")
        return True
        
    except AssertionError as e:
        print(f"✗ REGRESSION TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    print("\n=== REGRESSION TESTS (Baseline Compatibility) ===\n")
    
    passed = 0
    total = 0
    
    tests = [
        test_query_agent_baseline,
        test_eval_engine_compatibility,
        test_decision_threshold_logic,
    ]
    
    for test_fn in tests:
        total += 1
        if test_fn():
            passed += 1
    
    print(f"\nRegression Results: {passed}/{total} tests passed")
    if passed == total:
        print("Status: ALL REGRESSION TESTS PASS ✓")
    else:
        print("Status: SOME TESTS FAILED ✗")
