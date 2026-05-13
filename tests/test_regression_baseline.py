from backend.services.decision_engine import DecisionEngine
from backend.services.eval_engine import EvaluationEngine


def test_eval_engine_compatibility():
    engine = EvaluationEngine()
    scores = engine.evaluate(
        query="What is AI?",
        response="AI is artificial intelligence that enables machines to learn and perform tasks.",
        context="AI context document",
    )
    assert "relevance" in scores
    assert "grounding" in scores
    assert "completeness" in scores
    assert 0 <= engine.final_score(scores) <= 1.0


def test_decision_threshold_logic():
    engine = DecisionEngine(accept_threshold=0.75, improve_threshold=0.5)
    assert engine.decide(0.75) == "accept"
    assert engine.decide(0.74) == "improve"
    assert engine.decide(0.5) == "improve"
    assert engine.decide(0.49) == "retry"

