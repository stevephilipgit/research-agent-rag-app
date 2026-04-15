# Self-Healing Layer Documentation

## Overview

The self-healing layer is an optional, production-grade feature that improves LLM response quality through:

- **Evaluation Scoring**: Assesses responses on relevance, grounding, and completeness
- **Decision Engine**: Makes retry decisions based on score thresholds
- **Adaptive Retry Strategy**: Modifies queries when quality is insufficient
- **Metrics Logging**: Tracks evaluation scores, retry attempts, and latency

## Architecture

### Components

```
backend/services/
├── eval_engine.py          # Response evaluation with heuristics
├── decision_engine.py      # Manages accept/improve/retry decisions
├── strategy_manager.py     # Query modification strategies
├── self_healing.py         # Orchestrates the self-healing flow
└── metrics_service.py      # Logs metrics to telemetry system
```

### Flow Diagram

```
Query Input
    ↓
generate_answer(query) → LLM Response
    ↓
evaluate(query, response, context) → Scores
    ↓
decide(final_score) → Decision
    ↓
┌─────────────────────────────────────────┐
│ Decision Logic:                         │
│ • score >= 0.75 → Accept (STOP)        │
│ • 0.5 <= score < 0.75 → Improve       │
│ • score < 0.5 → Retry                  │
└─────────────────────────────────────────┘
    ↓
if improve: improve_prompt(query)
if retry: rewrite_query(query)
    ↓
[loop up to MAX_RETRIES=2]
    ↓
Return Final Response + Score + Retry Count
```

## Enabling the Feature

### 1. Set Environment Variable

Add to `.env` or Render environment settings:

```env
ENABLE_SELF_HEALING=true
```

Default is `false` (disabled) for backward compatibility.

### 2. Verify Installation

All required modules are in `backend/services/`:

```bash
ls -la backend/services/eval_engine.py
ls -la backend/services/decision_engine.py
ls -la backend/services/strategy_manager.py
ls -la backend/services/self_healing.py
ls -la backend/services/metrics_service.py
```

## Usage

### Default Behavior (Disabled)

When `ENABLE_SELF_HEALING=false` (default):

```python
# Uses original flow - no overhead
query_agent(query, session_id) → response (fast path)
```

### Enabled Behavior

When `ENABLE_SELF_HEALING=true`:

```python
# Self-healing flow wraps answer generation
query_agent(query, session_id) → response (with evaluation & potential retries)
```

The integration point is in [services/rag_service.py](services/rag_service.py#L257):

```python
def query_agent(query: str, session_id: str = "default") -> dict:
    # ... setup code ...
    
    if ENABLE_SELF_HEALING:
        # Wrap with self-healing
        answer, eval_score, retry_count = self_healing_flow(
            query, 
            generate_answer,  # closure that calls run_research_agent
            context=""
        )
    else:
        # Original path
        answer = run_research_agent(query, ...)
    
    # ... return result ...
```

## Scoring System

### Component Scores

Each response is evaluated on three dimensions:

| Dimension | Range | Calculation | Meaning |
|-----------|-------|-------------|---------|
| **Relevance** | 0.0-1.0 | Token overlap (query ∩ response) / query size | Response addresses query topics |
| **Grounding** | 0.0-1.0 | Token overlap (response ∩ context) / response size | Response backed by retrieved docs |
| **Completeness** | 0.0-1.0 | Response length heuristic (0.3-1.0) | Response depth and detail |

### Final Score

```
final_score = (relevance + grounding + completeness) / 3
```

### Decision Thresholds

| Score | Decision | Action |
|-------|----------|--------|
| `>= 0.75` | **Accept** | Return immediately |
| `0.5 - 0.75` | **Improve** | Add instructions to prompt |
| `< 0.5` | **Retry** | Rewrite query for better retrieval |

## LLM-Based Evaluation (Beta)

### Overview

By default, self-healing uses lightweight heuristics for fast evaluation. Optionally, you can enable LLM-based evaluation for more sophisticated response assessment.

### Enabling LLM Evaluation

```bash
# .env
USE_LLM_EVAL=true
```

### How It Works

When `USE_LLM_EVAL=true`:

1. **Response evaluation uses LLM** (instead of token overlap heuristics)
2. **LLM assesses**: relevance, grounding, completeness, and provides reasoning
3. **Structured JSON output**: Scores normalized to 0-1 range
4. **Graceful fallback**: Reverts to heuristics if LLM call fails

### Cost Optimization

LLM-based evaluation is **expensive** (requires additional API calls). The system implements **automatic cost optimization**:

```python
# Cost Optimization Logic:
if response_word_count < 50:
    # Short response (potentially incomplete)
    # → Use expensive LLM evaluation to verify quality
    use_llm_eval = True
else:
    # Long response (probably substantial)
    # → Use fast heuristics (cheaper)
    use_llm_eval = False
```

**Rationale**:
- Short responses (< 50 words) might be incomplete → Need LLM verification
- Long responses (≥ 50 words) are likely substantive → Heuristics suffice
- **Net effect**: Minimal cost increase, better quality assurance for risky short responses

### LLM Evaluation Output

```json
{
  "relevance": 0.85,
  "grounding": 0.92,
  "completeness": 0.78,
  "reasoning": "Response directly addresses all aspects of the query with evidence from retrieved documents."
}
```

### Performance Impact

**Without USE_LLM_EVAL** (default):
- Evaluation: ~10ms (token matching)
- No additional API calls

**With USE_LLM_EVAL** (optimized):
- Short responses (< 50 words): ~1500ms + 1 LLM call
- Long responses (≥ 50 words): ~10ms (heuristics used)
- **Average**: ~700ms per query (50% of queries trigger LLM eval in typical use)

### Cost Implications

If 50% of responses are short:
- Base cost: N API calls per query
- Self-healing cost: ~+10 additional API calls per 100 queries (assuming USE_LLM_EVAL)
- **Recommendation**: Use for high-stakes applications; disable for cost-sensitive deployments

## Metrics Logging

All self-healing events are logged to the telemetry system (scope: `self_healing`):

### Evaluation Logs

```json
{
  "step": "Self-Healing Evaluation",
  "status": "success",
  "detail": {
    "query": "...",
    "response_length": 150,
    "scores": {
      "relevance": 0.75,
      "grounding": 0.82,
      "completeness": 0.70
    },
    "final_score": 0.76,
    "decision": "accept",
    "elapsed_ms": 1250
  }
}
```

### Retry Logs

```json
{
  "step": "Self-Healing Retry",
  "status": "in_progress",
  "detail": {
    "attempt": 1,
    "strategy": "improve_prompt",
    "modified_query": "Provide a detailed, accurate...",
    "reason": "improve"
  }
}
```

### Completion Logs

```json
{
  "step": "Self-Healing Complete",
  "status": "success",
  "detail": {
    "total_retries": 1,
    "final_score": 0.78,
    "elapsed_ms": 2450,
    "accepted": true
  }
}
```

## Configuration

### Enabling LLM-Based Evaluation

Set in `.env` or environment:

```bash
# Default: false (use fast heuristics)
USE_LLM_EVAL=true

# Also requires ENABLE_SELF_HEALING to be active
ENABLE_SELF_HEALING=true
```

When enabled:
- Short responses (< 50 words): Evaluated by LLM (quality check)
- Long responses (≥ 50 words): Evaluated by heuristics (cost optimization)
- Fallback: Returns to heuristics if LLM call fails

### Adjusting Thresholds

Edit the decision engine thresholds in `services/self_healing.py`:

```python
# Current defaults
decision_engine = DecisionEngine(
    accept_threshold=0.75,    # Adjust to accept lower/higher quality
    improve_threshold=0.5     # Adjust improve/retry cutoff
)
```

### Customizing Strategies

Modify query modification strategies in `services/strategy_manager.py`:

```python
class StrategyManager:
    def improve_prompt(self, query: str) -> str:
        # Customize instruction prompt
        return f"..."
    
    def rewrite_query(self, query: str) -> str:
        # Customize query rewriting
        return f"..."
```

### Extending Evaluation

The evaluation engine uses lightweight heuristics and is designed for easy extension:

```python
# backend/services/eval_engine.py

class EvaluationEngine:
    def _relevance_score(self, query: str, response: str) -> float:
        # Replace with LLM-based relevance scoring
        pass
    
    def _grounding_score(self, response: str, context: str) -> float:
        # Use more sophisticated grounding validation
        pass
```

## Production Considerations

### Performance Impact

With `ENABLE_SELF_HEALING=true`:

- **Best case** (accept on first try): +0% overhead (1 LLM call, 1 evaluation)
- **Typical case** (1 retry): +50% latency (2 LLM calls)
- **Worst case** (2 retries): +100% latency (3 LLM calls)

The `MAX_RETRIES=2` limit ensures bounded latency even with low-quality responses.

### Cost Implications

With self-healing enabled, each low-quality response triggers additional LLM calls:

- 1 retry: ~2x cost for that query
- 2 retries: ~3x cost for that query

**Recommendation**: Enable for high-stakes applications; disable for cost-sensitive deployments.

### Monitoring

Track these metrics in your observability system:

```python
# Via logs scope="self_healing"
- eval score distribution (should be high if working)
- retry count distribution (should be low if working)
- latency overhead (2x-3x in worst case)
```

## Fallback Behavior

If self-healing fails or is disabled:

1. **Config flag disabled**: Original flow executes (no wrapper)
2. **Import error**: Gracefully caught; falls back to original flow
3. **Runtime error in evaluation**: Logs error; returns last response

The system is designed to never break the main query flow.

## Testing

### Unit Tests

```bash
# Test evaluation engine
python -m pytest tests/test_eval_engine.py

# Test decision engine
python -m pytest tests/test_decision_engine.py

# Test strategy manager
python -m pytest tests/test_strategy_manager.py

# Test self-healing flow
python -m pytest tests/test_self_healing.py
```

### Integration Test

With self-healing enabled:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-session" \
  -d '{"query": "What is machine learning?"}'
```

Check logs for `scope="self_healing"` entries.

## Troubleshooting

### Feature Not Activating

1. Verify `ENABLE_SELF_HEALING=true` in environment
2. Check Python imports: `from config.settings import ENABLE_SELF_HEALING`
3. Look for warnings in startup logs

### High Retry Rates

If most responses trigger retries:

1. **Lower `accept_threshold`**: Accept lower quality scores
2. **Improve evaluation**: Heuristics may not match your domain
3. **Check LLM quality**: Upstream model issues cause low scores

### Evaluation Logic Issues

If scores seem incorrect:

1. Review `_relevance_score`, `_grounding_score`, `_completeness_score`
2. Adjust thresholds for your use case
3. Consider domain-specific metrics

## Security Notes

- Self-healing modifies query text but does NOT execute arbitrary code
- All modifications go through `StrategyManager` (deterministic, safe)
- Query injection is still prevented by upstream security layers
- Metrics logging contains query text (limited to first 100 chars)

## Future Enhancements

Possible improvements (not yet implemented):

1. **LLM-based evaluation**: Use Claude/GPT-4 for sophisticated grounding
2. **Adaptive thresholds**: Learn optimal thresholds from data
3. **RAG-aware strategies**: Use retrieval scores in decision logic
4. **A/B testing framework**: Compare self-healing vs baseline
5. **Custom strategy plugins**: Extensible strategy system

## Examples

### Example 1: Basic Query with Self-Healing

```bash
ENABLE_SELF_HEALING=true python -m backend.main

# POST request
{
  "query": "What is transformer architecture?"
}

# Response includes scores in logs:
# score=0.68 (improve) → modified query → retry
# score=0.81 (accept) → returned with 1 retry
```

### Example 2: Long Context Queries

Self-healing is especially helpful for complex queries:

```json
{
  "query": "Compare supervised vs unsupervised learning with examples"
}
```

If initial response is generic, self-healing will:
1. Evaluate as low quality (low relevance to "examples")
2. Rewrite query: "Clarify and expand: Compare supervised vs..."
3. Retrieve more docs, generate better answer

### Example 3: Disabling for Specific Deployments

For cost-sensitive environments:

```bash
# .env
ENABLE_SELF_HEALING=false  # Disabled by default
```

No code changes needed; falls back automatically.

## Support

For issues or questions about the self-healing layer:

1. Check logs: `grep "Self-Healing" logs/app.log`
2. Enable debug logging: Set log level to DEBUG
3. Test individual components: `python -c "from services.eval_engine import EvaluationEngine; ..."`
