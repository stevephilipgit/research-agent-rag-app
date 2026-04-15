"""Evaluation Engine for LLM Response Assessment

Provides lightweight heuristic-based evaluation of LLM responses.
Extensible for future LLM-based evaluation backends.
"""

import json
import os
from typing import Callable, Dict, Optional

USE_LLM_EVAL = os.getenv("USE_LLM_EVAL", "false").lower() == "true"


class EvaluationEngine:
    """Evaluates LLM responses and returns structured scores."""

    def evaluate(self, query: str, response: str, context: str = "") -> Dict:
        """
        Evaluate a response against query and context.

        Args:
            query: Original user query
            response: LLM-generated response
            context: Retrieved context/RAG documents

        Returns:
            Dict with keys: relevance, grounding, completeness
        """
        return {
            "relevance": self._relevance_score(query, response),
            "grounding": self._grounding_score(response, context),
            "completeness": self._completeness_score(response),
        }

    def final_score(self, scores: Dict) -> float:
        """
        Compute aggregated score from component scores.

        Args:
            scores: Dict from evaluate()

        Returns:
            Float between 0.0 and 1.0
        """
        return round(sum(scores.values()) / len(scores), 3)

    def llm_evaluate(
        self, query: str, response: str, context: str, llm_fn: Callable[[str], str]
    ) -> Dict:
        """
        Evaluate response using LLM with structured JSON output.

        Args:
            query: Original user query
            response: LLM-generated response
            context: Retrieved context/RAG documents
            llm_fn: Callable that takes prompt string and returns response string

        Returns:
            Dict with keys: relevance, grounding, completeness, reasoning

        Falls back to heuristic evaluation if LLM evaluation fails.
        """
        prompt = f"""You are an expert evaluator for AI-generated responses.

Evaluate the following response quality:

Query:
{query}

Response:
{response}

Context (Retrieved Documents):
{context if context else "(No context provided)"}

Score each dimension from 0 to 1.0:
- relevance: How well does the response address the query?
- grounded: How much is the response supported by the context?
- completeness: Is the response thorough and complete?

Return ONLY valid JSON (no markdown, no extra text):
{{
  "relevance": <float 0-1>,
  "grounded": <float 0-1>,
  "completeness": <float 0-1>,
  "reasoning": "<brief explanation>"
}}"""

        try:
            # Call LLM function
            result_str = llm_fn(prompt)

            # Try to parse JSON
            # Handle potential markdown code blocks
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0]
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0]

            parsed = json.loads(result_str.strip())

            # Validate and normalize scores
            return {
                "relevance": min(max(float(parsed.get("relevance", 0.5)), 0.0), 1.0),
                "grounding": min(max(float(parsed.get("grounded", 0.5)), 0.0), 1.0),
                "completeness": min(max(float(parsed.get("completeness", 0.5)), 0.0), 1.0),
                "reasoning": str(parsed.get("reasoning", "")),
            }

        except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
            # Fallback to heuristic evaluation on any parsing error
            return self.evaluate(query, response, context)

    def evaluate_with_fallback(
        self, query: str, response: str, context: str, llm_fn: Optional[Callable[[str], str]] = None
    ) -> Dict:
        """
        Evaluate response using LLM if enabled, otherwise use heuristics.

        Args:
            query: Original user query
            response: LLM-generated response
            context: Retrieved context/RAG documents
            llm_fn: Optional LLM callable (required if USE_LLM_EVAL=true)

        Returns:
            Dict with keys: relevance, grounding, completeness, (and reasoning if LLM used)
        """
        if USE_LLM_EVAL and llm_fn is not None:
            return self.llm_evaluate(query, response, context, llm_fn)

        # Default: Use heuristic evaluation
        return self.evaluate(query, response, context)

    def _relevance_score(self, query: str, response: str) -> float:
        """
        Compute relevance: overlap between query and response tokens.

        Higher overlap = response addresses query topics.
        """
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        overlap = query_words.intersection(response_words)
        return min(len(overlap) / (len(query_words) + 1), 1.0)

    def _grounding_score(self, response: str, context: str) -> float:
        """
        Compute grounding: overlap between response and context.

        Higher overlap = response is grounded in retrieved docs.
        Falls back to neutral if no context provided.
        """
        if not context:
            return 0.5  # neutral if no context

        context_words = set(context.lower().split())
        response_words = set(response.lower().split())

        overlap = response_words.intersection(context_words)
        return min(len(overlap) / (len(response_words) + 1), 1.0)

    def _completeness_score(self, response: str) -> float:
        """
        Compute completeness: response length heuristic.

        Longer responses tend to be more complete.
        """
        length = len(response.split())

        if length > 100:
            return 1.0
        elif length > 50:
            return 0.7
        elif length > 20:
            return 0.5
        return 0.3
