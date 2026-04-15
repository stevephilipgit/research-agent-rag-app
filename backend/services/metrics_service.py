"""Metrics Service for Self-Healing Tracking

Logs evaluation scores, retry attempts, and performance metrics
using the existing telemetry infrastructure.
"""

import time
import json
from typing import Dict, Any, Optional
from core.telemetry import emit_log


class MetricsService:
    """Tracks and logs self-healing metrics."""

    @staticmethod
    def log_evaluation(
        query: str,
        response: str,
        scores: Dict[str, float],
        final_score: float,
        decision: str,
        elapsed_time: float
    ) -> None:
        """
        Log evaluation metrics to telemetry system.

        Args:
            query: Original query
            response: Generated response
            scores: Component scores (relevance, grounding, completeness)
            final_score: Aggregated score
            decision: Made decision (accept/improve/retry)
            elapsed_time: Response generation time in seconds
        """
        detail = {
            "query": query[:100],  # First 100 chars for brevity
            "response_length": len(response),
            "scores": scores,
            "final_score": final_score,
            "decision": decision,
            "elapsed_ms": round(elapsed_time * 1000, 2),
        }

        emit_log(
            step="Self-Healing Evaluation",
            status="success",
            detail=json.dumps(detail),
            scope="self_healing"
        )

    @staticmethod
    def log_retry_attempt(
        attempt: int,
        strategy: str,
        modified_query: str,
        reason: str
    ) -> None:
        """
        Log retry attempt during self-healing flow.

        Args:
            attempt: Retry attempt number (1-based)
            strategy: Strategy applied (improve_prompt/rewrite_query)
            modified_query: Updated query for retry
            reason: Reason for retry (improve/retry based on score)
        """
        detail = {
            "attempt": attempt,
            "strategy": strategy,
            "modified_query": modified_query[:100],
            "reason": reason,
        }

        emit_log(
            step="Self-Healing Retry",
            status="in_progress",
            detail=json.dumps(detail),
            scope="self_healing"
        )

    @staticmethod
    def log_self_healing_complete(
        total_retries: int,
        final_score: float,
        elapsed_time: float,
        accepted: bool,
        best_score: Optional[float] = None,
        model_used: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> None:
        """
        UPGRADE 5: Log completion with enhanced metrics.

        Args:
            total_retries: Number of retries performed
            final_score: Final evaluation score
            elapsed_time: Total time spent in self-healing flow
            accepted: Whether final response was accepted
            best_score: Best score among all attempts
            model_used: Model name if fallback used
            top_k: Retrieval parameter used
        """
        detail = {
            "total_retries": total_retries,
            "final_score": final_score,
            "best_score": best_score,
            "elapsed_ms": round(elapsed_time * 1000, 2),
            "accepted": accepted,
            "model_used": model_used,
            "top_k": top_k,
        }

        emit_log(
            step="Self-Healing Complete",
            status="success" if accepted else "warning",
            detail=json.dumps(detail),
            scope="self_healing"
        )

    @staticmethod
    def log_adaptive_decision(
        score: float,
        decision_type: str,
        action: str,
        reason: str
    ) -> None:
        """
        UPGRADE 5: Log adaptive decisions (retrieval, model, evaluation).

        Args:
            score: Current evaluation score
            decision_type: Type of decision (retrieval/model/evaluation)
            action: Action taken
            reason: Reason for action
        """
        detail = {
            "score": score,
            "decision_type": decision_type,
            "action": action,
            "reason": reason,
        }

        emit_log(
            step="Adaptive Decision",
            status="info",
            detail=json.dumps(detail),
            scope="self_healing"
        )
