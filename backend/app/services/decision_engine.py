"""Decision Engine for Self-Healing Retry Logic

Makes decisions on whether to accept, improve, or retry responses
based on evaluation scores.
"""


class DecisionEngine:
    """Decides next action based on evaluation score."""

    def __init__(self, accept_threshold: float = 0.75, improve_threshold: float = 0.5):
        """
        Initialize decision thresholds.

        Args:
            accept_threshold: Score >= this → "accept" (default 0.75)
            improve_threshold: Score >= this → "improve" (default 0.5)
                              Score < this → "retry"
        """
        self.accept_threshold = accept_threshold
        self.improve_threshold = improve_threshold

    def decide(self, score: float) -> str:
        """
        Decision logic based on score tiers.

        Args:
            score: Aggregated evaluation score (0.0 - 1.0)

        Returns:
            str: "accept", "improve", or "retry"
        """
        if score >= self.accept_threshold:
            return "accept"
        elif score >= self.improve_threshold:
            return "improve"
        return "retry"
