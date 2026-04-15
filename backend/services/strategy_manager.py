"""Strategy Manager for Adaptive Query/Parameter Modification

Handles query rewrites and parameter adjustments for retry attempts.
"""


class StrategyManager:
    """Applies strategies to improve response quality on retry."""

    def improve_prompt(self, query: str) -> str:
        """
        Strategy: Add instructions to improve response quality.

        Used when score is "improve" (medium quality).
        """
        return f"Provide a detailed, accurate, and well-structured answer: {query}"

    def rewrite_query(self, query: str) -> str:
        """
        Strategy: Rewrite query for better retrieval.

        Used when score is "retry" (low quality).
        Aims to expand context and improve document matching.
        """
        return f"Clarify and expand the following query for better retrieval: {query}"

    def expand_retrieval(self, params: dict) -> dict:
        """
        Strategy: Increase retrieval size for more context.

        Used with rewrite_query to fetch more documents.

        Args:
            params: Query parameters dict (should contain "top_k")

        Returns:
            Modified params with increased top_k
        """
        current_top_k = params.get("top_k", 5)
        params["top_k"] = min(current_top_k + 3, 15)
        return params
