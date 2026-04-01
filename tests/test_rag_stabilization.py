import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from services.query_rewriter import normalize_query
from core.agent import is_valid_answer
from core.rag import group_by_source
from langchain_core.documents import Document

class TestRAGStabilization(unittest.TestCase):
    
    def test_query_normalization(self):
        query = "Please tell me about artificial intelligence"
        normalized = normalize_query(query)
        self.assertEqual(normalized, "artificial intelligence")
        
        query = "Could you explain the process?"
        normalized = normalize_query(query)
        self.assertEqual(normalized, "process?")

    def test_is_valid_answer(self):
        context = "The capital of France is Paris. It is a major European city."
        
        # Valid answer (overlap > 3 words)
        answer = "The capital city of France is Paris."
        self.assertTrue(is_valid_answer(answer, context))
        
        # Invalid answer (no overlap)
        answer = "I don't know much about geography."
        self.assertFalse(is_valid_answer(answer, context))
        
        # Short answer
        answer = "Paris."
        self.assertFalse(is_valid_answer(answer, context))

    def test_source_grouping(self):
        docs = [
            Document(page_content="Content from A", metadata={"source": "Doc_A"}),
            Document(page_content="Content from B", metadata={"source": "Doc_B"}),
            Document(page_content="More from A", metadata={"source": "Doc_A"}),
        ]
        
        grouped = group_by_source(docs)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0].metadata["source"], "Doc_A")

    def test_no_web_fallback(self):
        from core.tools import all_tools
        tool_names = [getattr(t, "name", "") for t in all_tools]
        self.assertIn("document_search", tool_names)
        self.assertNotIn("read_url", tool_names)
        self.assertNotIn("web_search", tool_names)

if __name__ == "__main__":
    unittest.main()
