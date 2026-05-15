"""
Tests for Stage 3 addition to MilvusVectorTool:
  - point_get(): scalar filter point-get by chunk_id or node_id
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.app.config import DEFAULT_CONFIG
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def _make_tool() -> tuple[MilvusVectorTool, MagicMock]:
    """Return a MilvusVectorTool with a mock _client injected."""
    with patch("src.core.tools.vector.milvus_tool._get_milvus_client_by_uri") as mock_factory:
        mock_client = MagicMock()
        mock_factory.return_value = mock_client
        tool = MilvusVectorTool(DEFAULT_CONFIG)
    return tool, mock_client


class TestPointGet(unittest.TestCase):

    def test_chunk_id_filter_contains_field_and_ids(self):
        tool, client = _make_tool()
        client.query.return_value = []
        tool.point_get(["c1", "c2"], by="chunk_id")
        filter_used = client.query.call_args.kwargs["filter"]
        self.assertIn("chunk_id", filter_used)
        self.assertIn('"c1"', filter_used)
        self.assertIn('"c2"', filter_used)

    def test_node_id_filter_uses_node_id_field(self):
        tool, client = _make_tool()
        client.query.return_value = []
        tool.point_get(["n1", "n2"], by="node_id")
        filter_used = client.query.call_args.kwargs["filter"]
        self.assertIn("node_id", filter_used)
        self.assertIn('"n1"', filter_used)

    def test_empty_ids_returns_empty_without_query(self):
        tool, client = _make_tool()
        result = tool.point_get([])
        self.assertEqual(result, [])
        client.query.assert_not_called()

    def test_client_none_returns_empty(self):
        tool, _ = _make_tool()
        tool._client = None
        self.assertEqual(tool.point_get(["c1"]), [])

    def test_rows_converted_to_documents(self):
        tool, client = _make_tool()
        client.query.return_value = [
            {
                "text": "人参的功效",
                "doc_type": "drug",
                "node_id": "n1",
                "chunk_id": "c1",
                "chunk_type": "",
                "source": "",
                "drug_name": "人参",
                "canonical_name": "人参",
                "normalized_name": "",
                "aliases_text": "",
                "alias_norms_text": "",
                "answer": "",
            }
        ]
        docs = tool.point_get(["c1"])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "人参的功效")
        self.assertEqual(docs[0].metadata.get("node_id"), "n1")
        self.assertEqual(docs[0].metadata.get("search_source"), "point_get")

    def test_rows_without_text_are_skipped(self):
        tool, client = _make_tool()
        client.query.return_value = [
            {
                "text": "",
                "doc_type": "drug",
                "node_id": "n1",
                "chunk_id": "c1",
                "chunk_type": "",
                "source": "",
                "drug_name": "",
                "canonical_name": "",
                "normalized_name": "",
                "aliases_text": "",
                "alias_norms_text": "",
                "answer": "",
            }
        ]
        docs = tool.point_get(["c1"])
        self.assertEqual(docs, [])

    def test_client_exception_returns_empty(self):
        tool, client = _make_tool()
        client.query.side_effect = RuntimeError("Milvus unavailable")
        self.assertEqual(tool.point_get(["c1"]), [])

    def test_limit_is_passed_to_query(self):
        tool, client = _make_tool()
        client.query.return_value = []
        tool.point_get(["c1"], limit=42)
        self.assertEqual(client.query.call_args.kwargs["limit"], 42)


if __name__ == "__main__":
    unittest.main()
