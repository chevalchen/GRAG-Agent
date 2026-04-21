import unittest
from unittest.mock import MagicMock, patch

from src.app.online_qa.nodes.drug_entity_resolve_node import _candidate_mentions, make_drug_entity_resolve_node
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool


class DrugEntityResolveTests(unittest.TestCase):
    def test_candidate_mentions_include_reduced_name(self):
        mentions = _candidate_mentions("乌鸡 白凤丸的适应症是什么？", ["乌鸡 白凤丸"])
        self.assertIn("乌鸡 白凤丸", mentions)
        self.assertIn("乌鸡 白凤丸的适应症是什么？", mentions)
        self.assertTrue(any("乌鸡 白凤丸" in x or "乌鸡白凤丸" in x for x in mentions))

    def test_resolve_node_stores_resolved_drug(self):
        neo4j = MagicMock(spec=Neo4jGraphTool)
        node = make_drug_entity_resolve_node(neo4j)
        resolved = {
            "canonical_name": "乌鸡白凤丸",
            "normalized_name": "乌鸡白凤丸",
            "node_id": "drug_123",
            "matched_alias": "乌鸡 白凤丸",
            "match_type": "alias_norm",
        }
        with patch("src.app.online_qa.nodes.drug_entity_resolve_node._resolve_with_graph", return_value=resolved):
            state = node(
                {
                    "query": "乌鸡 白凤丸的适应症是什么？",
                    "routing": {"use_graph": True, "use_drug_vec": True, "keywords": ["乌鸡 白凤丸"]},
                }
            )
        self.assertEqual(state["resolved_drug"]["canonical_name"], "乌鸡白凤丸")
        self.assertEqual(state["resolved_drug"]["node_id"], "drug_123")

    def test_resolve_node_skips_when_routes_disabled(self):
        neo4j = MagicMock(spec=Neo4jGraphTool)
        node = make_drug_entity_resolve_node(neo4j)
        state = node({"query": "乌鸡白凤丸", "routing": {"use_graph": False, "use_drug_vec": False, "keywords": []}})
        self.assertIsNone(state["resolved_drug"])


if __name__ == "__main__":
    unittest.main()
