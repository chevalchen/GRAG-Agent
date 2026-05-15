"""
Tests for Stage 3 additions to Neo4jGraphTool:
  - expand_by_node_id(): multi-hop subgraph expansion with depth whitelist
  - find_drugs_by_caution(): caution/adverse/indication lookup
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.app.config import DEFAULT_CONFIG
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool, _DEPTH_WHITELIST


def _make_tool() -> tuple[Neo4jGraphTool, MagicMock]:
    """Return a Neo4jGraphTool with a mock _graph injected."""
    with patch("src.core.tools.graph.neo4j_tool._get_graph") as mock_get:
        mock_graph = MagicMock()
        mock_get.return_value = mock_graph
        tool = Neo4jGraphTool(DEFAULT_CONFIG)
    return tool, mock_graph


# ---------------------------------------------------------------------------
# expand_by_node_id
# ---------------------------------------------------------------------------

class TestExpandByNodeId(unittest.TestCase):

    def test_depth_whitelist_values(self):
        self.assertEqual(_DEPTH_WHITELIST, frozenset({1, 2, 3}))

    def test_valid_depth_2_injected_in_cypher(self):
        tool, graph = _make_tool()
        graph.query.return_value = []
        tool.expand_by_node_id("node-001", depth=2)
        cypher = graph.query.call_args[0][0]
        self.assertIn("1..2", cypher)

    def test_valid_depth_1_injected_in_cypher(self):
        tool, graph = _make_tool()
        graph.query.return_value = []
        tool.expand_by_node_id("node-001", depth=1)
        cypher = graph.query.call_args[0][0]
        self.assertIn("1..1", cypher)

    def test_valid_depth_3_injected_in_cypher(self):
        tool, graph = _make_tool()
        graph.query.return_value = []
        tool.expand_by_node_id("node-001", depth=3)
        cypher = graph.query.call_args[0][0]
        self.assertIn("1..3", cypher)

    def test_invalid_depth_clamps_to_2(self):
        tool, graph = _make_tool()
        graph.query.return_value = []
        tool.expand_by_node_id("node-001", depth=99)
        cypher = graph.query.call_args[0][0]
        self.assertIn("1..2", cypher)

    def test_depth_0_clamps_to_2(self):
        tool, graph = _make_tool()
        graph.query.return_value = []
        tool.expand_by_node_id("node-001", depth=0)
        cypher = graph.query.call_args[0][0]
        self.assertIn("1..2", cypher)

    def test_rows_mapped_to_dicts(self):
        tool, graph = _make_tool()
        graph.query.return_value = [
            {
                "source_id": "s1",
                "source_name": "Drug A",
                "source_type": "Drug",
                "rel_path": ["CONTAINS"],
                "target_id": "t1",
                "target_name": "Ingredient X",
                "target_type": "Ingredient",
            }
        ]
        results = tool.expand_by_node_id("s1", depth=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["target_name"], "Ingredient X")
        self.assertEqual(results[0]["rel_path"], ["CONTAINS"])

    def test_rel_types_filter_applied_in_python(self):
        tool, graph = _make_tool()
        graph.query.return_value = [
            {"source_id": "s1", "source_name": "A", "source_type": "Drug",
             "rel_path": ["CONTAINS"], "target_id": "t1", "target_name": "B", "target_type": "Ingredient"},
            {"source_id": "s1", "source_name": "A", "source_type": "Drug",
             "rel_path": ["INDICATES"], "target_id": "t2", "target_name": "C", "target_type": "Symptom"},
        ]
        results = tool.expand_by_node_id("s1", depth=1, rel_types=["CONTAINS"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["target_id"], "t1")

    def test_graph_none_returns_empty(self):
        tool, _ = _make_tool()
        tool._graph = None
        self.assertEqual(tool.expand_by_node_id("node-001"), [])

    def test_graph_exception_returns_empty(self):
        tool, graph = _make_tool()
        graph.query.side_effect = RuntimeError("Neo4j connection lost")
        self.assertEqual(tool.expand_by_node_id("node-001"), [])

    def test_empty_rows_returns_empty_list(self):
        tool, graph = _make_tool()
        graph.query.return_value = []
        self.assertEqual(tool.expand_by_node_id("node-001"), [])


# ---------------------------------------------------------------------------
# find_drugs_by_caution
# ---------------------------------------------------------------------------

class TestFindDrugsByCaution(unittest.TestCase):

    def test_population_query_uses_caution_for_cypher(self):
        tool, graph = _make_tool()
        graph.query.return_value = [
            {"drug_node_id": "d1", "drug_name": "人参", "reason_label": "孕妇"}
        ]
        results = tool.find_drugs_by_caution(population_name="孕妇")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["drug_name"], "人参")
        self.assertIn("CAUTION_FOR", results[0]["reason_path"])

    def test_adverse_query_uses_has_adverse_reaction_cypher(self):
        tool, graph = _make_tool()
        graph.query.return_value = [
            {"drug_node_id": "d2", "drug_name": "附子", "reason_label": "头晕"}
        ]
        results = tool.find_drugs_by_caution(adverse_name="头晕")
        self.assertEqual(len(results), 1)
        self.assertIn("HAS_ADVERSE_REACTION", results[0]["reason_path"])

    def test_symptom_query_uses_indicates_cypher(self):
        tool, graph = _make_tool()
        graph.query.return_value = [
            {"drug_node_id": "d3", "drug_name": "乌鸡白凤丸", "reason_label": "月经不调"}
        ]
        results = tool.find_drugs_by_caution(symptom_name="月经不调")
        self.assertEqual(len(results), 1)
        self.assertIn("INDICATES", results[0]["reason_path"])

    def test_multiple_params_issues_multiple_queries(self):
        tool, graph = _make_tool()
        graph.query.side_effect = [
            [{"drug_node_id": "d1", "drug_name": "人参", "reason_label": "孕妇"}],
            [{"drug_node_id": "d2", "drug_name": "附子", "reason_label": "头晕"}],
        ]
        results = tool.find_drugs_by_caution(population_name="孕妇", adverse_name="头晕")
        self.assertEqual(graph.query.call_count, 2)
        self.assertEqual(len(results), 2)

    def test_deduplicates_by_drug_node_id(self):
        tool, graph = _make_tool()
        # Same drug returned by both population and adverse queries
        graph.query.side_effect = [
            [{"drug_node_id": "d1", "drug_name": "附子", "reason_label": "孕妇"}],
            [{"drug_node_id": "d1", "drug_name": "附子", "reason_label": "头晕"}],
        ]
        results = tool.find_drugs_by_caution(population_name="孕妇", adverse_name="头晕")
        self.assertEqual(len(results), 1)

    def test_no_args_returns_empty_without_querying(self):
        tool, graph = _make_tool()
        results = tool.find_drugs_by_caution()
        self.assertEqual(results, [])
        graph.query.assert_not_called()

    def test_graph_none_returns_empty(self):
        tool, _ = _make_tool()
        tool._graph = None
        self.assertEqual(tool.find_drugs_by_caution(population_name="孕妇"), [])

    def test_query_exception_is_swallowed(self):
        tool, graph = _make_tool()
        graph.query.side_effect = RuntimeError("Neo4j error")
        # Should not raise; returns empty
        results = tool.find_drugs_by_caution(population_name="孕妇")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
