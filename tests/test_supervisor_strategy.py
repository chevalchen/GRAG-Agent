import unittest
from unittest.mock import MagicMock

from src.app.online_qa.nodes.supervisor_node import _classify_complexity, make_supervisor_node
from src.core.tools.llm.generation_tool import LLMGenerationTool


def _make_node_with_payload(payload_text: str):
    llm = MagicMock(spec=LLMGenerationTool)
    llm.invoke_text.return_value = payload_text
    return make_supervisor_node(llm)


class SupervisorStrategyTests(unittest.TestCase):
    def test_empty_query_returns_safe_default_route(self):
        node = _make_node_with_payload("")
        state = node({"query": "", "metrics": {}})
        routing = state["routing"]
        self.assertEqual(routing["query_intent"], "tcm_theory")
        self.assertFalse(routing["use_graph"])
        self.assertFalse(routing["use_drug_vec"])
        self.assertTrue(routing["use_lit_vec"])
        self.assertFalse(routing["use_health_vec"])

    def test_invalid_intent_falls_back_to_tcm_theory(self):
        node = _make_node_with_payload(
            '{"query_intent":"unknown","use_graph":true,"use_drug_vec":true,"keywords":["补气"],"source_hint":[]}'
        )
        state = node({"query": "补气是什么意思", "metrics": {}})
        routing = state["routing"]
        self.assertEqual(routing["query_intent"], "tcm_theory")
        self.assertFalse(routing["use_graph"])
        self.assertFalse(routing["use_drug_vec"])
        self.assertTrue(routing["use_lit_vec"])
        self.assertFalse(routing["use_health_vec"])

    def test_drug_specific_uses_expected_routes_when_flags_missing(self):
        node = _make_node_with_payload('{"query_intent":"drug_specific","keywords":["乌鸡白凤丸"],"source_hint":[]}')
        state = node({"query": "乌鸡白凤丸适应症", "metrics": {}})
        routing = state["routing"]
        self.assertEqual(routing["query_intent"], "drug_specific")
        self.assertTrue(routing["use_graph"])
        self.assertTrue(routing["use_drug_vec"])
        self.assertFalse(routing["use_lit_vec"])
        self.assertFalse(routing["use_health_vec"])


class SupervisorComplexityTests(unittest.TestCase):
    def test_classify_simple_drug_query(self):
        level = _classify_complexity("乌鸡白凤丸适应症", "drug_specific", ["乌鸡白凤丸", "适应症"])
        self.assertEqual(level, "simple")

    def test_classify_complex_symptom_query(self):
        level = _classify_complexity("失眠并且气虚如何调理，同时伴有乏力", "symptom_disease", ["失眠", "气虚", "调理", "乏力"])
        self.assertEqual(level, "complex")


if __name__ == "__main__":
    unittest.main()
