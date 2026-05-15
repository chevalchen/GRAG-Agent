"""Tests for router_node: intent classification, rewrite, entity extraction."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.app.online_qa.nodes.router_node import make_router_node
from src.app.online_qa.state import OnlineQAState


def _make_llm(json_response: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke_text.return_value = json_response
    return llm


def _state(**kwargs) -> OnlineQAState:
    base: OnlineQAState = {"query": "test", "metrics": {}, "iteration_count": 0}
    base.update(kwargs)  # type: ignore[arg-type]
    return base


class TestRouterNode:
    def test_symptom_to_drug_intent(self):
        llm = _make_llm(
            '{"intent":"symptom_to_drug","rewritten_query":"感冒头疼用什么药",'
            '"drugs":[],"diseases":[],"symptoms":["感冒","头疼"],'
            '"keywords":["感冒","头疼"],"complexity_level":"simple","reason":"症状寻药"}'
        )
        node = make_router_node(llm)
        result = node(_state(query="感冒头疼用什么药"))
        assert result["routing"]["intent"] == "symptom_to_drug"
        assert result["rewritten_query"] == "感冒头疼用什么药"
        assert "感冒" in result["extracted_entities"]["symptoms"]

    def test_conditions_exact_intent(self):
        llm = _make_llm(
            '{"intent":"conditions_exact","rewritten_query":"老人禁用哪些中成药",'
            '"drugs":[],"diseases":[],"symptoms":[],'
            '"keywords":["老人","禁用"],"complexity_level":"complex","reason":"禁忌查询"}'
        )
        node = make_router_node(llm)
        result = node(_state(query="老人禁用哪些中成药"))
        assert result["routing"]["intent"] == "conditions_exact"
        assert result["routing"]["complexity_level"] == "complex"

    def test_chat_intent(self):
        llm = _make_llm(
            '{"intent":"chat","rewritten_query":"你好",'
            '"drugs":[],"diseases":[],"symptoms":[],'
            '"keywords":[],"complexity_level":"simple","reason":"闲聊"}'
        )
        node = make_router_node(llm)
        result = node(_state(query="你好"))
        assert result["routing"]["intent"] == "chat"

    def test_fallback_on_invalid_intent(self):
        """LLM returns unknown intent; should fallback to symptom_to_drug."""
        llm = _make_llm(
            '{"intent":"unknown_intent","rewritten_query":"test",'
            '"drugs":[],"diseases":[],"symptoms":[],'
            '"keywords":[],"complexity_level":"simple","reason":"?"}'
        )
        node = make_router_node(llm)
        result = node(_state(query="test"))
        assert result["routing"]["intent"] == "symptom_to_drug"

    def test_fallback_on_llm_exception(self):
        """LLM raises exception; should return symptom_to_drug with empty entities."""
        llm = MagicMock()
        llm.invoke_text.side_effect = RuntimeError("LLM down")
        node = make_router_node(llm)
        result = node(_state(query="什么药治头疼"))
        assert result["routing"]["intent"] == "symptom_to_drug"
        assert result["rewritten_query"] == "什么药治头疼"

    def test_drug_entities_extracted(self):
        llm = _make_llm(
            '{"intent":"symptom_to_drug","rewritten_query":"乌鸡白凤丸适应症",'
            '"drugs":["乌鸡白凤丸"],"diseases":[],"symptoms":[],'
            '"keywords":["乌鸡白凤丸","适应症"],"complexity_level":"simple","reason":"寻药"}'
        )
        node = make_router_node(llm)
        result = node(_state(query="乌鸡白凤丸适应症"))
        assert "乌鸡白凤丸" in result["extracted_entities"]["drugs"]

    def test_empty_query_returns_default(self):
        llm = _make_llm("")
        node = make_router_node(llm)
        result = node(_state(query=""))
        assert result["routing"]["intent"] == "symptom_to_drug"
        assert result["rewritten_query"] == ""

    def test_iteration_count_preserved(self):
        llm = _make_llm('{"intent":"theory","rewritten_query":"q","drugs":[],"diseases":[],"symptoms":[],"keywords":[],"complexity_level":"simple","reason":"r"}')
        node = make_router_node(llm)
        result = node(_state(query="q", iteration_count=1))
        assert result["iteration_count"] == 1
