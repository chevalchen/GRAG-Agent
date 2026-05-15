"""Tests for sufficiency_node: verdict extraction, iteration counting, max-iter guard."""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.app.online_qa.nodes.sufficiency_node import make_sufficiency_node
from src.app.online_qa.state import OnlineQAState


def _doc(text: str) -> Document:
    return Document(page_content=text)


def _state(**kwargs) -> OnlineQAState:
    base: OnlineQAState = {
        "query": "老人感冒用什么药",
        "rewritten_query": "老人感冒用什么药",
        "docs_final": [_doc("药品：感冒颗粒\n功效：清热解毒")],
        "metrics": {},
        "iteration_count": 0,
    }
    base.update(kwargs)  # type: ignore[arg-type]
    return base


def _llm(response: str) -> MagicMock:
    m = MagicMock()
    m.invoke_text.return_value = response
    return m


class TestSufficiencyNode:
    def test_sufficient_verdict(self):
        node = make_sufficiency_node(_llm("sufficient"))
        result = node(_state())
        assert result["sufficiency"] == "sufficient"
        assert result["iteration_count"] == 0  # not incremented on sufficient

    def test_insufficient_increments_count(self):
        node = make_sufficiency_node(_llm("insufficient"))
        result = node(_state())
        assert result["sufficiency"] == "insufficient"
        assert result["iteration_count"] == 1

    def test_chinese_verdict_sufficient(self):
        node = make_sufficiency_node(_llm("充足，资料完整"))
        result = node(_state())
        assert result["sufficiency"] == "sufficient"

    def test_chinese_verdict_insufficient(self):
        node = make_sufficiency_node(_llm("不足，缺少禁忌信息"))
        result = node(_state())
        assert result["sufficiency"] == "insufficient"
        assert result["iteration_count"] == 1

    def test_empty_docs_first_pass_insufficient(self):
        """No docs on first pass → insufficient to trigger loop."""
        node = make_sufficiency_node(_llm("sufficient"))  # LLM not reached for empty docs
        result = node(_state(docs_final=[]))
        assert result["sufficiency"] == "insufficient"
        assert result["iteration_count"] == 1

    def test_empty_docs_second_pass_force_sufficient(self):
        """No docs on second pass → force sufficient to prevent infinite loop."""
        node = make_sufficiency_node(_llm("sufficient"))
        result = node(_state(docs_final=[], iteration_count=1))
        assert result["sufficiency"] == "sufficient"

    def test_llm_exception_defaults_to_sufficient(self):
        m = MagicMock()
        m.invoke_text.side_effect = RuntimeError("down")
        node = make_sufficiency_node(m)
        result = node(_state())
        assert result["sufficiency"] == "sufficient"
