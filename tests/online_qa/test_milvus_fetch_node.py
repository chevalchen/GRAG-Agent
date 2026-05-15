"""Tests for milvus_fetch_node: point-get filter shape and fallback behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.app.online_qa.nodes.milvus_fetch_node import make_milvus_fetch_node
from src.app.online_qa.state import OnlineQAState


def _milvus_with_client(rows: list[dict]) -> MagicMock:
    """Build a MilvusVectorTool mock whose .client.query() returns rows."""
    milvus = MagicMock()
    milvus.client = MagicMock()
    milvus.client.query.return_value = rows
    milvus._collection_name = "tcm_knowledge"
    return milvus


def _state(**kwargs) -> OnlineQAState:
    base: OnlineQAState = {
        "query": "乌鸡白凤丸适应症",
        "rewritten_query": "乌鸡白凤丸适应症",
        "milvus_doc_ids": ["node-001", "node-002"],
        "metrics": {},
    }
    base.update(kwargs)  # type: ignore[arg-type]
    return base


class TestMilvusFetchNode:
    def test_point_get_builds_correct_filter(self):
        milvus = _milvus_with_client([
            {"chunk_id": "c1", "node_id": "node-001", "doc_type": "drug", "text": "药品说明文本1"},
            {"chunk_id": "c2", "node_id": "node-002", "doc_type": "drug", "text": "药品说明文本2"},
        ])
        node = make_milvus_fetch_node(milvus)
        result = node(_state())

        call_args = milvus.client.query.call_args
        expr = call_args[1]["filter"] if call_args[1] else call_args[0][1]
        assert "node-001" in expr
        assert "node-002" in expr
        assert len(result["milvus_docs"]) == 2
        assert result["metrics"]["milvus_fetch_strategy"] == "point_get"

    def test_empty_doc_ids_falls_back_to_dense(self):
        milvus = MagicMock()
        milvus.invoke.return_value = [Document(page_content="fallback doc")]
        milvus.client = None
        node = make_milvus_fetch_node(milvus, fallback_top_k=3)
        result = node(_state(milvus_doc_ids=[]))

        assert result["metrics"]["milvus_fetch_strategy"] == "fallback_dense"
        assert len(result["milvus_docs"]) == 1

    def test_client_exception_falls_back_to_dense(self):
        milvus = MagicMock()
        milvus.client = MagicMock()
        milvus.client.query.side_effect = RuntimeError("connection error")
        milvus._collection_name = "tcm_knowledge"
        milvus.invoke.return_value = [Document(page_content="fallback")]
        node = make_milvus_fetch_node(milvus, fallback_top_k=3)
        result = node(_state())
        assert result["metrics"]["milvus_fetch_strategy"] == "fallback_dense"

    def test_rows_without_text_are_skipped(self):
        milvus = _milvus_with_client([
            {"chunk_id": "c1", "node_id": "n1", "doc_type": "drug", "text": ""},
            {"chunk_id": "c2", "node_id": "n2", "doc_type": "drug", "text": "有效文本"},
        ])
        node = make_milvus_fetch_node(milvus)
        result = node(_state())
        assert len(result["milvus_docs"]) == 1
        assert result["milvus_docs"][0].page_content == "有效文本"
