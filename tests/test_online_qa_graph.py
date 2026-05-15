"""
Tests for the Stage 2 online-QA LangGraph workflow.

Architecture under test:
  START → router → [milvus_search → graph_enrich → fuse | neo4j_search → milvus_fetch → fuse]
       → rerank → sufficiency → [answer | router (loop ≤1)]
  answer → END

All graph traversals use ainvoke() because answer_node is async.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.app.config import DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.cli import _delete_session, _list_history_sessions
from src.app.online_qa.graphs.online_qa_graph import build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_llm(answer_text: str = "ANSWER"):
    """
    Build a mock LLMGenerationTool.

    invoke_text discriminates calls by prompt content:
      - router prompt  : contains "symptom_to_drug" (one of the listed valid intents)
      - sufficiency    : contains "sufficient" (English keyword in the prompt)
      - suggest        : everything else (post-answer follow-up generation)

    llm._llm.astream() is an async generator yielding a single mock chunk.
    """
    llm = MagicMock()

    def _invoke_text(prompt: str) -> str:
        if "symptom_to_drug" in prompt:
            # Router call — return valid intent JSON
            return (
                '{"intent":"symptom_to_drug","complexity_level":"simple",'
                '"keywords":["测试"],"reason":"test"}'
            )
        if "sufficient" in prompt:
            # Sufficiency judge call
            return "sufficient"
        # Suggest / next_questions call
        return '["追问1","追问2","追问3"]'

    llm.invoke_text.side_effect = _invoke_text

    # answer_node calls llm._llm.astream() for token streaming
    async def _astream(messages):
        yield SimpleNamespace(content=answer_text)

    llm._llm = MagicMock()
    llm._llm.astream = _astream
    return llm


def _make_stub_tools(answer_text: str = "ANSWER", docs: list[Document] | None = None):
    docs = docs or [
        Document(page_content="H", metadata={"node_id": "n1", "chunk_id": "c1", "doc_type": "drug"})
    ]
    llm = _make_stub_llm(answer_text)

    bm25 = MagicMock()
    bm25.invoke.return_value = docs

    milvus = MagicMock()
    milvus.invoke.return_value = docs

    neo4j = MagicMock()
    neo4j.invoke.return_value = []
    neo4j._graph = MagicMock()
    neo4j._graph.query.return_value = []

    return llm, bm25, milvus, neo4j


def _build(answer_text: str = "ANSWER", docs: list[Document] | None = None):
    config = GraphRAGConfig(**{**DEFAULT_CONFIG.to_dict(), "top_k": 3})
    llm, bm25, milvus, neo4j = _make_stub_tools(answer_text, docs)
    graph = build_graph(config, bm25_tool=bm25, milvus_tool=milvus, neo4j_tool=neo4j, llm_tool=llm)
    return graph, llm, bm25, milvus, neo4j


def _run(coro):
    """Run an async coroutine from sync test code."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Basic forward flow (symptom_to_drug → milvus_search → ... → answer)
# ---------------------------------------------------------------------------

class TestBasicQAFlow(unittest.TestCase):
    def test_retrieval_tools_are_called(self):
        graph, llm, bm25, milvus, neo4j = _build()
        _run(graph.ainvoke(
            {"query": "红烧肉怎么做", "metrics": {}},
            config={"configurable": {"thread_id": "t-retrieval"}},
        ))
        self.assertTrue(bm25.invoke.called or milvus.invoke.called)

    def test_docs_final_is_populated(self):
        graph, *_ = _build()
        state = _run(graph.ainvoke(
            {"query": "乌鸡白凤丸的适应症", "metrics": {}},
            config={"configurable": {"thread_id": "t-docs"}},
        ))
        self.assertGreater(len(state.get("docs_final") or []), 0)

    def test_answer_is_populated(self):
        graph, *_ = _build(answer_text="答：这是测试答案。")
        state = _run(graph.ainvoke(
            {"query": "红烧肉", "metrics": {}},
            config={"configurable": {"thread_id": "t-answer"}},
        ))
        self.assertEqual(state["answer"], "答：这是测试答案。")

    def test_metrics_contains_total_seconds(self):
        graph, *_ = _build()
        state = _run(graph.ainvoke(
            {"query": "时延指标测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-metrics"}},
        ))
        metrics = state.get("metrics") or {}
        self.assertIn("total_seconds", metrics)
        self.assertGreaterEqual(float(metrics["total_seconds"]), 0.0)


# ---------------------------------------------------------------------------
# Chat intent: routes directly to answer without retrieval
# ---------------------------------------------------------------------------

class TestChatIntent(unittest.TestCase):
    def _build_chat_graph(self):
        config = GraphRAGConfig(**{**DEFAULT_CONFIG.to_dict(), "top_k": 3})

        llm = MagicMock()

        def _invoke_text_chat(prompt: str) -> str:
            if "symptom_to_drug" in prompt:
                return (
                    '{"intent":"chat","complexity_level":"simple",'
                    '"keywords":[],"reason":"greeting"}'
                )
            return "sufficient"

        llm.invoke_text.side_effect = _invoke_text_chat

        async def _astream(messages):
            yield SimpleNamespace(content="你好！有什么可以帮您？")

        llm._llm = MagicMock()
        llm._llm.astream = _astream

        bm25 = MagicMock()
        bm25.invoke.return_value = []
        milvus = MagicMock()
        milvus.invoke.return_value = []
        neo4j = MagicMock()
        neo4j._graph = MagicMock()
        neo4j._graph.query.return_value = []

        graph = build_graph(
            config,
            bm25_tool=bm25,
            milvus_tool=milvus,
            neo4j_tool=neo4j,
            llm_tool=llm,
        )
        return graph, bm25, milvus

    def test_chat_skips_retrieval(self):
        """chat intent → answer directly, no bm25/milvus calls."""
        graph, bm25, milvus = self._build_chat_graph()
        _run(graph.ainvoke(
            {"query": "你好", "metrics": {}},
            config={"configurable": {"thread_id": "t-chat"}},
        ))
        bm25.invoke.assert_not_called()
        milvus.invoke.assert_not_called()

    def test_chat_answer_is_not_empty(self):
        graph, *_ = self._build_chat_graph()
        state = _run(graph.ainvoke(
            {"query": "你好", "metrics": {}},
            config={"configurable": {"thread_id": "t-chat-ans"}},
        ))
        # chat with no docs → early-exit "请先输入问题" or "没有检索到" message
        self.assertIsNotNone(state.get("answer"))
        self.assertGreater(len(state["answer"]), 0)


# ---------------------------------------------------------------------------
# Multi-turn memory
# ---------------------------------------------------------------------------

class TestMemory(unittest.TestCase):
    def test_history_accumulates_across_turns(self):
        graph, *_ = _build()
        sid = "t-memory-accum"
        _run(graph.ainvoke(
            {"query": "第一问", "metrics": {}},
            config={"configurable": {"thread_id": sid}},
        ))
        state = _run(graph.ainvoke(
            {"query": "第二问", "metrics": {}},
            config={"configurable": {"thread_id": sid}},
        ))
        history = state.get("history") or []
        self.assertGreaterEqual(len(history), 4)
        roles = [h["role"] for h in history]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_history_roles_are_correct(self):
        graph, *_ = _build()
        state = _run(graph.ainvoke(
            {"query": "角色测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-roles"}},
        ))
        history = state.get("history") or []
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(history[-2]["role"], "user")
        self.assertEqual(history[-1]["role"], "assistant")


# ---------------------------------------------------------------------------
# Routing signal shape
# ---------------------------------------------------------------------------

class TestRoutingField(unittest.TestCase):
    def test_routing_has_intent_key(self):
        graph, *_ = _build()
        state = _run(graph.ainvoke(
            {"query": "分析字段测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-routing-field"}},
        ))
        routing = state.get("routing")
        self.assertIsNotNone(routing)
        self.assertIn(
            routing.get("intent"),
            {"chat", "theory", "symptom_to_drug", "conditions_exact"},
        )

    def test_empty_query_returns_safe_answer(self):
        graph, *_ = _build()
        state = _run(graph.ainvoke(
            {"query": "", "metrics": {}},
            config={"configurable": {"thread_id": "t-empty-query"}},
        ))
        # answer_node detects empty query and returns a polite message
        answer = state.get("answer") or ""
        self.assertGreater(len(answer), 0)


# ---------------------------------------------------------------------------
# Session management (pure SQLite — unchanged from original)
# ---------------------------------------------------------------------------

class TestSessionManagement(unittest.TestCase):
    def _make_db(self) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        con = sqlite3.connect(tmp.name)
        cur = con.cursor()
        cur.execute(
            """CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT, checkpoint BLOB, metadata BLOB
            )"""
        )
        cur.execute(
            """CREATE TABLE writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT, value BLOB
            )"""
        )
        for row in [("s1", "c1"), ("s2", "c2"), ("s1", "c3")]:
            cur.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', ?)",
                row,
            )
        cur.execute(
            "INSERT INTO writes(thread_id,checkpoint_ns,checkpoint_id,task_id,idx,channel)"
            " VALUES ('s1','','c3','t',0,'x')"
        )
        con.commit()
        con.close()
        return tmp.name

    def test_list_sessions_returns_most_recent_first(self):
        db = self._make_db()
        try:
            sessions = _list_history_sessions(db_path=db)
            self.assertEqual(sessions[0][0], "s1")
            self.assertEqual(sessions[0][1], "c3")
        finally:
            os.unlink(db)

    def test_delete_session_removes_from_list(self):
        db = self._make_db()
        try:
            self.assertTrue(_delete_session(db_path=db, session_id="s1"))
            sessions = _list_history_sessions(db_path=db)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0][0], "s2")
        finally:
            os.unlink(db)


if __name__ == "__main__":
    unittest.main()
