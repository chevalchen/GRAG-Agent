import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.app.config import DEFAULT_CONFIG, GraphRAGConfig
from src.app.online_qa.cli import _delete_session, _list_history_sessions
from src.app.online_qa.graphs.online_qa_graph import build_graph
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def _make_stub_tools(
    supervisor_strategy: str = "hybrid",
    hybrid_docs: list[Document] | None = None,
    graph_docs: list[Document] | None = None,
    answer_text: str = "ANSWER",
):
    hybrid_docs = hybrid_docs or [Document(page_content="H", metadata={"node_id": "n1", "chunk_id": "c1", "doc_type": "drug"})]
    graph_docs = graph_docs or [
        Document(page_content="G", metadata={"node_id": "n2", "chunk_id": "c2", "doc_type": "drug"})
    ]

    llm = MagicMock(spec=LLMGenerationTool)

    def _invoke_text(prompt: str) -> str:
        if "query_intent" in prompt:
            if supervisor_strategy == "graph_rag":
                return '{"query_intent":"symptom_disease","use_graph":true,"use_drug_vec":false,"use_lit_vec":false,"use_health_vec":false,"source_hint":[],"keywords":["测试"],"reason":"test"}'
            if supervisor_strategy == "combined":
                return '{"query_intent":"symptom_disease","use_graph":true,"use_drug_vec":true,"use_lit_vec":false,"use_health_vec":false,"source_hint":[],"keywords":["测试"],"reason":"test"}'
            return '{"query_intent":"drug_specific","use_graph":false,"use_drug_vec":true,"use_lit_vec":false,"use_health_vec":false,"source_hint":[],"keywords":["测试"],"reason":"test"}'
        return answer_text

    llm.invoke_text.side_effect = _invoke_text

    bm25 = MagicMock(spec=BM25Tool)
    bm25.invoke.return_value = hybrid_docs

    milvus = MagicMock(spec=MilvusVectorTool)
    milvus.invoke.return_value = hybrid_docs

    neo4j = MagicMock(spec=Neo4jGraphTool)
    neo4j.invoke.return_value = graph_docs
    neo4j._graph = MagicMock()
    neo4j._graph.query.return_value = [
        {"node_id": d.metadata.get("node_id", ""), "drug_name": d.page_content, "ingredients": []} for d in graph_docs
    ]

    return llm, bm25, milvus, neo4j


def _build(
    supervisor_strategy: str = "hybrid",
    hybrid_docs: list[Document] | None = None,
    graph_docs: list[Document] | None = None,
    answer_text: str = "ANSWER",
):
    config = GraphRAGConfig(**{**DEFAULT_CONFIG.to_dict(), "top_k": 3})
    llm, bm25, milvus, neo4j = _make_stub_tools(supervisor_strategy, hybrid_docs, graph_docs, answer_text)
    graph = build_graph(config, bm25_tool=bm25, milvus_tool=milvus, neo4j_tool=neo4j, llm_tool=llm)
    return graph, llm, bm25, milvus, neo4j


class TestRouteHybrid(unittest.TestCase):
    def test_hybrid_strategy_uses_bm25_and_milvus(self):
        graph, llm, bm25, milvus, neo4j = _build("hybrid")
        state = graph.invoke(
            {"query": "红烧肉怎么做", "metrics": {}},
            config={"configurable": {"thread_id": "t-hybrid"}},
        )
        self.assertTrue(bm25.invoke.called or milvus.invoke.called)
        self.assertIn("H", state["docs_final"][0].page_content)

    def test_hybrid_answer_is_populated(self):
        graph, *_ = _build("hybrid", answer_text="答：红烧肉做法")
        state = graph.invoke(
            {"query": "红烧肉", "metrics": {}},
            config={"configurable": {"thread_id": "t-hybrid-ans"}},
        )
        self.assertEqual(state["answer"], "答：红烧肉做法")

    def test_metrics_contains_total_seconds(self):
        graph, *_ = _build("hybrid")
        state = graph.invoke(
            {"query": "时延指标测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-metrics-total"}},
        )
        metrics = state.get("metrics") or {}
        self.assertIn("total_seconds", metrics)
        self.assertGreaterEqual(float(metrics["total_seconds"]), 0.0)


class TestRouteGraphRAG(unittest.TestCase):
    def test_graph_rag_strategy_uses_neo4j(self):
        graph, llm, bm25, milvus, neo4j = _build("graph_rag")
        state = graph.invoke(
            {"query": "猪肉和豆腐有什么关系", "metrics": {}},
            config={"configurable": {"thread_id": "t-graph"}},
        )
        neo4j._graph.query.assert_called()
        self.assertIn("G", state["docs_final"][0].page_content)


class TestRouteCombined(unittest.TestCase):
    def test_combined_fuses_both_sources(self):
        h_docs = [Document(page_content="H" * 50, metadata={"node_id": "h1", "chunk_id": "ch1", "recipe_name": "H菜"})]
        g_docs = [Document(page_content="G" * 50, metadata={"node_id": "g1", "chunk_id": "cg1", "recipe_name": "G菜"})]
        graph, *_ = _build("combined", hybrid_docs=h_docs, graph_docs=g_docs)
        state = graph.invoke(
            {"query": "综合检索测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-combined"}},
        )
        contents = [d.page_content for d in state["docs_final"]]
        self.assertTrue(any("H" in c for c in contents))
        self.assertTrue(any("G" in c for c in contents))


class TestMemory(unittest.TestCase):
    def test_history_accumulates_across_turns(self):
        graph, *_ = _build("hybrid")
        sid = "t-memory"
        graph.invoke({"query": "第一问", "metrics": {}}, config={"configurable": {"thread_id": sid}})
        state = graph.invoke({"query": "第二问", "metrics": {}}, config={"configurable": {"thread_id": sid}})
        history = state.get("history") or []
        self.assertGreaterEqual(len(history), 4)
        roles = [h["role"] for h in history]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_history_roles_are_correct(self):
        graph, *_ = _build("hybrid")
        state = graph.invoke(
            {"query": "角色测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-roles"}},
        )
        history = state.get("history") or []
        self.assertTrue(len(history) >= 2)
        self.assertEqual(history[-2]["role"], "user")
        self.assertEqual(history[-1]["role"], "assistant")


class TestAnalysisField(unittest.TestCase):
    def test_analysis_is_routing_signal(self):
        graph, *_ = _build("hybrid")
        state = graph.invoke(
            {"query": "分析字段测试", "metrics": {}},
            config={"configurable": {"thread_id": "t-analysis"}},
        )
        routing = state.get("routing")
        self.assertIsNotNone(routing)
        self.assertIn(routing.get("query_intent"), {"drug_specific", "symptom_disease", "tcm_theory", "clinical_case", "health_advice"})

    def test_empty_query_returns_default_routing(self):
        graph, *_ = _build("hybrid")
        state = graph.invoke(
            {"query": "", "metrics": {}},
            config={"configurable": {"thread_id": "t-empty"}},
        )
        routing = state.get("routing")
        self.assertIsNotNone(routing)
        self.assertEqual(routing.get("query_intent"), "tcm_theory")


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
