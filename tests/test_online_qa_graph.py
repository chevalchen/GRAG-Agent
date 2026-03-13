import unittest
import os
import sqlite3
import tempfile

from langchain_core.documents import Document

from src.legacy.rag_modules.intelligent_query_router import QueryAnalysis, SearchStrategy
from src.app.online_qa.graphs.online_qa_graph import OnlineQAGraph
from src.app.online_qa.cli import _delete_session, _list_history_sessions
from src.app.online_qa.tools.answer_generation import AnswerGenerationTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool


class _FakeRouter:
    def __init__(self, strategy: str):
        self._strategy = strategy

    def analyze_query(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            query_complexity=0.1,
            relationship_intensity=0.1,
            reasoning_required=False,
            entity_count=1,
            recommended_strategy=SearchStrategy(self._strategy),
            confidence=0.9,
            reasoning="test",
        )

    def post_process_results(self, documents, analysis):
        for d in documents:
            d.metadata["route_strategy"] = analysis.recommended_strategy.value
        return documents


class _FakeHybrid:
    def __init__(self, docs):
        self._docs = docs

    def hybrid_search(self, query, top_k):
        return self._docs[:top_k]


class _FakeGraph:
    def __init__(self, docs):
        self._docs = docs

    def graph_rag_search(self, query, top_k):
        return self._docs[:top_k]


class _FakeGen:
    def generate_adaptive_answer(self, question, documents):
        return "ANSWER:" + question

    def generate_adaptive_answer_stream(self, question, documents, max_retries=3):
        yield "A"
        yield "B"


class OnlineQAGraphTests(unittest.TestCase):
    def test_history_persists_with_same_session_id(self):
        router = _FakeRouter("hybrid_traditional")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        graph.invoke("第一问", stream=False, session_id="t-history")
        state = graph.invoke("第二问", stream=False, session_id="t-history")
        history = state.get("history", [])
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(history[-2]["role"], "user")
        self.assertEqual(history[-1]["role"], "assistant")

    def test_missing_query_falls_back_to_hybrid(self):
        router = _FakeRouter("graph_rag")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = graph.invoke("", stream=False, session_id="t-missing")
        self.assertEqual(state["route"], "hybrid")

    def test_route_hybrid(self):
        router = _FakeRouter("hybrid_traditional")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = graph.invoke("q", stream=False, session_id="t-hybrid")
        self.assertEqual(state["docs_final"][0].page_content, "H")
        self.assertEqual(state["answer"], "ANSWER:q")

    def test_route_graph(self):
        router = _FakeRouter("graph_rag")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = graph.invoke("q", stream=False, session_id="t-graph")
        self.assertEqual(state["docs_final"][0].page_content, "G")

    def test_route_combined_and_stream_leaves_answer_none(self):
        router = _FakeRouter("combined")
        hybrid_docs = [Document(page_content="T1" * 100, metadata={})]
        graph_docs = [Document(page_content="G1" * 100, metadata={})]
        hybrid_tool = HybridSearchTool(_FakeHybrid(hybrid_docs))
        graph_tool = GraphRAGSearchTool(_FakeGraph(graph_docs))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = graph.invoke("q", stream=True, session_id="t-combined")
        self.assertIsNone(state["answer"])
        self.assertEqual(state["docs_final"][0].page_content[:2], "G1")


class OnlineQACliSessionsTests(unittest.TestCase):
    def test_list_and_delete_sessions_from_sqlite(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        try:
            con = sqlite3.connect(tmp.name)
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB
                )
                """
            )
            cur.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', ?)",
                ("s1", "c1"),
            )
            cur.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', ?)",
                ("s2", "c2"),
            )
            cur.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id) VALUES (?, '', ?)",
                ("s1", "c3"),
            )
            cur.execute(
                "INSERT INTO writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel) VALUES (?, '', ?, ?, ?, ?)",
                ("s1", "c3", "t", 0, "x"),
            )
            con.commit()
            con.close()

            sessions = _list_history_sessions(db_path=tmp.name)
            self.assertEqual(sessions[0][0], "s1")
            self.assertEqual(sessions[0][1], "c3")
            self.assertEqual(sessions[1][0], "s2")

            self.assertTrue(_delete_session(db_path=tmp.name, session_id="s1"))
            sessions2 = _list_history_sessions(db_path=tmp.name)
            self.assertEqual(len(sessions2), 1)
            self.assertEqual(sessions2[0][0], "s2")
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
