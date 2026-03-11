import unittest

from langchain_core.documents import Document

from src.legacy.rag_modules.intelligent_query_router import QueryAnalysis, SearchStrategy
from src.app.online_qa.graphs.online_qa_graph import OnlineQAGraph
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


class OnlineQAGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_query_falls_back_to_hybrid(self):
        router = _FakeRouter("graph_rag")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = await graph._graph.ainvoke({
            "query": "",
            "analysis": None,
            "route": None,
            "docs_hybrid": [],
            "docs_graph": [],
            "docs_final": [],
            "answer": None,
            "metrics": {"stream": False},
            "errors": [],
        })
        self.assertEqual(state["route"], "hybrid_traditional")

    async def test_route_hybrid(self):
        router = _FakeRouter("hybrid_traditional")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = await graph.ainvoke("q", stream=False)
        self.assertEqual(state["docs_final"][0].page_content, "H")
        self.assertEqual(state["answer"], "ANSWER:q")

    async def test_route_graph(self):
        router = _FakeRouter("graph_rag")
        hybrid_tool = HybridSearchTool(_FakeHybrid([Document(page_content="H", metadata={})]))
        graph_tool = GraphRAGSearchTool(_FakeGraph([Document(page_content="G", metadata={})]))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = await graph.ainvoke("q", stream=False)
        self.assertEqual(state["docs_final"][0].page_content, "G")

    async def test_route_combined_and_stream_leaves_answer_none(self):
        router = _FakeRouter("combined")
        hybrid_docs = [Document(page_content="T1" * 100, metadata={})]
        graph_docs = [Document(page_content="G1" * 100, metadata={})]
        hybrid_tool = HybridSearchTool(_FakeHybrid(hybrid_docs))
        graph_tool = GraphRAGSearchTool(_FakeGraph(graph_docs))
        answer_tool = AnswerGenerationTool(_FakeGen())
        graph = OnlineQAGraph(router, hybrid_tool, graph_tool, answer_tool, top_k=2)
        state = await graph.ainvoke("q", stream=True)
        self.assertIsNone(state["answer"])
        self.assertEqual(state["docs_final"][0].page_content[:2], "G1")
