from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis


class GraphState(TypedDict):
    query: str
    analysis: Optional[QueryAnalysis]
    documents: List[Document]


class GraphRAGWorkflow:
    def __init__(self, router: IntelligentQueryRouter, config):
        self.router = router
        self.config = config
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("post_process", self._post_process)
        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "retrieve")
        graph.add_edge("retrieve", "post_process")
        graph.add_edge("post_process", END)
        return graph.compile()

    def _analyze(self, state: GraphState):
        analysis = self.router.analyze_query(state["query"])
        return {"analysis": analysis}

    def _retrieve(self, state: GraphState):
        analysis = state["analysis"]
        documents = self.router.retrieve_with_analysis(state["query"], analysis, self.config.top_k)
        return {"documents": documents}

    def _post_process(self, state: GraphState):
        analysis = state["analysis"]
        documents = self.router.post_process_results(state["documents"], analysis)
        return {"documents": documents}

    def run(self, query: str):
        result = self._graph.invoke({"query": query, "analysis": None, "documents": []})
        return result["documents"], result["analysis"]
