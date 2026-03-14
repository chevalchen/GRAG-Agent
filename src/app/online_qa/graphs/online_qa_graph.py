from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.app.config import GraphRAGConfig
from src.app.online_qa.checkpointer import get_checkpointer
from src.app.online_qa.nodes.answer_node import make_answer_node
from src.app.online_qa.nodes.fuse_node import make_fuse_node
from src.app.online_qa.nodes.graph_retrieve_node import make_graph_retrieve_node
from src.app.online_qa.nodes.hybrid_retrieve_node import make_hybrid_retrieve_node
from src.app.online_qa.nodes.supervisor_node import make_supervisor_node
from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def route_condition(state: OnlineQAState) -> list[str] | str:
    analysis = state.get("analysis")
    strategy = getattr(analysis, "recommended_strategy", "hybrid") if analysis else "hybrid"
    if strategy == "combined":
        return ["hybrid_retrieve", "graph_retrieve"]
    return "graph_retrieve" if strategy == "graph_rag" else "hybrid_retrieve"


def build_graph(
    config: GraphRAGConfig,
    *,
    bm25_tool: BM25Tool,
    milvus_tool: MilvusVectorTool,
    neo4j_tool: Neo4jGraphTool,
    llm_tool: LLMGenerationTool,
):
    graph = StateGraph(OnlineQAState)
    graph.add_node("supervisor", make_supervisor_node(llm_tool))
    graph.add_node("hybrid_retrieve", make_hybrid_retrieve_node(bm25_tool, milvus_tool, neo4j_tool, top_k=config.top_k))
    graph.add_node("graph_retrieve", make_graph_retrieve_node(neo4j_tool, top_k=config.top_k))
    graph.add_node("fuse", make_fuse_node(top_k=config.top_k))
    graph.add_node("answer", make_answer_node(llm_tool, history_window=getattr(config, "history_window", 10)))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_condition, {"hybrid_retrieve": "hybrid_retrieve", "graph_retrieve": "graph_retrieve"})
    graph.add_edge("hybrid_retrieve", "fuse")
    graph.add_edge("graph_retrieve", "fuse")
    graph.add_edge("fuse", "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=get_checkpointer())
