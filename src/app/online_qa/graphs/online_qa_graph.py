from types import SimpleNamespace

from langgraph.graph import END, START, StateGraph

from src.app.online_qa.checkpointer import get_checkpointer
from src.app.online_qa.nodes import OnlineQARuntime, set_runtime
from src.app.online_qa.nodes.answer_node import answer_node
from src.app.online_qa.nodes.fuse_node import fuse_node
from src.app.online_qa.nodes.graph_retrieve_node import graph_retrieve_node
from src.app.online_qa.nodes.hybrid_retrieve_node import hybrid_retrieve_node
from src.app.online_qa.nodes.query_analysis_node import query_analysis_node
from src.app.online_qa.nodes.route_node import route_node
from src.app.online_qa.state import OnlineQAState
from src.app.online_qa.tools.answer_generation import AnswerGenerationTool
from src.app.online_qa.tools.fusion import FusionTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool
from src.app.online_qa.tools.query_analysis import QueryAnalysisTool


def route_condition(state: OnlineQAState) -> list[str] | str:
    route = state.get("route", "hybrid")
    if route == "hybrid":
        return "hybrid"
    if route == "graph_rag":
        return "graph_rag"
    if route == "combined":
        return ["combined_hybrid", "combined_graph"]
    return "hybrid"


def build_graph():
    graph = StateGraph(OnlineQAState)
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("route", route_node)
    graph.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph.add_node("graph_retrieve", graph_retrieve_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "query_analysis")
    graph.add_edge("query_analysis", "route")
    graph.add_conditional_edges(
        "route",
        route_condition,
        {
            "hybrid": "hybrid_retrieve",
            "graph_rag": "graph_retrieve",
            "combined_hybrid": "hybrid_retrieve",
            "combined_graph": "graph_retrieve",
        },
    )
    graph.add_edge("hybrid_retrieve", "fuse")
    graph.add_edge("graph_retrieve", "fuse")
    graph.add_edge("fuse", "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=get_checkpointer())


class OnlineQAGraph:
    def __init__(
        self,
        router,
        hybrid_tool: HybridSearchTool,
        graph_tool: GraphRAGSearchTool,
        answer_tool: AnswerGenerationTool,
        top_k: int,
        llm_concurrency: int = 2,
        retrieve_concurrency: int = 4,
    ):
        set_runtime(
            OnlineQARuntime(
                query_analysis_tool=QueryAnalysisTool(router),
                hybrid_tool=hybrid_tool,
                graph_tool=graph_tool,
                fusion_tool=FusionTool(),
                answer_tool=answer_tool,
                top_k=top_k,
            )
        )
        self._graph = build_graph()

    def invoke(
        self,
        query: str,
        stream: bool = False,
        session_id: str = "default",
        history: list[dict] | None = None,
    ) -> OnlineQAState:
        state: OnlineQAState = {
            "query": query,
            "metrics": {"stream": stream},
        }
        if history:
            state["history"] = history
        config = {"configurable": {"thread_id": session_id}}
        result = self._graph.invoke(state, config=config)
        analysis = result.get("analysis")
        if isinstance(analysis, dict):
            strategy = analysis.get("recommended_strategy", "hybrid")
            result["analysis"] = SimpleNamespace(
                original_query=analysis.get("original_query", query),
                route=analysis.get("route", "hybrid"),
                keywords=analysis.get("keywords", []),
                intent=analysis.get("intent", ""),
                query_complexity=float(analysis.get("query_complexity", 0.0)),
                relationship_intensity=float(analysis.get("relationship_intensity", 0.0)),
                recommended_strategy=SimpleNamespace(value=strategy),
            )
        return result
