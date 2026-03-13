from typing import Type

from src.app.online_qa.tools.fusion import FusionTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool
from src.app.online_qa.tools.query_analysis import QueryAnalysisTool
from src.core.tools.llm.generation_tool import GenerationTool

AGENT_TOOL_REGISTRY: dict[str, list[Type]] = {
    "query_analysis_node": [QueryAnalysisTool],
    "hybrid_retrieve_node": [HybridSearchTool],
    "graph_retrieve_node": [GraphRAGSearchTool],
    "fuse_node": [FusionTool],
    "answer_node": [GenerationTool],
    "route_node": [],
}


def assert_tool_allowed(node_name: str, tool_cls: Type) -> None:
    allowed = AGENT_TOOL_REGISTRY.get(node_name, [])
    if tool_cls not in allowed:
        raise PermissionError(
            f"Node '{node_name}' is not allowed to use tool '{tool_cls.__name__}'. "
            f"Allowed tools: {[t.__name__ for t in allowed]}"
        )

