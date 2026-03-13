from dataclasses import dataclass
from typing import Optional

from src.app.online_qa.tools.answer_generation import AnswerGenerationTool
from src.app.online_qa.tools.fusion import FusionTool
from src.app.online_qa.tools.graph_rag_search import GraphRAGSearchTool
from src.app.online_qa.tools.hybrid_search import HybridSearchTool
from src.app.online_qa.tools.query_analysis import QueryAnalysisTool


@dataclass
class OnlineQARuntime:
    query_analysis_tool: QueryAnalysisTool
    hybrid_tool: HybridSearchTool
    graph_tool: GraphRAGSearchTool
    fusion_tool: FusionTool
    answer_tool: AnswerGenerationTool
    top_k: int


RUNTIME: Optional[OnlineQARuntime] = None


def set_runtime(runtime: OnlineQARuntime) -> None:
    global RUNTIME
    RUNTIME = runtime

