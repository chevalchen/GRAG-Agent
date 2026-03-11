from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document

from src.legacy.rag_modules.intelligent_query_router import QueryAnalysis


class OnlineQAState(TypedDict):
    query: str
    analysis: Optional[QueryAnalysis]
    route: Optional[str]
    docs_hybrid: List[Document]
    docs_graph: List[Document]
    docs_final: List[Document]
    answer: Optional[str]
    metrics: Dict[str, Any]
    errors: List[str]
