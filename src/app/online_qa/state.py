from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.documents import Document
from typing_extensions import TypedDict

class RoutingSignal(TypedDict):
    query_intent: str
    complexity_level: str
    use_graph: bool
    use_drug_vec: bool
    use_lit_vec: bool
    use_health_vec: bool
    source_hint: list[str]
    keywords: list[str]
    reason: str


class ResolvedDrug(TypedDict, total=False):
    canonical_name: str
    normalized_name: str
    node_id: str
    matched_alias: str
    match_type: str


class OnlineQAState(TypedDict, total=False):
    """
    在线问答状态
    
    Attributes:
        query: 用户查询
        analysis: 查询分析
        hybrid_docs: 混合检索文档
        graph_docs: 图检索文档
        docs_final: 最终文档
        metrics: 指标
        answer: 答案
        history: 对话历史
        error: 错误信息
    """
    query: str
    routing: RoutingSignal | None
    resolved_drug: ResolvedDrug | None
    graph_docs: list[Document]
    drug_docs: list[Document]
    lit_docs: list[Document]
    health_docs: list[Document]
    docs_rrf: list[Document]
    docs_final: list[Document]
    top_k: int
    metrics: Annotated[dict, operator.or_]
    answer: str
    history: Annotated[list[dict], operator.add]
    error: str | None
