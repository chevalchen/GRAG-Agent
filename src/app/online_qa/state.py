from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.documents import Document
from typing_extensions import TypedDict

from src.core.schemas.document import QueryAnalysis


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
    analysis: QueryAnalysis | None
    hybrid_docs: list[Document]
    graph_docs: list[Document]
    docs_final: list[Document]
    metrics: dict
    answer: str
    history: Annotated[list[dict], operator.add]
    error: str | None
