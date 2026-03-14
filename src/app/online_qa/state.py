from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.documents import Document
from typing_extensions import TypedDict

from src.core.schemas.document import QueryAnalysis


class OnlineQAState(TypedDict, total=False):
    query: str
    analysis: QueryAnalysis | None
    hybrid_docs: list[Document]
    graph_docs: list[Document]
    docs_final: list[Document]
    metrics: dict
    answer: str
    history: Annotated[list[dict], operator.add]
    error: str | None
