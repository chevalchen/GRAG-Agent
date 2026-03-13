from __future__ import annotations

from typing import Annotated, Literal

from typing_extensions import TypedDict

from src.core.schemas.document import Document, QueryAnalysis


def append_history(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    if not existing:
        existing = []
    if not new:
        return existing
    return [*existing, *new]


class OnlineQAState(TypedDict, total=False):
    query: str
    history: Annotated[list[dict], append_history]
    analysis: QueryAnalysis | dict
    route: Literal["hybrid", "graph_rag", "combined"]
    hybrid_docs: list[Document]
    graph_docs: list[Document]
    fused_docs: list[Document]
    answer: str | None
    error: str | None
    docs_hybrid: list
    docs_graph: list
    docs_final: list
    metrics: dict
    errors: list[str]
