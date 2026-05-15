from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.documents import Document
from typing_extensions import TypedDict


class RoutingSignal(TypedDict, total=False):
    intent: str            # chat | theory | symptom_to_drug | conditions_exact
    complexity_level: str  # simple | complex
    keywords: list[str]
    reason: str


class OnlineQAState(TypedDict, total=False):
    # --- input ---
    query: str
    rewritten_query: str
    extracted_entities: dict[str, list[str]]  # {drugs:[], diseases:[], symptoms:[]}

    # --- routing ---
    routing: RoutingSignal | None

    # --- retrieval bridges ---
    kg_anchor_id: str | None       # forward: MilvusSearch → GraphEnrich
    milvus_doc_ids: list[str]      # reverse: Neo4jSearch → MilvusFetch

    # --- retrieval results ---
    milvus_docs: list[Document]
    graph_docs: list[Document]

    # --- fusion / rerank ---
    docs_rrf: list[Document]
    docs_final: list[Document]

    # --- control flow ---
    sufficiency: str | None        # "sufficient" | "insufficient"
    iteration_count: int           # starts 0; incremented on insufficient
    safety_blocked: bool
    safety_reason: str

    # --- output ---
    answer: str
    next_questions: list[str]

    # --- multi-turn ---
    history: Annotated[list[dict], operator.add]
    metrics: Annotated[dict, operator.or_]
    error: str | None
