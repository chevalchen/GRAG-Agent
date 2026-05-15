from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace

from langgraph.graph import END, START, StateGraph

from src.app.config import GraphRAGConfig
from src.app.online_qa.checkpointer import get_checkpointer
from src.app.online_qa.nodes.answer_node import make_answer_node
from src.app.online_qa.nodes.fuse_node import make_fuse_node
from src.app.online_qa.nodes.graph_enrich_node import make_graph_enrich_node
from src.app.online_qa.nodes.milvus_fetch_node import make_milvus_fetch_node
from src.app.online_qa.nodes.milvus_search_node import make_milvus_search_node
from src.app.online_qa.nodes.neo4j_search_node import make_neo4j_search_node
from src.app.online_qa.nodes.rerank_node import make_rerank_node
from src.app.online_qa.nodes.router_node import make_router_node
from src.app.online_qa.nodes.sufficiency_node import make_sufficiency_node
from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import BM25Tool
from src.core.tools.vector.milvus_tool import MilvusVectorTool


@asynccontextmanager
async def online_qa_lifespan(config: GraphRAGConfig):
    neo4j_tool = Neo4jGraphTool(config)
    milvus_tool = MilvusVectorTool(config)
    _ = milvus_tool.load_collection()
    bm25_tool = BM25Tool(
        milvus_tool=milvus_tool,
        cache_path=getattr(config, "bm25_cache_path", ".cache/bm25_index.pkl"),
    )
    llm_tool = LLMGenerationTool(config)
    try:
        yield {
            "bm25_tool": bm25_tool,
            "milvus_tool": milvus_tool,
            "neo4j_tool": neo4j_tool,
            "llm_tool": llm_tool,
        }
    finally:
        return


def _route_intent(state: OnlineQAState) -> str:
    """Conditional edge: router → retrieval branch or answer (for chat / loop-limit)."""
    max_iters = 1
    iteration_count = int(state.get("iteration_count") or 0)
    if iteration_count > max_iters:
        # Exceeded loop limit — force generate regardless of intent
        return "answer"
    intent = str((state.get("routing") or {}).get("intent") or "symptom_to_drug")
    if intent == "chat":
        return "answer"
    if intent == "conditions_exact":
        return "neo4j_search"
    # symptom_to_drug and theory both go to milvus_search
    return "milvus_search"


def _route_sufficiency(state: OnlineQAState) -> str:
    """Conditional edge: sufficiency → answer or back to router for one more retrieval pass."""
    max_iters = 1
    iteration_count = int(state.get("iteration_count") or 0)
    verdict = str(state.get("sufficiency") or "sufficient")
    # Allow loop if we haven't exceeded max_iters yet (sufficiency_node already incremented)
    if verdict == "insufficient" and iteration_count <= max_iters:
        return "router"
    return "answer"


def build_graph(
    config: GraphRAGConfig,
    *,
    bm25_tool: BM25Tool,
    milvus_tool: MilvusVectorTool,
    neo4j_tool: Neo4jGraphTool,
    llm_tool: LLMGenerationTool,
    health_milvus_tool: MilvusVectorTool | None = None,  # kept for backwards compat, unused
):
    """
    Build Stage 2 LangGraph workflow.

    Flow:
      START → router
      router → [milvus_search | neo4j_search | answer (chat/loop-limit)]
      milvus_search → graph_enrich → fuse
      neo4j_search → milvus_fetch → fuse
      fuse → rerank → sufficiency
      sufficiency → [answer (sufficient/max-iters) | router (insufficient)]
      answer → END
    """
    top_k = config.top_k
    max_iters: int = getattr(config, "sufficiency_max_iters", 1)

    # Override the loop limit from config in the edge functions
    def _route_intent_cfg(state: OnlineQAState) -> str:
        iteration_count = int(state.get("iteration_count") or 0)
        if iteration_count > max_iters:
            return "answer"
        intent = str((state.get("routing") or {}).get("intent") or "symptom_to_drug")
        if intent == "chat":
            return "answer"
        if intent == "conditions_exact":
            return "neo4j_search"
        return "milvus_search"

    def _route_sufficiency_cfg(state: OnlineQAState) -> str:
        iteration_count = int(state.get("iteration_count") or 0)
        verdict = str(state.get("sufficiency") or "sufficient")
        if verdict == "insufficient" and iteration_count <= max_iters:
            return "router"
        return "answer"

    workflow = StateGraph(OnlineQAState)

    workflow.add_node("router", make_router_node(llm_tool))
    workflow.add_node(
        "milvus_search",
        make_milvus_search_node(
            bm25_tool,
            milvus_tool,
            neo4j_tool,
            top_k=top_k,
            timeout_seconds=getattr(config, "retrieval_timeout_seconds", 1.2),
        ),
    )
    workflow.add_node(
        "graph_enrich",
        make_graph_enrich_node(neo4j_tool, limit=getattr(config, "max_graph_rows", 8)),
    )
    workflow.add_node(
        "neo4j_search",
        make_neo4j_search_node(neo4j_tool, limit=getattr(config, "max_graph_rows", 10)),
    )
    workflow.add_node(
        "milvus_fetch",
        make_milvus_fetch_node(
            milvus_tool,
            fallback_top_k=top_k,
        ),
    )
    workflow.add_node("fuse", make_fuse_node(top_k=top_k))
    workflow.add_node(
        "rerank",
        make_rerank_node(
            getattr(config, "reranker_model", "BAAI/bge-reranker-base"),
            top_k=top_k,
            simple_skip_threshold=getattr(config, "rerank_simple_skip_threshold", 6),
            simple_candidate_limit=getattr(config, "rerank_simple_candidate_limit", 8),
            complex_candidate_limit=getattr(config, "rerank_complex_candidate_limit", 14),
        ),
    )
    workflow.add_node(
        "sufficiency",
        make_sufficiency_node(llm_tool),
    )
    workflow.add_node(
        "answer",
        make_answer_node(
            llm_tool,
            history_window=getattr(config, "history_window", 10),
            simple_context_budget_chars=getattr(config, "simple_context_budget_chars", 1800),
            complex_context_budget_chars=getattr(config, "complex_context_budget_chars", 3600),
            simple_per_doc_chars=getattr(config, "simple_per_doc_chars", 480),
            complex_per_doc_chars=getattr(config, "complex_per_doc_chars", 820),
        ),
    )

    # Edges
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        _route_intent_cfg,
        {"milvus_search": "milvus_search", "neo4j_search": "neo4j_search", "answer": "answer"},
    )
    workflow.add_edge("milvus_search", "graph_enrich")
    workflow.add_edge("graph_enrich", "fuse")
    workflow.add_edge("neo4j_search", "milvus_fetch")
    workflow.add_edge("milvus_fetch", "fuse")
    workflow.add_edge("fuse", "rerank")
    workflow.add_edge("rerank", "sufficiency")
    workflow.add_conditional_edges(
        "sufficiency",
        _route_sufficiency_cfg,
        {"answer": "answer", "router": "router"},
    )
    workflow.add_edge("answer", END)

    return workflow.compile(
        checkpointer=get_checkpointer(getattr(config, "checkpointer_path", ".checkpoints/tcm.db"))
    )
