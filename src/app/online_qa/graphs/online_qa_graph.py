from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.graph import END, START, StateGraph

from src.app.config import GraphRAGConfig
from src.app.online_qa.checkpointer import get_checkpointer
from src.app.online_qa.nodes.answer_node import make_answer_node
from src.app.online_qa.nodes.drug_entity_resolve_node import make_drug_entity_resolve_node
from src.app.online_qa.nodes.drug_retrieve_node import make_drug_retrieve_node
from src.app.online_qa.nodes.fuse_node import make_fuse_node
from src.app.online_qa.nodes.graph_retrieve_node import make_graph_retrieve_node
from src.app.online_qa.nodes.health_retrieve_node import make_health_retrieve_node
from src.app.online_qa.nodes.lit_retrieve_node import make_lit_retrieve_node
from src.app.online_qa.nodes.rerank_node import make_rerank_node
from src.app.online_qa.nodes.supervisor_node import make_supervisor_node
from src.app.online_qa.state import OnlineQAState
from src.core.tools.graph.neo4j_tool import Neo4jGraphTool
from src.core.tools.llm.generation_tool import LLMGenerationTool
from src.core.tools.retrieval.bm25_tool import BM25Tool, build_bm25_from_milvus
from src.core.tools.vector.milvus_tool import MilvusVectorTool


@asynccontextmanager
async def online_qa_lifespan(config: GraphRAGConfig):
    neo4j_tool = Neo4jGraphTool(config)
    milvus_tool = MilvusVectorTool(config)
    _ = milvus_tool.load_collection()
    bm25_tool = build_bm25_from_milvus(
        milvus_tool.client,
        top_k=getattr(config, "bm25_top_k", config.top_k),
        collection_name=milvus_tool.collection_name,
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


def build_graph(
    config: GraphRAGConfig,
    *,
    bm25_tool: BM25Tool,
    milvus_tool: MilvusVectorTool,
    neo4j_tool: Neo4jGraphTool,
    llm_tool: LLMGenerationTool,
):
    """
    构建在线问答图
    
    Args:
        config: 图配置
        bm25_tool: BM25 工具
        milvus_tool: Milvus 向量工具
        neo4j_tool: Neo4j 图工具
        llm_tool: LLM 生成工具
    
    Returns:
        在线问答图
    
    Nodes:
        supervisor: 监督节点
        hybrid_retrieve: 混合检索节点
        graph_retrieve: 图检索节点
        fuse: 融合节点
        answer: 回答节点
        
    Edges:
        START -> supervisor
        supervisor -> hybrid_retrieve
        supervisor -> graph_retrieve
        hybrid_retrieve -> fuse
        graph_retrieve -> fuse
        fuse -> answer
        answer -> END
        
    """
    workflow = StateGraph(OnlineQAState)
    workflow.add_node("supervisor_node", make_supervisor_node(llm_tool))
    workflow.add_node("drug_entity_resolve_node", make_drug_entity_resolve_node(neo4j_tool))
    workflow.add_node(
        "graph_retrieve_node",
        make_graph_retrieve_node(
            neo4j_tool,
            top_k=config.top_k,
            expand_factor=getattr(config, "graph_expand_factor", 3.0),
            max_graph_rows=getattr(config, "max_graph_rows", 24),
            graph_fallback_max_nodes=getattr(config, "graph_fallback_max_nodes", 48),
        ),
    )
    workflow.add_node(
        "drug_retrieve_node",
        make_drug_retrieve_node(
            bm25_tool,
            milvus_tool,
            top_k=config.top_k,
            expand_factor=getattr(config, "retrieve_expand_factor", 2.0),
            max_retrieval_top_k=getattr(config, "max_retrieval_top_k", 12),
            merge_strategy=getattr(config, "retrieval_balance_strategy", "balanced"),
            timeout_seconds=getattr(config, "retrieval_timeout_seconds", 1.2),
        ),
    )
    workflow.add_node(
        "lit_retrieve_node",
        make_lit_retrieve_node(
            milvus_tool,
            top_k=config.top_k,
            expand_factor=getattr(config, "lit_expand_factor", 1.5),
            max_retrieval_top_k=getattr(config, "max_retrieval_top_k", 12),
        ),
    )
    workflow.add_node("health_retrieve_node", make_health_retrieve_node())
    workflow.add_node("fuse_node", make_fuse_node(top_k=config.top_k))
    workflow.add_node(
        "rerank_node",
        make_rerank_node(
            getattr(config, "reranker_model", "BAAI/bge-reranker-base"),
            top_k=config.top_k,
            simple_skip_threshold=getattr(config, "rerank_simple_skip_threshold", 6),
            simple_candidate_limit=getattr(config, "rerank_simple_candidate_limit", 8),
            complex_candidate_limit=getattr(config, "rerank_complex_candidate_limit", 14),
        ),
    )
    workflow.add_node(
        "answer_node",
        make_answer_node(
            llm_tool,
            history_window=getattr(config, "history_window", 10),
            simple_context_budget_chars=getattr(config, "simple_context_budget_chars", 1800),
            complex_context_budget_chars=getattr(config, "complex_context_budget_chars", 3600),
            simple_per_doc_chars=getattr(config, "simple_per_doc_chars", 480),
            complex_per_doc_chars=getattr(config, "complex_per_doc_chars", 820),
        ),
    )

    workflow.add_edge(START, "supervisor_node")
    workflow.add_edge("supervisor_node", "drug_entity_resolve_node")
    workflow.add_edge("drug_entity_resolve_node", "graph_retrieve_node")
    workflow.add_edge("drug_entity_resolve_node", "drug_retrieve_node")
    workflow.add_edge("drug_entity_resolve_node", "lit_retrieve_node")
    workflow.add_edge("drug_entity_resolve_node", "health_retrieve_node")
    workflow.add_edge("graph_retrieve_node", "fuse_node")
    workflow.add_edge("drug_retrieve_node", "fuse_node")
    workflow.add_edge("lit_retrieve_node", "fuse_node")
    workflow.add_edge("health_retrieve_node", "fuse_node")
    workflow.add_edge("fuse_node", "rerank_node")
    workflow.add_edge("rerank_node", "answer_node")
    workflow.add_edge("answer_node", END)
    return workflow.compile(checkpointer=get_checkpointer(getattr(config, "checkpointer_path", ".checkpoints/tcm.db")))
