from dataclasses import dataclass
from typing import Any, Dict

from src.utils import env_utils


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass
class GraphRAGConfig:
    neo4j_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "medrag123"
    neo4j_database: str = "neo4j"

    milvus_host: str = "localhost"
    milvus_port: int = 19531
    milvus_uri: str = "http://localhost:19531"
    milvus_collection_name: str = "tcm_knowledge"
    milvus_dimension: int = 512

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"
    reranker_model: str = "BAAI/bge-reranker-base"
    retrieval_balance_strategy: str = "balanced"
    retrieval_timeout_seconds: float = 1.2
    retrieve_expand_factor: float = 2.0
    lit_expand_factor: float = 1.5
    graph_expand_factor: float = 3.0
    max_retrieval_top_k: int = 12
    max_graph_rows: int = 24
    graph_fallback_max_nodes: int = 48
    rerank_simple_skip_threshold: int = 6
    rerank_simple_candidate_limit: int = 8
    rerank_complex_candidate_limit: int = 14
    simple_context_budget_chars: int = 1800
    complex_context_budget_chars: int = 3600
    simple_per_doc_chars: int = 480
    complex_per_doc_chars: int = 820

    top_k: int = 3
    bm25_top_k: int = 10

    temperature: float = 0.1
    max_tokens: int = 2048

    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2
    history_window: int = 10
    checkpointer_path: str = ".checkpoints/tcm.db"
    enable_long_term_write: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "neo4j_database": self.neo4j_database,
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "milvus_uri": self.milvus_uri,
            "milvus_collection_name": self.milvus_collection_name,
            "milvus_dimension": self.milvus_dimension,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "reranker_model": self.reranker_model,
            "retrieval_balance_strategy": self.retrieval_balance_strategy,
            "retrieval_timeout_seconds": self.retrieval_timeout_seconds,
            "retrieve_expand_factor": self.retrieve_expand_factor,
            "lit_expand_factor": self.lit_expand_factor,
            "graph_expand_factor": self.graph_expand_factor,
            "max_retrieval_top_k": self.max_retrieval_top_k,
            "max_graph_rows": self.max_graph_rows,
            "graph_fallback_max_nodes": self.graph_fallback_max_nodes,
            "rerank_simple_skip_threshold": self.rerank_simple_skip_threshold,
            "rerank_simple_candidate_limit": self.rerank_simple_candidate_limit,
            "rerank_complex_candidate_limit": self.rerank_complex_candidate_limit,
            "simple_context_budget_chars": self.simple_context_budget_chars,
            "complex_context_budget_chars": self.complex_context_budget_chars,
            "simple_per_doc_chars": self.simple_per_doc_chars,
            "complex_per_doc_chars": self.complex_per_doc_chars,
            "top_k": self.top_k,
            "bm25_top_k": self.bm25_top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_graph_depth": self.max_graph_depth,
            "history_window": self.history_window,
            "checkpointer_path": self.checkpointer_path,
            "enable_long_term_write": self.enable_long_term_write,
        }


DEFAULT_CONFIG = GraphRAGConfig()


class Config:
    NEO4J_URI = env_utils.NEO4J_URI or DEFAULT_CONFIG.neo4j_uri
    NEO4J_USER = env_utils.NEO4J_USER or DEFAULT_CONFIG.neo4j_user
    NEO4J_PASSWORD = env_utils.NEO4J_PASSWORD or DEFAULT_CONFIG.neo4j_password
    NEO4J_DATABASE = env_utils.NEO4J_DATABASE or DEFAULT_CONFIG.neo4j_database
    MILVUS_HOST = env_utils.MILVUS_HOST or DEFAULT_CONFIG.milvus_host
    MILVUS_PORT = int(env_utils.MILVUS_PORT or DEFAULT_CONFIG.milvus_port)
    MILVUS_URI = getattr(env_utils, "MILVUS_URI", None) or DEFAULT_CONFIG.milvus_uri
    MILVUS_COLLECTION_NAME = env_utils.MILVUS_COLLECTION_NAME or "tcm_knowledge"
    EMBEDDING_MODEL = getattr(env_utils, "EMBEDDING_MODEL", None) or DEFAULT_CONFIG.embedding_model
    LLM_MODEL = getattr(env_utils, "LLM_MODEL", None) or DEFAULT_CONFIG.llm_model
    RERANKER_MODEL = getattr(env_utils, "RERANKER_MODEL", None) or DEFAULT_CONFIG.reranker_model
    BM25_TOP_K = int(getattr(env_utils, "BM25_TOP_K", None) or DEFAULT_CONFIG.bm25_top_k)
    RETRIEVAL_BALANCE_STRATEGY = getattr(env_utils, "RETRIEVAL_BALANCE_STRATEGY", None) or DEFAULT_CONFIG.retrieval_balance_strategy
    RETRIEVAL_TIMEOUT_SECONDS = float(getattr(env_utils, "RETRIEVAL_TIMEOUT_SECONDS", None) or DEFAULT_CONFIG.retrieval_timeout_seconds)
    RETRIEVE_EXPAND_FACTOR = float(getattr(env_utils, "RETRIEVE_EXPAND_FACTOR", None) or DEFAULT_CONFIG.retrieve_expand_factor)
    LIT_EXPAND_FACTOR = float(getattr(env_utils, "LIT_EXPAND_FACTOR", None) or DEFAULT_CONFIG.lit_expand_factor)
    GRAPH_EXPAND_FACTOR = float(getattr(env_utils, "GRAPH_EXPAND_FACTOR", None) or DEFAULT_CONFIG.graph_expand_factor)
    MAX_RETRIEVAL_TOP_K = int(getattr(env_utils, "MAX_RETRIEVAL_TOP_K", None) or DEFAULT_CONFIG.max_retrieval_top_k)
    MAX_GRAPH_ROWS = int(getattr(env_utils, "MAX_GRAPH_ROWS", None) or DEFAULT_CONFIG.max_graph_rows)
    GRAPH_FALLBACK_MAX_NODES = int(getattr(env_utils, "GRAPH_FALLBACK_MAX_NODES", None) or DEFAULT_CONFIG.graph_fallback_max_nodes)
    RERANK_SIMPLE_SKIP_THRESHOLD = int(
        getattr(env_utils, "RERANK_SIMPLE_SKIP_THRESHOLD", None) or DEFAULT_CONFIG.rerank_simple_skip_threshold
    )
    RERANK_SIMPLE_CANDIDATE_LIMIT = int(
        getattr(env_utils, "RERANK_SIMPLE_CANDIDATE_LIMIT", None) or DEFAULT_CONFIG.rerank_simple_candidate_limit
    )
    RERANK_COMPLEX_CANDIDATE_LIMIT = int(
        getattr(env_utils, "RERANK_COMPLEX_CANDIDATE_LIMIT", None) or DEFAULT_CONFIG.rerank_complex_candidate_limit
    )
    SIMPLE_CONTEXT_BUDGET_CHARS = int(
        getattr(env_utils, "SIMPLE_CONTEXT_BUDGET_CHARS", None) or DEFAULT_CONFIG.simple_context_budget_chars
    )
    COMPLEX_CONTEXT_BUDGET_CHARS = int(
        getattr(env_utils, "COMPLEX_CONTEXT_BUDGET_CHARS", None) or DEFAULT_CONFIG.complex_context_budget_chars
    )
    SIMPLE_PER_DOC_CHARS = int(getattr(env_utils, "SIMPLE_PER_DOC_CHARS", None) or DEFAULT_CONFIG.simple_per_doc_chars)
    COMPLEX_PER_DOC_CHARS = int(getattr(env_utils, "COMPLEX_PER_DOC_CHARS", None) or DEFAULT_CONFIG.complex_per_doc_chars)
    FAST_MODE = _as_bool(getattr(env_utils, "FAST_MODE", None), default=False)
    FAST_EMBEDDING_MODEL = getattr(env_utils, "FAST_EMBEDDING_MODEL", None) or EMBEDDING_MODEL
    FAST_LLM_MODEL = getattr(env_utils, "FAST_LLM_MODEL", None) or LLM_MODEL
    FAST_RERANKER_MODEL = getattr(env_utils, "FAST_RERANKER_MODEL", None) or RERANKER_MODEL
    CHECKPOINTER_PATH = getattr(env_utils, "CHECKPOINTER_PATH", None) or DEFAULT_CONFIG.checkpointer_path
    SESSION_ID = env_utils.C9_SESSION_ID or None
