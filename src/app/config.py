from dataclasses import dataclass
from typing import Any, Dict

from src.utils import env_utils


@dataclass
class GraphRAGConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"

    top_k: int = 3

    temperature: float = 0.1
    max_tokens: int = 2048

    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2
    history_window: int = 10
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
            "milvus_collection_name": self.milvus_collection_name,
            "milvus_dimension": self.milvus_dimension,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_graph_depth": self.max_graph_depth,
            "history_window": self.history_window,
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
    MILVUS_COLLECTION_NAME = env_utils.MILVUS_COLLECTION_NAME or DEFAULT_CONFIG.milvus_collection_name
    SESSION_ID = env_utils.C9_SESSION_ID or None
