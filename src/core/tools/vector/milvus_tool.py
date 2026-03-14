from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.app.config import GraphRAGConfig


class _MilvusVectorToolInput(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=100)
    filter_expr: str = ""


@lru_cache(maxsize=8)
def _get_milvus_client(host: str, port: int):
    from pymilvus import MilvusClient as _MilvusClient

    return _MilvusClient(uri=f"http://{host}:{int(port)}")


@lru_cache(maxsize=4)
def _get_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class MilvusVectorTool(BaseTool):
    name: str = "milvus_search"
    description: str = "在 Milvus 向量库中检索相关文本，返回 LangChain Documents"
    args_schema: type[BaseModel] = _MilvusVectorToolInput

    def __init__(self, config: GraphRAGConfig):
        super().__init__()
        self._config = config
        try:
            self._client = _get_milvus_client(config.milvus_host, config.milvus_port)
        except Exception:
            self._client = None
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        try:
            self._embeddings = _get_embeddings(self._config.embedding_model)
            return self._embeddings
        except Exception:
            return None

    @property
    def collection_name(self) -> str:
        return self._config.milvus_collection_name

    def _run(self, query: str, top_k: int = 10, filter_expr: str = "") -> list[Document]:
        if self._client is None:
            return []
        q = (query or "").strip()
        if not q:
            return []
        embeddings = self._get_embeddings()
        if embeddings is None:
            return []
        vector = embeddings.embed_query(q)
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                data=[vector],
                anns_field="vector",
                limit=int(top_k),
                filter=(filter_expr or None),
                output_fields=[
                    "text",
                    "node_id",
                    "recipe_name",
                    "node_type",
                    "category",
                    "cuisine_type",
                    "difficulty",
                    "doc_type",
                    "chunk_id",
                    "parent_id",
                ],
            )
        except Exception:
            return []

        docs: list[Document] = []
        for hit in (hits[0] if hits else []):
            entity: dict[str, Any] = hit.get("entity") or {}
            score = hit.get("distance", hit.get("score", 0.0))
            metadata = {k: entity.get(k) for k in entity.keys()}
            metadata["score"] = float(score) if score is not None else 0.0
            metadata["search_source"] = "milvus"
            docs.append(Document(page_content=str(entity.get("text") or ""), metadata=metadata))
        return docs

    def has_collection(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.has_collection(self.collection_name))
        except Exception:
            return False

    def load_collection(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.load_collection(self.collection_name)
            return True
        except Exception:
            return False

    def ensure_collection(self, *, force_recreate: bool = False) -> bool:
        if self._client is None:
            return False
        try:
            if self._client.has_collection(self.collection_name):
                if not force_recreate:
                    return True
                self._client.drop_collection(self.collection_name)
            schema = self._create_collection_schema()
            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",
                consistency_level="Strong",
            )
            return True
        except Exception:
            return False

    def _create_collection_schema(self):
        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=int(self._config.milvus_dimension)),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
        ]
        return CollectionSchema(fields=fields, description="中式烹饪知识图谱向量集合")

    def create_index(self) -> bool:
        if self._client is None:
            return False
        try:
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self._client.create_index(collection_name=self.collection_name, index_params=index_params)
            return True
        except Exception:
            return False

    def upsert_documents(self, docs: list[Document]) -> int:
        if self._client is None:
            return 0
        if not docs:
            return 0
        embeddings = self._get_embeddings()
        if embeddings is None:
            return 0
        texts = [d.page_content for d in docs]
        vectors = embeddings.embed_documents(texts)
        rows: list[dict[str, Any]] = []
        for i, (doc, vector) in enumerate(zip(docs, vectors)):
            md = doc.metadata or {}
            chunk_id = str(md.get("chunk_id") or md.get("id") or f"chunk_{i}")
            rows.append(
                {
                    "id": chunk_id[:150],
                    "vector": vector,
                    "text": (doc.page_content or "")[:15000],
                    "node_id": str(md.get("node_id") or "")[:100],
                    "recipe_name": str(md.get("recipe_name") or "")[:300],
                    "node_type": str(md.get("node_type") or "")[:100],
                    "category": str(md.get("category") or "")[:100],
                    "cuisine_type": str(md.get("cuisine_type") or "")[:200],
                    "difficulty": int(md.get("difficulty") or 0),
                    "doc_type": str(md.get("doc_type") or "")[:50],
                    "chunk_id": chunk_id[:150],
                    "parent_id": str(md.get("parent_id") or "")[:100],
                }
            )
        batch_size = 100
        inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._client.insert(collection_name=self.collection_name, data=batch)
            inserted += len(batch)
        return inserted
