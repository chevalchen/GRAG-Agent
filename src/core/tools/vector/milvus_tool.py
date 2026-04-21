from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.app.config import GraphRAGConfig


def _join_text_list(values: Any) -> str:
    if isinstance(values, str):
        return values
    if not isinstance(values, list):
        return ""
    return " | ".join([str(x).strip() for x in values if str(x).strip()])


class _MilvusVectorToolInput(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=100)
    expr: str = ""
    source_hint: list[str] = Field(default_factory=list)
    filter_expr: str = ""


@lru_cache(maxsize=8)
def _get_milvus_client(host: str, port: int):
    from pymilvus import MilvusClient as _MilvusClient

    return _MilvusClient(uri=f"http://{host}:{int(port)}")


@lru_cache(maxsize=8)
def _get_milvus_client_by_uri(uri: str):
    from pymilvus import MilvusClient as _MilvusClient

    return _MilvusClient(uri=uri)


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
            uri = str(getattr(config, "milvus_uri", "") or "").strip()
            if uri:
                self._client = _get_milvus_client_by_uri(uri)
            else:
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

    @property
    def client(self):
        return self._client

    def _run(self, query: str, top_k: int = 10, expr: str = "", source_hint: list[str] | None = None, filter_expr: str = "") -> list[Document]:
        if self._client is None:
            return []
        q = (query or "").strip()
        if not q:
            return []
        embeddings = self._get_embeddings()
        if embeddings is None:
            return []
        vector = embeddings.embed_query(q)
        filters = []
        if expr:
            filters.append(f"({expr})")
        elif filter_expr:
            filters.append(f"({filter_expr})")
        hints = [str(x).strip() for x in (source_hint or []) if str(x).strip()]
        if hints:
            or_expr = " OR ".join([f'source == "{x}"' for x in hints])
            filters.append(f"({or_expr})")
        final_filter = " AND ".join(filters) if filters else None
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                data=[vector],
                anns_field="vector",
                limit=int(top_k),
                filter=final_filter,
                output_fields=[
                    "text",
                    "doc_type",
                    "chunk_type",
                    "source",
                    "drug_name",
                    "canonical_name",
                    "normalized_name",
                    "aliases_text",
                    "alias_norms_text",
                    "answer",
                    "node_id",
                    "chunk_id",
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
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="drug_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="canonical_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="normalized_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="aliases_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="alias_norms_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
        ]
        return CollectionSchema(fields=fields, description="中医药知识图谱与文献向量集合")

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
                    "doc_type": str(md.get("doc_type") or "")[:50],
                    "chunk_type": str(md.get("chunk_type") or "")[:50],
                    "source": str(md.get("source") or "")[:200],
                    "drug_name": str(md.get("drug_name") or "")[:300],
                    "canonical_name": str(md.get("canonical_name") or md.get("drug_name") or "")[:300],
                    "normalized_name": str(md.get("normalized_name") or "")[:300],
                    "aliases_text": _join_text_list(md.get("aliases"))[:2000],
                    "alias_norms_text": _join_text_list(md.get("alias_norms"))[:2000],
                    "answer": str(md.get("answer") or "")[:15000],
                    "node_id": str(md.get("node_id") or "")[:100],
                    "chunk_id": chunk_id[:150],
                }
            )
        batch_size = 100
        inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._client.insert(collection_name=self.collection_name, data=batch)
            inserted += len(batch)
        return inserted
