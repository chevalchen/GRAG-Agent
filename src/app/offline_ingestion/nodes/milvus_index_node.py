from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.config import GraphRAGConfig
from src.app.offline_ingestion.state import OfflineIngestionState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_milvus_index_node(config: GraphRAGConfig) -> Callable[[OfflineIngestionState], dict]:
    def milvus_index_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}

        start = time.time()
        milvus = MilvusVectorTool(config)
        if not milvus.ensure_collection(force_recreate=True):
            metrics = {**(state.get("metrics") or {}), "milvus_index_seconds": time.time() - start, "milvus_indexed": False}
            return {"indexed_count": 0, "metrics": metrics}

        parsed_records = list(state.get("parsed_records") or [])
        chunk_records = list(state.get("chunk_records") or [])
        docs: list[Document] = []

        for row in parsed_records:
            drug_name = str(row.get("drug_name") or "").strip()
            if not drug_name:
                continue
            ingredients = [str(x).strip() for x in (row.get("ingredients") or []) if str(x).strip()]
            merged_ingredients = "、".join(ingredients)
            text = f"药品{drug_name}，含有成分：{merged_ingredients}"
            chunk_id = f"drug_{hashlib.md5(drug_name.encode('utf-8')).hexdigest()[:12]}"
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_type": "drug",
                        "chunk_type": "profile",
                        "source": "medical_ner_entities",
                        "drug_name": drug_name,
                        "answer": "",
                        "node_id": chunk_id,
                        "chunk_id": chunk_id,
                    },
                )
            )

        for row in chunk_records:
            text = str(row.get("text") or "").strip()
            node_id = str(row.get("node_id") or "").strip()
            if not text or not node_id:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_type": "tcm_literature",
                        "chunk_type": str(row.get("chunk_type") or ""),
                        "source": str(row.get("source") or ""),
                        "drug_name": "",
                        "answer": str(row.get("answer") or ""),
                        "node_id": node_id,
                        "chunk_id": node_id,
                    },
                )
            )
        try:
            inserted = milvus.upsert_documents(docs)
            _ = milvus.create_index()
            _ = milvus.load_collection()
        except Exception:
            inserted = 0

        metrics = {**(state.get("metrics") or {}), "milvus_index_seconds": time.time() - start, "milvus_indexed": True}
        return {"indexed_count": int(inserted), "metrics": metrics}

    return milvus_index_node
