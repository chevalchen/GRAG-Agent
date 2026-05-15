from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from langchain_core.documents import Document

from src.app.config import GraphRAGConfig
from src.app.offline_ingestion.state import OfflineIngestionState
from src.core.tools.vector.milvus_tool import MilvusVectorTool


def make_health_milvus_index_node(config: GraphRAGConfig) -> Callable[[OfflineIngestionState], dict]:
    def health_milvus_index_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}

        health_records = list(state.get("health_chunk_records") or [])
        if not health_records:
            return {
                "health_indexed_count": 0,
                "metrics": {
                    **(state.get("metrics") or {}),
                    "health_milvus_index_seconds": 0.0,
                    "health_indexed": False,
                },
            }

        start = time.time()

        # 创建 health collection 专用的 config
        health_config = replace(config, milvus_collection_name=config.milvus_health_collection_name)
        milvus = MilvusVectorTool(health_config)

        if not milvus.ensure_collection(force_recreate=True):
            return {
                "health_indexed_count": 0,
                "metrics": {
                    **(state.get("metrics") or {}),
                    "health_milvus_index_seconds": time.time() - start,
                    "health_indexed": False,
                },
            }

        docs: list[Document] = []
        for row in health_records:
            text = str(row.get("text") or "").strip()
            node_id = str(row.get("node_id") or "").strip()
            if not text or not node_id:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_type": "health_popularization",
                        "chunk_type": str(row.get("chunk_type") or ""),
                        "source": str(row.get("source") or ""),
                        "drug_name": "",
                        "answer": str(row.get("answer") or ""),
                        "node_id": node_id,
                        "chunk_id": node_id,
                    },
                )
            )

        create_index_ok = False
        load_ok = False
        try:
            inserted = milvus.upsert_documents(docs)
            create_index_ok = bool(milvus.create_index())
            load_ok = bool(milvus.load_collection())
        except Exception:
            inserted = 0

        indexed_ok = bool(inserted) and create_index_ok and load_ok
        metrics = {
            **(state.get("metrics") or {}),
            "health_milvus_index_seconds": time.time() - start,
            "health_indexed": indexed_ok,
        }
        return {"health_indexed_count": int(inserted), "metrics": metrics}

    return health_milvus_index_node
