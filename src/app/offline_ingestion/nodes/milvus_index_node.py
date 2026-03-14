from __future__ import annotations

import time
from collections.abc import Callable

from src.app.config import GraphRAGConfig
from src.app.offline_ingestion.state import OfflineIngestionState
from src.core.utils.recipe_loader import load_recipe_docs
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

        docs = load_recipe_docs(config)
        try:
            inserted = milvus.upsert_documents(docs)
            _ = milvus.create_index()
            _ = milvus.load_collection()
        except Exception:
            inserted = 0

        metrics = {**(state.get("metrics") or {}), "milvus_index_seconds": time.time() - start, "milvus_indexed": True}
        return {"indexed_count": int(inserted), "metrics": metrics}

    return milvus_index_node
