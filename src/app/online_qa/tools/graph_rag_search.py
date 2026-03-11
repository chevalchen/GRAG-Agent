import asyncio
from typing import List

from langchain_core.documents import Document


class GraphRAGSearchTool:
    def __init__(self, graph_rag_retrieval):
        self._graph_rag = graph_rag_retrieval

    async def search(self, query: str, top_k: int) -> List[Document]:
        return await asyncio.to_thread(self._graph_rag.graph_rag_search, query, top_k)

