import asyncio
from typing import List

from langchain_core.documents import Document


class HybridSearchTool:
    def __init__(self, hybrid_retrieval_module):
        self._hybrid = hybrid_retrieval_module

    async def search(self, query: str, top_k: int) -> List[Document]:
        return await asyncio.to_thread(self._hybrid.hybrid_search, query, top_k)

