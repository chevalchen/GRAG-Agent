import asyncio
from typing import List

from langchain_core.documents import Document

from src.app.online_qa.tools.registry import ToolContext


class GraphRetrievalAgent:
    def __init__(self, ctx: ToolContext, retrieve_semaphore: asyncio.Semaphore):
        self._ctx = ctx
        self._retrieve_sem = retrieve_semaphore

    async def retrieve(self, query: str, top_k: int) -> List[Document]:
        tool = self._ctx.get("graph_rag_search")
        async with self._retrieve_sem:
            return await tool.search(query, top_k)

