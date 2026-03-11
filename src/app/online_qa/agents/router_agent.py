import asyncio
from typing import Optional

from src.legacy.rag_modules.intelligent_query_router import QueryAnalysis
from src.app.online_qa.tools.registry import ToolContext


class RouterAgent:
    def __init__(self, ctx: ToolContext, llm_semaphore: asyncio.Semaphore):
        self._ctx = ctx
        self._llm_sem = llm_semaphore

    async def analyze(self, query: str) -> Optional[QueryAnalysis]:
        router = self._ctx.get("router")
        async with self._llm_sem:
            return await asyncio.to_thread(router.analyze_query, query)
