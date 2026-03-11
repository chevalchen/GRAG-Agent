import asyncio

from src.app.online_qa.tools.registry import ToolContext


class AnswerAgent:
    def __init__(self, ctx: ToolContext, llm_semaphore: asyncio.Semaphore):
        self._ctx = ctx
        self._llm_sem = llm_semaphore

    async def generate(self, query: str, docs, stream: bool = False) -> str:
        tool = self._ctx.get("answer_generation")
        async with self._llm_sem:
            return await tool.generate(query, docs, stream=stream)

