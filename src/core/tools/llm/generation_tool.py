from __future__ import annotations

from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from openai import RateLimitError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.app.config import GraphRAGConfig


class _LLMGenerationToolInput(BaseModel):
    """LLM 生成工具输入"""
    prompt: str = Field(..., description="生成任务提示词")


class LLMGenerationTool(BaseTool):
    """LLM 生成工具"""
    name: str = "llm_generate"
    description: str = "使用 ChatOpenAI 生成文本，支持重试与流式输出"
    args_schema: type[BaseModel] = _LLMGenerationToolInput

    def __init__(self, config: GraphRAGConfig):
        super().__init__()
        from langchain_openai import ChatOpenAI
        from src.utils import env_utils

        api_key = env_utils.MOONSHOT_API_KEY or env_utils.KIMI_API_KEY or env_utils.OPENAI_API_KEY
        base_url = env_utils.MOONSHOT_BASE_URL
        self._llm = ChatOpenAI(
            model=config.llm_model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(config.temperature),
            max_tokens=int(config.max_tokens),
        )

    def _run(self, prompt: str) -> str:
        return self.invoke_text(prompt)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def invoke_text(self, prompt: str) -> str:
        msg = self._llm.invoke([HumanMessage(content=prompt)])
        return getattr(msg, "content", "") or ""

    def stream_text(self, prompt: str) -> Iterator[str]:
        for chunk in self._llm.stream([HumanMessage(content=prompt)]):
            text = getattr(chunk, "content", None)
            if text:
                yield text
