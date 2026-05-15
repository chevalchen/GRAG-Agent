from __future__ import annotations

import time
from collections.abc import Callable

from langchain_core.documents import Document

from src.app.online_qa.state import OnlineQAState
from src.core.tools.llm.generation_tool import LLMGenerationTool


def _extract_verdict(text: str) -> str:
    """Extract sufficient/insufficient from LLM response."""
    t = (text or "").lower().strip()
    if "insufficient" in t or "不充足" in t or "不足" in t:
        return "insufficient"
    if "sufficient" in t or "充足" in t:
        return "sufficient"
    # Default conservative: if unclear, treat as sufficient to avoid loops
    return "sufficient"


def make_sufficiency_node(
    llm: LLMGenerationTool,
    *,
    max_context_chars: int = 800,
) -> Callable[[OnlineQAState], dict]:
    """
    Sufficiency judge node.

    Uses a lightweight LLM call to decide whether docs_final covers
    the key facets of the query. Increments iteration_count on insufficient.

    Control flow is handled by the graph's conditional edge, not here.
    """

    def sufficiency_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        query = (state.get("rewritten_query") or state.get("query") or "").strip()
        docs_final: list[Document] = list(state.get("docs_final") or [])
        iteration_count = int(state.get("iteration_count") or 0)

        if not docs_final:
            # No docs at all → insufficient only on first pass, otherwise force sufficient
            verdict = "insufficient" if iteration_count == 0 else "sufficient"
            return _result(state, verdict, iteration_count, t0)

        # Build a compact context preview for the judge prompt
        snippets: list[str] = []
        total = 0
        for doc in docs_final:
            text = (doc.page_content or "").strip()[:200]
            if total + len(text) > max_context_chars:
                break
            snippets.append(text)
            total += len(text)
        context_preview = "\n---\n".join(snippets)

        prompt = "\n".join([
            "你是一个中医药知识充足性评估器。",
            "请判断以下检索结果是否足够回答用户问题。",
            "只输出一个词：sufficient（充足）或 insufficient（不足）。不要输出其他内容。",
            "",
            f"用户问题：{query}",
            "",
            "检索结果摘要：",
            context_preview,
        ])

        try:
            raw = llm.invoke_text(prompt)
            verdict = _extract_verdict(raw)
        except Exception:
            verdict = "sufficient"  # safe default on LLM failure

        return _result(state, verdict, iteration_count, t0)

    return sufficiency_node


def _result(state: OnlineQAState, verdict: str, iteration_count: int, t0: float) -> dict:
    new_count = iteration_count + 1 if verdict == "insufficient" else iteration_count
    return {
        "sufficiency": verdict,
        "iteration_count": new_count,
        "metrics": {
            **(state.get("metrics") or {}),
            "sufficiency_verdict": verdict,
            "sufficiency_seconds": time.time() - t0,
            "iteration_count": new_count,
        },
    }
