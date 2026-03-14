from __future__ import annotations

import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.llm.generation_tool import LLMGenerationTool


def make_answer_node(llm: LLMGenerationTool, *, history_window: int) -> Callable[[OnlineQAState], dict]:
    """
    构建回答节点
    
    Args:
        llm: LLM 生成工具
        history_window: 历史对话窗口大小
        
    Returns:
        回答节点
    """
    def answer_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        analysis = state.get("analysis")
        strategy = getattr(analysis, "recommended_strategy", "hybrid") if analysis else "hybrid"
        if strategy == "combined" and not (state.get("metrics") or {}).get("fuse_ready"):
            return {}

        query = (state.get("query") or "").strip()
        if not query:
            return {"answer": "请先输入问题。"}

        docs = state.get("docs_final") or []
        if not docs:
            answer = "抱歉，没有找到相关的烹饪信息。请尝试其他问题。"
            metrics = {**(state.get("metrics") or {}), "generate_seconds": time.time() - t0}
            return {
                "answer": answer,
                "history": [{"role": "user", "content": query}, {"role": "assistant", "content": answer}],
                "metrics": metrics,
            }

        history = state.get("history") or []
        recent = history[-int(history_window) :] if history_window > 0 else []
        hist_lines: list[str] = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if not content:
                continue
            prefix = "用户" if role == "user" else "助手" if role == "assistant" else str(role or "未知")
            hist_lines.append(f"{prefix}: {content}")
        history_block = "\n".join(hist_lines)

        context = "\n\n".join([d.page_content for d in docs[:10]])
        prompt = "\n".join(
            [
                "你是一个中文菜谱问答助手。你必须只基于提供的资料回答，不要编造不存在的细节。",
                "如果资料不足以回答，就说明缺少哪些信息，并给出可追问的问题。",
                "",
                "对话历史（最近几轮）：",
                history_block,
                "",
                "资料：",
                context,
                "",
                f"问题：{query}",
                "回答：",
            ]
        ).strip()
        try:
            answer_text = llm.invoke_text(prompt) or ""
        except Exception:
            answer_text = ""
        metrics = {**(state.get("metrics") or {}), "generate_seconds": time.time() - t0}
        return {
            "answer": answer_text,
            "history": [{"role": "user", "content": query}, {"role": "assistant", "content": answer_text}],
            "metrics": metrics,
        }

    return answer_node
