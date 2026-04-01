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
        query = (state.get("query") or "").strip()
        if not query:
            return {"answer": "请先输入问题。"}

        docs = state.get("docs_final") or []
        if not docs:
            answer = "抱歉，没有检索到可用的中医药资料，请尝试补充症状或药品名称。"
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

        grouped: dict[str, list[tuple[str, str]]] = {
            "graph_chain": [],
            "drug": [],
            "tcm_literature": [],
            "health_science": [],
            "other": [],
        }
        for doc in docs[:10]:
            metadata = doc.metadata or {}
            doc_type = str(metadata.get("doc_type") or "other")
            source = str(metadata.get("source") or "").strip()
            if doc_type not in grouped:
                doc_type = "other"
            grouped[doc_type].append((doc.page_content, source))
        context_parts: list[str] = []
        if grouped["graph_chain"]:
            context_parts.append("【图谱证据链】\n" + "\n\n".join([x[0] for x in grouped["graph_chain"]]))
        if grouped["drug"]:
            context_parts.append("【药品说明书】\n" + "\n\n".join([x[0] for x in grouped["drug"]]))
        if grouped["tcm_literature"]:
            lit_parts = []
            for content, source in grouped["tcm_literature"]:
                title = f"【中医文献 · {source}】" if source else "【中医文献】"
                lit_parts.append(f"{title}\n{content}")
            context_parts.append("\n\n".join(lit_parts))
        if grouped["health_science"]:
            context_parts.append("【健康科普】\n" + "\n\n".join([x[0] for x in grouped["health_science"]]))
        if grouped["other"]:
            context_parts.append("【其他资料】\n" + "\n\n".join([x[0] for x in grouped["other"]]))
        context = "\n\n".join(context_parts)
        prompt = "\n".join(
            [
                "你是一个中文中医药问诊助手。你必须只基于提供的资料回答，不要编造不存在的细节。",
                "如果资料不足以回答，就说明缺少哪些信息，并给出可追问的问题。",
                "若缺少直接“适应症”字样，必须明确说明是基于功效/证候/禁忌/组方做的替代推理。",
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
