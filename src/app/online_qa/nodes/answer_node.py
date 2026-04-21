from __future__ import annotations

import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.llm.generation_tool import LLMGenerationTool


def _clip_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 8:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _build_context_with_budget(docs, resolved_drug: dict, *, total_budget: int, per_doc_chars: int) -> tuple[str, int, int]:
    grouped: dict[str, list[tuple[str, str]]] = {
        "graph_chain": [],
        "drug": [],
        "tcm_literature": [],
        "health_science": [],
        "other": [],
    }
    for doc in docs:
        metadata = doc.metadata or {}
        doc_type = str(metadata.get("doc_type") or "other")
        source = str(metadata.get("source") or "").strip()
        if doc_type not in grouped:
            doc_type = "other"
        grouped[doc_type].append((doc.page_content, source))
    section_plan = [
        ("graph_chain", "【图谱证据链】", 0.34),
        ("drug", "【药品说明书】", 0.30),
        ("tcm_literature", "【中医文献】", 0.20),
        ("health_science", "【健康科普】", 0.11),
        ("other", "【其他资料】", 0.05),
    ]
    context_parts: list[str] = []
    used_chars = 0
    used_docs = 0
    canonical_name = str(resolved_drug.get("canonical_name") or "").strip()
    matched_alias = str(resolved_drug.get("matched_alias") or "").strip()
    if canonical_name and matched_alias and matched_alias != canonical_name:
        alias_note = f"【药品名归一】\n已将问题中的“{matched_alias}”识别为标准药品“{canonical_name}”。"
        context_parts.append(alias_note)
        used_chars += len(alias_note)
    for doc_type, title, ratio in section_plan:
        section_docs = grouped.get(doc_type) or []
        if not section_docs or used_chars >= total_budget:
            continue
        section_budget = min(max(int(total_budget * ratio), 180), max(total_budget - used_chars, 0))
        lines: list[str] = []
        section_used = 0
        for content, source in section_docs:
            if section_used >= section_budget or used_chars >= total_budget:
                break
            header = title if not source or doc_type != "tcm_literature" else f"【中医文献 · {source}】"
            remain_in_section = section_budget - section_used
            remain_total = total_budget - used_chars
            clip_limit = min(per_doc_chars, remain_in_section, remain_total)
            if clip_limit <= 0:
                break
            clipped = _clip_text(content, clip_limit)
            if not clipped:
                continue
            lines.append(f"{header}\n{clipped}")
            consumed = len(clipped) + len(header) + 1
            section_used += consumed
            used_chars += consumed
            used_docs += 1
        if lines:
            context_parts.append("\n\n".join(lines))
    return "\n\n".join(context_parts), used_chars, used_docs


def make_answer_node(
    llm: LLMGenerationTool,
    *,
    history_window: int,
    simple_context_budget_chars: int = 1800,
    complex_context_budget_chars: int = 3600,
    simple_per_doc_chars: int = 480,
    complex_per_doc_chars: int = 820,
) -> Callable[[OnlineQAState], dict]:
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
            started_at = metrics.get("pipeline_started_at")
            if isinstance(started_at, (int, float)):
                metrics["total_seconds"] = max(time.time() - float(started_at), 0.0)
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

        routing = state.get("routing") or {}
        complexity = str(routing.get("complexity_level") or "complex")
        resolved_drug = state.get("resolved_drug") or {}
        total_budget = int(complex_context_budget_chars if complexity == "complex" else simple_context_budget_chars)
        per_doc_chars = int(complex_per_doc_chars if complexity == "complex" else simple_per_doc_chars)
        context, context_used_chars, context_docs_used = _build_context_with_budget(
            docs,
            resolved_drug,
            total_budget=total_budget,
            per_doc_chars=per_doc_chars,
        )
        history_block = _clip_text(history_block, 900 if complexity == "complex" else 500)
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
        metrics = {
            **(state.get("metrics") or {}),
            "generate_seconds": time.time() - t0,
            "context_budget_chars": total_budget,
            "context_used_chars": context_used_chars,
            "context_docs_used": context_docs_used,
        }
        started_at = metrics.get("pipeline_started_at")
        if isinstance(started_at, (int, float)):
            metrics["total_seconds"] = max(time.time() - float(started_at), 0.0)
        return {
            "answer": answer_text,
            "history": [{"role": "user", "content": query}, {"role": "assistant", "content": answer_text}],
            "metrics": metrics,
        }

    return answer_node
