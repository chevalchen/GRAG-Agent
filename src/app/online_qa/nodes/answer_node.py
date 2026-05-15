from __future__ import annotations

import json
import time
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from src.app.online_qa.nodes.router_node import SAFETY_KEYWORDS
from src.app.online_qa.state import OnlineQAState
from src.core.tools.llm.generation_tool import LLMGenerationTool


def _clip_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 8:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _doc_context_text(doc) -> str:
    metadata = doc.metadata or {}
    question = str(doc.page_content or "").strip()
    answer = str(metadata.get("answer") or "").strip()
    chunk_type = str(metadata.get("chunk_type") or "").strip()
    doc_type = str(metadata.get("doc_type") or "").strip()
    if answer and (chunk_type == "qa" or doc_type == "health_popularization"):
        return f"问题：{question}\n回答：{answer}" if question else f"回答：{answer}"
    return question


def _build_context_with_budget(docs, *, total_budget: int, per_doc_chars: int) -> tuple[str, int, int]:
    grouped: dict[str, list[tuple[str, str]]] = {
        "graph_enrich": [],
        "graph_reverse": [],
        "graph_chain": [],
        "drug": [],
        "tcm_literature": [],
        "health_popularization": [],
        "other": [],
    }
    for doc in docs:
        metadata = doc.metadata or {}
        doc_type = str(metadata.get("doc_type") or "other")
        source = str(metadata.get("source") or "").strip()
        content = _doc_context_text(doc)
        grouped.setdefault(doc_type, [])
        grouped[doc_type].append((content, source))

    section_plan = [
        ("graph_enrich",       "【图谱证据】",   0.30),
        ("graph_reverse",      "【禁忌/适应】",  0.25),
        ("graph_chain",        "【图谱证据链】", 0.10),
        ("drug",               "【药品说明书】", 0.15),
        ("tcm_literature",     "【中医文献】",   0.12),
        ("health_popularization", "【健康科普】", 0.06),
        ("other",              "【其他资料】",   0.02),
    ]
    context_parts: list[str] = []
    used_chars = 0
    used_docs = 0
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
            clip_limit = min(per_doc_chars, section_budget - section_used, total_budget - used_chars)
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


def _extract_next_questions(raw: str) -> list[str]:
    """Extract list of question strings from LLM JSON output."""
    t = (raw or "").strip()
    start = t.find("[")
    end = t.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(t[start : end + 1])
        return [str(x).strip() for x in items if str(x).strip()][:3]
    except Exception:
        return []


def _check_safety(text: str) -> tuple[bool, str]:
    """Check for 十八反/十九畏 ingredient keywords in generated text."""
    hits = [kw for kw in SAFETY_KEYWORDS if kw in text]
    if len(hits) >= 2:
        return True, f"触发配伍禁忌关键词：{'、'.join(hits[:4])}"
    return False, ""


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
    Async answer generation node.

    - Uses llm._llm.astream() so LangGraph astream_events emits token events.
    - Generates next_questions in a second LLM call.
    - Applies inline 十八反/十九畏 safety check; appends disclaimer if triggered.
    """

    async def answer_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        query = (state.get("rewritten_query") or state.get("query") or "").strip()
        if not query:
            return {"answer": "请先输入问题。", "next_questions": []}

        docs = state.get("docs_final") or []
        if not docs:
            answer = "抱歉，没有检索到可用的中医药资料，请尝试补充症状或药品名称。"
            return {
                "answer": answer,
                "next_questions": [],
                "safety_blocked": False,
                "safety_reason": "",
                "history": [{"role": "user", "content": query}, {"role": "assistant", "content": answer}],
                "metrics": _metrics(state, t0),
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

        routing = state.get("routing") or {}
        complexity = str(routing.get("complexity_level") or "complex")
        total_budget = complex_context_budget_chars if complexity == "complex" else simple_context_budget_chars
        per_doc_chars = complex_per_doc_chars if complexity == "complex" else simple_per_doc_chars

        context, context_used_chars, context_docs_used = _build_context_with_budget(
            docs,
            total_budget=total_budget,
            per_doc_chars=per_doc_chars,
        )
        history_block = _clip_text("\n".join(hist_lines), 900 if complexity == "complex" else 500)

        prompt = "\n".join([
            "你是一个中文中医药问诊助手。你必须只基于提供的资料回答，不要编造不存在的细节。",
            "如果资料不足以回答，就说明缺少哪些信息，并给出可追问的问题。",
            "若资料中无【适应症】字段，必须明确说明是基于功效/证候/禁忌/组方做的替代推理。",
            "",
            "对话历史（最近几轮）：",
            history_block,
            "",
            "资料：",
            context,
            "",
            f"问题：{query}",
            "回答：",
        ]).strip()

        # Stream via astream — LangGraph astream_events captures per-token events
        chunks: list[str] = []
        async for chunk in llm._llm.astream([HumanMessage(content=prompt)]):
            text = getattr(chunk, "content", None)
            if text:
                chunks.append(text)
        answer_text = "".join(chunks)

        # Inline safety check
        safety_blocked, safety_reason = _check_safety(answer_text)
        if safety_blocked:
            answer_text += "\n\n⚠️ 注意：本回答涉及配伍禁忌，请在中医医师指导下使用。"

        # Generate follow-up questions (sync, post-stream)
        next_questions: list[str] = []
        try:
            suggest_prompt = "\n".join([
                "请为以下中医药问答生成3个相关的追问问题，以JSON字符串数组格式输出，不要输出其他内容。",
                f"问题：{query}",
                f"回答摘要：{_clip_text(answer_text, 300)}",
                '示例格式：["问题1", "问题2", "问题3"]',
            ])
            nq_raw = llm.invoke_text(suggest_prompt)
            next_questions = _extract_next_questions(nq_raw)
        except Exception:
            pass

        m = _metrics(state, t0, extra={
            "context_budget_chars": total_budget,
            "context_used_chars": context_used_chars,
            "context_docs_used": context_docs_used,
        })

        return {
            "answer": answer_text,
            "next_questions": next_questions,
            "safety_blocked": safety_blocked,
            "safety_reason": safety_reason,
            "history": [{"role": "user", "content": query}, {"role": "assistant", "content": answer_text}],
            "metrics": m,
        }

    return answer_node


def _metrics(state: OnlineQAState, t0: float, extra: dict | None = None) -> dict:
    base = state.get("metrics") or {}
    m = {**base, "generate_seconds": time.time() - t0}
    started_at = m.get("pipeline_started_at")
    if isinstance(started_at, (int, float)):
        m["total_seconds"] = max(time.time() - float(started_at), 0.0)
    if extra:
        m.update(extra)
    return m
