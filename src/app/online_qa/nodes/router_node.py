from __future__ import annotations

import json
import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.tools.llm.generation_tool import LLMGenerationTool


_COMPLEXITY_MARKERS = [
    "并且", "同时", "以及", "分别", "区别", "对比",
    "联合", "搭配", "方案", "病例", "为什么", "怎么", "如何",
]

_VALID_INTENTS = {"chat", "theory", "symptom_to_drug", "conditions_exact"}

# 十八反 + 十九畏 pairs used by answer_node safety check (also imported there)
SAFETY_KEYWORDS = [
    "乌头", "附子", "半夏", "瓜蒌", "贝母", "白蔹", "白及",
    "藜芦", "人参", "苦参", "细辛", "芍药", "硫黄", "朴硝",
    "水银", "砒霜", "狼毒", "密陀僧", "巴豆", "牵牛", "丁香",
    "郁金", "川乌", "草乌", "犀角",
]


def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        return {}
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return {}
    try:
        return json.loads(t[start : end + 1])
    except Exception:
        return {}


def _classify_complexity(query: str, intent: str, keywords: list[str]) -> str:
    text = (query or "").strip()
    kw_count = len([k for k in keywords if str(k).strip()])
    marker_hit = any(m in text for m in _COMPLEXITY_MARKERS)
    if intent == "conditions_exact":
        return "complex"
    if marker_hit or kw_count >= 4 or len(text) >= 26:
        return "complex"
    if intent == "symptom_to_drug" and kw_count <= 3 and len(text) <= 22:
        return "simple"
    return "simple"


def make_router_node(llm: LLMGenerationTool) -> Callable[[OnlineQAState], dict]:
    """
    Build the intent router node.

    Single LLM call per invocation:
    - Classifies intent (4 classes)
    - Performs coreference resolution → rewritten_query
    - Extracts entity anchors (drugs / diseases / symptoms)
    """

    def router_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        query = (state.get("query") or "").strip()
        iteration_count = state.get("iteration_count") or 0

        if not query:
            return {
                "routing": {
                    "intent": "symptom_to_drug",
                    "complexity_level": "simple",
                    "keywords": [],
                    "reason": "empty_query",
                },
                "rewritten_query": query,
                "extracted_entities": {"drugs": [], "diseases": [], "symptoms": []},
                "iteration_count": iteration_count,
                "metrics": _metrics(state, t0),
            }

        history = state.get("history") or []
        history_snippet = ""
        if history:
            recent = history[-3:]
            history_snippet = "\n".join(
                f"{m.get('role','')}: {m.get('content','')[:80]}" for m in recent
            )

        prompt = "\n".join([
            "你是中医药智能问诊路由器。请输出严格 JSON，不要输出其他内容。",
            "",
            "JSON schema (all fields required):",
            '{',
            '  "intent": "chat|theory|symptom_to_drug|conditions_exact",',
            '  "rewritten_query": "<指代消解后的完整问题>",',
            '  "drugs": ["<药名>", ...],',
            '  "diseases": ["<病名>", ...],',
            '  "symptoms": ["<症状>", ...],',
            '  "keywords": ["<关键词>", ...],',
            '  "complexity_level": "simple|complex",',
            '  "reason": "<路由理由>"',
            '}',
            "",
            "意图定义:",
            "- chat: 非医疗闲聊、问候 → 直接回答，不检索",
            "- theory: 中医理论/科普问题，如：阴阳五行、气血理论",
            "- symptom_to_drug: 症状描述寻药，如：感冒用什么药",
            "- conditions_exact: 明确的配伍禁忌/禁用人群/不良反应查询",
            "",
            "规则:",
            "- rewritten_query: 结合对话历史消解指代，补全省略主语",
            "- keywords: 不超过 8 个",
            "- drugs/diseases/symptoms: 如无则为空列表",
            "",
            *(["最近对话历史:", history_snippet, ""] if history_snippet else []),
            f"当前问题：{query}",
        ])

        try:
            raw = llm.invoke_text(prompt)
        except Exception:
            raw = ""

        payload = _extract_json(raw)

        # --- intent ---
        intent = str(payload.get("intent") or "").strip()
        if intent not in _VALID_INTENTS:
            intent = "symptom_to_drug"  # one-level fallback

        # --- rewritten_query ---
        rewritten = str(payload.get("rewritten_query") or query).strip() or query

        # --- entities ---
        def _clean_list(v: object) -> list[str]:
            if not isinstance(v, list):
                return []
            return [str(x).strip() for x in v if str(x).strip()]

        extracted_entities = {
            "drugs": _clean_list(payload.get("drugs")),
            "diseases": _clean_list(payload.get("diseases")),
            "symptoms": _clean_list(payload.get("symptoms")),
        }

        keywords = _clean_list(payload.get("keywords"))
        reason = str(payload.get("reason") or f"intent={intent}").strip()

        # complexity: trust LLM only for simple/complex, else derive heuristically
        llm_complexity = str(payload.get("complexity_level") or "").strip()
        if llm_complexity not in {"simple", "complex"}:
            llm_complexity = _classify_complexity(rewritten, intent, keywords)

        return {
            "routing": {
                "intent": intent,
                "complexity_level": llm_complexity,
                "keywords": keywords,
                "reason": reason,
            },
            "rewritten_query": rewritten,
            "extracted_entities": extracted_entities,
            "iteration_count": iteration_count,
            "metrics": _metrics(state, t0, extra={"query_complexity": llm_complexity}),
        }

    return router_node


def _metrics(state: OnlineQAState, t0: float, extra: dict | None = None) -> dict:
    base = state.get("metrics") or {}
    m = {
        **base,
        "pipeline_started_at": base.get("pipeline_started_at", t0),
        "router_seconds": time.time() - t0,
    }
    if extra:
        m.update(extra)
    return m
