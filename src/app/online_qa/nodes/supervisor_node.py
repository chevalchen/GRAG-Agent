from __future__ import annotations

import json
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
from src.core.schemas.document import QueryAnalysis
from src.core.tools.llm.generation_tool import LLMGenerationTool


def _extract_json(text: str) -> dict:
    """
    从文本中提取 JSON 字符串
    
    Args:
        text: 输入文本
        
    Returns:
        提取到的 JSON 字典
    """
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


def make_supervisor_node(llm: LLMGenerationTool) -> Callable[[OnlineQAState], dict]:
    """
    构建监督节点
    
    Args:
        llm: LLM 生成工具
    Returns:
        监督节点
    """
    def supervisor_node(state: OnlineQAState) -> dict:
        """
        监督节点
        
        Args:
            state: 在线问答状态
            
        Returns:
            分析结果
        """
        query = (state.get("query") or "").strip()
        if not query:
            return {"analysis": QueryAnalysis(original_query="", recommended_strategy="hybrid")}

        prompt = "\n".join(
            [
                "你是一个用于菜谱知识问答的查询分析器。请输出严格 JSON，不要输出其他内容。",
                'JSON schema: {"intent": str, "keywords": list[str], "query_complexity": float, "relationship_intensity": float, "recommended_strategy": "hybrid"|"graph_rag"|"combined"}',
                "规则：",
                "- keywords 尽量提取食材/做法/菜名/关键动作，不超过 8 个",
                "- query_complexity 取 0~1，越复杂越接近 1",
                "- relationship_intensity 取 0~1，越需要多跳关系推理越接近 1",
                '- recommended_strategy 只允许 "hybrid" 或 "graph_rag" 或 "combined"',
                "",
                f"用户问题：{query}",
            ]
        )
        try:
            raw = llm.invoke_text(prompt)
        except Exception:
            raw = ""
        payload = _extract_json(raw)
        intent = str(payload.get("intent") or "")
        keywords = payload.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(x).strip() for x in keywords if str(x).strip()]
        try:
            query_complexity = float(payload.get("query_complexity") or 0.0)
        except Exception:
            query_complexity = 0.0
        try:
            relationship_intensity = float(payload.get("relationship_intensity") or 0.0)
        except Exception:
            relationship_intensity = 0.0
        recommended_strategy = str(payload.get("recommended_strategy") or "hybrid").strip()
        if recommended_strategy not in {"hybrid", "graph_rag", "combined"}:
            recommended_strategy = "hybrid"

        return {
            "analysis": QueryAnalysis(
                original_query=query,
                intent=intent,
                keywords=keywords,
                query_complexity=max(0.0, min(1.0, query_complexity)),
                relationship_intensity=max(0.0, min(1.0, relationship_intensity)),
                recommended_strategy=recommended_strategy,
            )
        }

    return supervisor_node
