from __future__ import annotations

import json
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState
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
            return {
                "routing": {
                    "query_intent": "tcm_theory",
                    "use_graph": False,
                    "use_drug_vec": False,
                    "use_lit_vec": True,
                    "use_health_vec": False,
                    "source_hint": [],
                    "keywords": [],
                    "reason": "empty_query",
                }
            }

        prompt = "\n".join(
            [
                "你是中医药问诊路由器。请输出严格 JSON，不要输出其他内容。",
                'JSON schema: {"query_intent":"drug_specific|symptom_disease|tcm_theory|clinical_case|health_advice","use_graph":bool,"use_drug_vec":bool,"use_lit_vec":bool,"use_health_vec":bool,"source_hint":list[str],"keywords":list[str],"reason":str}',
                "路由规则：",
                "- drug_specific: use_drug_vec=true,use_graph=true",
                "- symptom_disease: use_graph=true,use_drug_vec=true",
                "- tcm_theory: use_lit_vec=true",
                "- clinical_case: use_lit_vec=true,use_graph=true",
                "- health_advice: use_health_vec=true,use_lit_vec=true",
                "- keywords 不超过 8 个，source_hint 可为空列表",
                "",
                f"用户问题：{query}",
            ]
        )
        try:
            raw = llm.invoke_text(prompt)
        except Exception:
            raw = ""
        payload = _extract_json(raw)
        query_intent = str(payload.get("query_intent") or "").strip()
        keywords = payload.get("keywords") or []
        source_hint = payload.get("source_hint") or []
        if not isinstance(keywords, list):
            keywords = []
        if not isinstance(source_hint, list):
            source_hint = []
        keywords = [str(x).strip() for x in keywords if str(x).strip()]
        source_hint = [str(x).strip() for x in source_hint if str(x).strip()]
        valid_intent = {"drug_specific", "symptom_disease", "tcm_theory", "clinical_case", "health_advice"}
        if query_intent not in valid_intent:
            query_intent = "tcm_theory"

        route_by_intent = {
            "drug_specific": {"use_graph": True, "use_drug_vec": True, "use_lit_vec": False, "use_health_vec": False},
            "symptom_disease": {"use_graph": True, "use_drug_vec": True, "use_lit_vec": False, "use_health_vec": False},
            "tcm_theory": {"use_graph": False, "use_drug_vec": False, "use_lit_vec": True, "use_health_vec": False},
            "clinical_case": {"use_graph": True, "use_drug_vec": False, "use_lit_vec": True, "use_health_vec": False},
            "health_advice": {"use_graph": False, "use_drug_vec": False, "use_lit_vec": True, "use_health_vec": True},
        }
        defaults = route_by_intent[query_intent]
        use_graph = bool(payload.get("use_graph", defaults["use_graph"]))
        use_drug_vec = bool(payload.get("use_drug_vec", defaults["use_drug_vec"]))
        use_lit_vec = bool(payload.get("use_lit_vec", defaults["use_lit_vec"]))
        use_health_vec = bool(payload.get("use_health_vec", defaults["use_health_vec"]))
        reason = str(payload.get("reason") or f"intent={query_intent}")

        return {
            "routing": {
                "query_intent": query_intent,
                "use_graph": use_graph,
                "use_drug_vec": use_drug_vec,
                "use_lit_vec": use_lit_vec,
                "use_health_vec": use_health_vec,
                "source_hint": source_hint,
                "keywords": keywords,
                "reason": reason,
            }
        }

    return supervisor_node
