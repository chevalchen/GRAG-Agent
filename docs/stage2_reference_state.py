"""
Stage 2 Reference: AgenticRagState design (supplementary reference only).
Original plan design in docs/stage2_backend_design.md remains the authority.

Source: user-provided reference 2026-05-13
"""
from typing import TypedDict, Annotated, List, Dict, Any
import operator


class AgenticRagState(TypedDict):
    # ================= 1. 基础会话与元数据 (Base Info) =================
    user_id: str
    thread_id: str
    original_query: str

    # ================= 2. 记忆与画像层 (Memory & Profile) =================
    # operator.add ensures multi-turn history appends rather than overwrites
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    # long-term profile (from SQLite): chronic conditions, allergy history
    user_profile: Dict[str, Any]

    # ================= 3. 意图与重写层 (Routing & Rewriting) =================
    intent_type: str                   # chat / theory / symptom_to_drug / condition_to_drug
    rewritten_query: str               # after coreference resolution + profile injection
    extracted_entities: Dict[str, List[str]]  # {'drugs': [], 'diseases': [], 'symptoms': []}

    # ================= 4. 检索数据层 (Retrieval Payload) =================
    milvus_docs: List[Dict[str, Any]]  # long-text slices from Milvus
    neo4j_rules: List[str]             # definitive contraindication paths from Neo4j
    final_context: str                 # Top-6 context after RRF + Cross-Encoder

    # ================= 5. 控制流与安全标识 (Control Flags) =================
    is_sufficient: bool                # sufficiency reflection flag
    safety_blocked: bool               # safety circuit-breaker flag
    safety_reason: str                 # e.g. "触发十八反禁忌"

    # ================= 6. 最终输出 (Output) =================
    final_answer: str
