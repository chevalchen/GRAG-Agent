from langchain_core.documents import Document as LCDocument

from src.app.online_qa import nodes as online_nodes
from src.app.online_qa.state import OnlineQAState


def answer_node(state: OnlineQAState) -> dict:
    if online_nodes.RUNTIME is None:
        return {"error": "runtime_not_initialized"}
    if (state.get("metrics") or {}).get("stream"):
        return {"answer": None}
    query = state.get("query", "")
    if not query:
        return {"answer": "请先输入问题。"}
    docs = state.get("fused_docs") or state.get("hybrid_docs") or state.get("graph_docs") or []
    if not docs:
        answer = "抱歉，没有找到相关的烹饪信息。请尝试其他问题。"
        return {
            "answer": answer,
            "history": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ],
        }
    lc_docs = [LCDocument(page_content=d.content, metadata=d.metadata) for d in docs]
    history = state.get("history") or []
    answer = online_nodes.RUNTIME.answer_tool.generate(query, lc_docs, stream=False, history=history)
    answer_text = answer or ""
    return {
        "answer": answer_text,
        "history": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer_text},
        ],
    }
