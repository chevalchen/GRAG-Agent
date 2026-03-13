from langchain_core.documents import Document as LCDocument

from src.app.online_qa import nodes as online_nodes
from src.app.online_qa.state import OnlineQAState


def fuse_node(state: OnlineQAState) -> dict:
    if online_nodes.RUNTIME is None:
        return {"error": "runtime_not_initialized"}
    if state.get("route") == "combined":
        fused_docs = online_nodes.RUNTIME.fusion_tool.fuse(
            state.get("graph_docs", []), state.get("hybrid_docs", []), online_nodes.RUNTIME.top_k
        )
    else:
        fused_docs = state.get("hybrid_docs") or state.get("graph_docs") or []
    docs_final = [LCDocument(page_content=d.content, metadata=d.metadata) for d in fused_docs]
    return {"fused_docs": fused_docs, "docs_final": docs_final}
