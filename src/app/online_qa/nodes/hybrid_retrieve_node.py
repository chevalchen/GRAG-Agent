from src.app.online_qa import nodes as online_nodes
from src.app.online_qa.state import OnlineQAState
from src.core.schemas.document import Document


def hybrid_retrieve_node(state: OnlineQAState) -> dict:
    if online_nodes.RUNTIME is None:
        return {"error": "runtime_not_initialized"}
    query = state.get("query", "")
    docs = online_nodes.RUNTIME.hybrid_tool.search(query, online_nodes.RUNTIME.top_k)
    hybrid_docs = [
        Document(
            content=d.page_content,
            metadata=dict(d.metadata),
            source="hybrid",
        )
        for d in docs
    ]
    return {"hybrid_docs": hybrid_docs}
