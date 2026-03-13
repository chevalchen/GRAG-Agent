from src.app.online_qa import nodes as online_nodes
from src.app.online_qa.state import OnlineQAState
from src.core.schemas.document import Document


def graph_retrieve_node(state: OnlineQAState) -> dict:
    if online_nodes.RUNTIME is None:
        return {"error": "runtime_not_initialized"}
    query = state.get("query", "")
    docs = online_nodes.RUNTIME.graph_tool.search(query, online_nodes.RUNTIME.top_k)
    graph_docs = [
        Document(
            content=d.page_content,
            metadata=dict(d.metadata),
            source="graph_rag",
        )
        for d in docs
    ]
    return {"graph_docs": graph_docs}
