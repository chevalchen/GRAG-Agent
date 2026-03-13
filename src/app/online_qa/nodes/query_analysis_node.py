from src.app.online_qa import nodes as online_nodes
from src.app.online_qa.state import OnlineQAState


def query_analysis_node(state: OnlineQAState) -> dict:
    if online_nodes.RUNTIME is None:
        return {"error": "runtime_not_initialized"}
    query = state.get("query", "")
    if not query:
        return {"analysis": {"original_query": "", "route": "hybrid", "keywords": [], "intent": "empty_query"}}
    analysis = online_nodes.RUNTIME.query_analysis_tool.analyze(query)
    return {"analysis": analysis}
