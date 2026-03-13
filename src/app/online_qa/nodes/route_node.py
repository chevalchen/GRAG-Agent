from src.app.online_qa.state import OnlineQAState


def route_node(state: OnlineQAState) -> dict:
    analysis = state.get("analysis")
    if isinstance(analysis, dict):
        route = analysis.get("route", "hybrid")
    else:
        route = getattr(analysis, "route", "hybrid")
    return {"route": route}
