from __future__ import annotations

from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState


def make_health_retrieve_node() -> Callable[[OnlineQAState], dict]:
    def health_retrieve_node(state: OnlineQAState) -> dict:
        routing = state.get("routing") or {}
        if not routing.get("use_health_vec"):
            return {"health_docs": []}
        return {"health_docs": []}

    return health_retrieve_node
