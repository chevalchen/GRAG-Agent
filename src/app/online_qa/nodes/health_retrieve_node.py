from __future__ import annotations

import time
from collections.abc import Callable

from src.app.online_qa.state import OnlineQAState


def make_health_retrieve_node() -> Callable[[OnlineQAState], dict]:
    def health_retrieve_node(state: OnlineQAState) -> dict:
        t0 = time.time()
        routing = state.get("routing") or {}
        if not routing.get("use_health_vec"):
            return {"health_docs": [], "metrics": {**(state.get("metrics") or {}), "health_retrieve_seconds": 0.0}}
        metrics = {**(state.get("metrics") or {}), "health_retrieve_seconds": time.time() - t0}
        return {"health_docs": [], "metrics": metrics}

    return health_retrieve_node
