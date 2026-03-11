import time
from typing import Dict

from src.app.online_qa.agents.router_agent import RouterAgent
from src.app.online_qa.state import OnlineQAState
from src.app.online_qa.validators import append_error, validate_has_query


class SupervisorAgent:
    async def run(self, state: OnlineQAState, router_agent: RouterAgent) -> Dict:
        start = time.perf_counter()
        err = validate_has_query(state)
        if err:
            return {"route": "hybrid_traditional", "errors": append_error(state, err)}

        try:
            analysis = await router_agent.analyze(state["query"])
            if not analysis:
                return {"route": "hybrid_traditional", "errors": append_error(state, "missing_analysis")}
            metrics = {**state["metrics"], "analyze_seconds": time.perf_counter() - start, "route": analysis.recommended_strategy.value}
            return {"analysis": analysis, "route": analysis.recommended_strategy.value, "metrics": metrics}
        except Exception as e:
            metrics = {**state["metrics"], "analyze_seconds": time.perf_counter() - start, "route": "hybrid_traditional"}
            return {"route": "hybrid_traditional", "errors": append_error(state, f"supervisor_analyze_error:{e}"), "metrics": metrics}

