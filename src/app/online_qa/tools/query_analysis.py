from __future__ import annotations

class QueryAnalysisTool:
    def __init__(self, router):
        self._router = router

    def analyze(self, query: str) -> dict:
        raw = self._router.analyze_query(query)
        route_map = {
            "hybrid_traditional": "hybrid",
            "graph_rag": "graph_rag",
            "combined": "combined",
        }
        route = route_map.get(raw.recommended_strategy.value, "hybrid")
        return {
            "original_query": query,
            "route": route,
            "keywords": [],
            "intent": raw.reasoning or "",
            "query_complexity": float(getattr(raw, "query_complexity", 0.0)),
            "relationship_intensity": float(getattr(raw, "relationship_intensity", 0.0)),
            "recommended_strategy": str(getattr(raw.recommended_strategy, "value", route)),
        }
