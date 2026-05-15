"""
Integration test: sufficiency loop through the graph.

Uses a mock graph to verify:
1. First pass: insufficient → router → second pass: sufficient → answer
2. Max-iters guard: insufficient but iteration_count already at limit → answer
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.app.online_qa.graphs.online_qa_graph import _route_intent, _route_sufficiency
from src.app.online_qa.state import OnlineQAState


class TestRouteIntentEdge:
    def test_symptom_to_drug_routes_to_milvus_search(self):
        state: OnlineQAState = {
            "routing": {"intent": "symptom_to_drug", "complexity_level": "simple"},
            "iteration_count": 0,
        }
        assert _route_intent(state) == "milvus_search"

    def test_theory_routes_to_milvus_search(self):
        state: OnlineQAState = {
            "routing": {"intent": "theory", "complexity_level": "simple"},
            "iteration_count": 0,
        }
        assert _route_intent(state) == "milvus_search"

    def test_conditions_exact_routes_to_neo4j_search(self):
        state: OnlineQAState = {
            "routing": {"intent": "conditions_exact", "complexity_level": "complex"},
            "iteration_count": 0,
        }
        assert _route_intent(state) == "neo4j_search"

    def test_chat_routes_to_answer(self):
        state: OnlineQAState = {
            "routing": {"intent": "chat", "complexity_level": "simple"},
            "iteration_count": 0,
        }
        assert _route_intent(state) == "answer"

    def test_max_iters_reached_forces_answer(self):
        """iteration_count > max_iters (1) must force answer at router."""
        state: OnlineQAState = {
            "routing": {"intent": "symptom_to_drug", "complexity_level": "simple"},
            "iteration_count": 2,
        }
        assert _route_intent(state) == "answer"

    def test_no_routing_defaults_to_milvus_search(self):
        state: OnlineQAState = {"iteration_count": 0}
        assert _route_intent(state) == "milvus_search"


class TestRouteSufficiencyEdge:
    def test_sufficient_goes_to_answer(self):
        state: OnlineQAState = {"sufficiency": "sufficient", "iteration_count": 0}
        assert _route_sufficiency(state) == "answer"

    def test_insufficient_first_pass_goes_to_router(self):
        state: OnlineQAState = {"sufficiency": "insufficient", "iteration_count": 1}
        assert _route_sufficiency(state) == "router"

    def test_insufficient_but_max_iters_forces_answer(self):
        """iteration_count already >= max_iters (1) → answer despite insufficient."""
        state: OnlineQAState = {"sufficiency": "insufficient", "iteration_count": 2}
        assert _route_sufficiency(state) == "answer"

    def test_none_sufficiency_treated_as_sufficient(self):
        state: OnlineQAState = {"sufficiency": None, "iteration_count": 0}
        assert _route_sufficiency(state) == "answer"
