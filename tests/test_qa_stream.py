"""SSE endpoint smoke test — verifies that /api/v1/qa/stream emits
event:token, event:suggest, and event:done frames."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_mock_runtime(answer_text: str, next_questions: list[str], session_id: str):
    """Build a mock ServerRuntime whose graph.astream_events yields token events."""

    # Simulate on_chat_model_stream events from the answer node
    chunk_obj = SimpleNamespace(content=answer_text)
    events = [
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "answer"},
            "data": {"chunk": chunk_obj},
        },
    ]

    async def _astream_events(input_state, *, config=None, version="v2"):
        for ev in events:
            yield ev

    # Mock get_state to return a snapshot with next_questions
    snapshot = MagicMock()
    snapshot.values = {"next_questions": next_questions, "answer": answer_text}

    graph = MagicMock()
    graph.astream_events = _astream_events
    graph.get_state = MagicMock(return_value=snapshot)

    runtime = MagicMock()
    runtime.graph = graph
    return runtime


@pytest.fixture()
def client():
    mock_runtime = _make_mock_runtime(
        answer_text="乌鸡白凤丸主要用于妇女气血两虚引起的月经不调等症状。",
        next_questions=["哪些人群慎用？", "有哪些副作用？", "可以长期服用吗？"],
        session_id="test-session-001",
    )
    with patch("src.app.api_server.get_runtime", return_value=mock_runtime):
        from src.app.api_server import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def _parse_sse(raw: str) -> list[dict]:
    """Parse an SSE response body into a list of {event, data} dicts."""
    events = []
    for block in raw.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        ev: dict = {"event": "message", "data": {}}
        for line in lines:
            if line.startswith("event:"):
                ev["event"] = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    ev["data"] = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    ev["data"] = {}
        if ev["data"]:
            events.append(ev)
    return events


def test_stream_emits_token_suggest_done(client: TestClient):
    response = client.post(
        "/api/v1/qa/stream",
        json={"question": "乌鸡白凤丸的适应症是什么？"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse(response.text)
    event_types = [e["event"] for e in events]

    assert "token" in event_types, f"expected token event, got {event_types}"
    assert "suggest" in event_types, f"expected suggest event, got {event_types}"
    assert "done" in event_types, f"expected done event, got {event_types}"

    token_ev = next(e for e in events if e["event"] == "token")
    assert "乌鸡白凤丸" in token_ev["data"]["chunk"]

    suggest_ev = next(e for e in events if e["event"] == "suggest")
    assert isinstance(suggest_ev["data"]["questions"], list)
    assert len(suggest_ev["data"]["questions"]) == 3

    done_ev = next(e for e in events if e["event"] == "done")
    assert "session_id" in done_ev["data"]
    assert "trace_id" in done_ev["data"]


def test_stream_empty_question_returns_400(client: TestClient):
    response = client.post("/api/v1/qa/stream", json={"question": "  "})
    assert response.status_code == 400
