import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app.api_server import app


def _create_db_file() -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    con = sqlite3.connect(tmp.name)
    cur = con.cursor()
    cur.execute(
        """CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT, checkpoint BLOB, metadata BLOB
        )"""
    )
    cur.execute(
        """CREATE TABLE writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT, value BLOB
        )"""
    )
    con.commit()
    con.close()
    return tmp.name


class _FakeGraph:
    def invoke(self, state, config=None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id", "")
        return {"answer": f"ok:{state.get('query')}:{sid}"}

    async def ainvoke(self, state, config=None):
        return self.invoke(state, config)


class TestApiServer(unittest.TestCase):
    def test_cors_preflight_options(self):
        # Default FRONTEND_ORIGINS includes http://localhost:5173 (Vite dev server)
        with TestClient(app) as client:
            response = client.options(
                "/api/v1/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "x-trace-id,content-type",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_health_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("ok"))
        self.assertEqual((payload.get("data") or {}).get("status"), "ok")
        self.assertTrue(payload.get("trace_id"))

    def test_qa_success(self):
        runtime = SimpleNamespace(config=SimpleNamespace(checkpointer_path="ignored"), graph=_FakeGraph())
        with patch("src.app.api_server._get_runtime", return_value=runtime):
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/qa",
                    json={"question": "测试问题", "session_id": "sid-qa"},
                    headers={"x-trace-id": "trace-qa"},
                )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("trace_id"), "trace-qa")
        data = payload.get("data") or {}
        self.assertEqual(data.get("session_id"), "sid-qa")
        self.assertIn("ok:测试问题:sid-qa", data.get("answer", ""))

    def test_qa_validation_error(self):
        runtime = SimpleNamespace(config=SimpleNamespace(checkpointer_path="ignored"), graph=_FakeGraph())
        with patch("src.app.api_server._get_runtime", return_value=runtime):
            with TestClient(app) as client:
                response = client.post("/api/v1/qa", json={"session_id": "sid-qa"})
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertFalse(payload.get("ok"))
        self.assertEqual((payload.get("error") or {}).get("code"), "VALIDATION_ERROR")

    def test_qa_empty_question(self):
        runtime = SimpleNamespace(config=SimpleNamespace(checkpointer_path="ignored"), graph=_FakeGraph())
        with patch("src.app.api_server._get_runtime", return_value=runtime):
            with TestClient(app) as client:
                response = client.post("/api/v1/qa", json={"question": "   ", "session_id": "sid-qa"})
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertFalse(payload.get("ok"))
        self.assertEqual((payload.get("error") or {}).get("code"), "INVALID_ARGUMENT")

    def test_sessions_list_and_delete(self):
        db_path = _create_db_file()
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id) VALUES ('sid-a', '', 'cp-a')"
            )
            con.commit()
            con.close()
            runtime = SimpleNamespace(config=SimpleNamespace(checkpointer_path=db_path), graph=_FakeGraph())
            with patch("src.app.api_server._get_runtime", return_value=runtime):
                with TestClient(app) as client:
                    list_response = client.get("/api/v1/sessions")
                    del_response = client.delete("/api/v1/sessions/sid-a")
                    missing_response = client.delete("/api/v1/sessions/sid-a")
            self.assertEqual(list_response.status_code, 200)
            sessions = ((list_response.json().get("data") or {}).get("sessions")) or []
            self.assertTrue(any(item.get("session_id") == "sid-a" for item in sessions))

            self.assertEqual(del_response.status_code, 200)
            self.assertTrue(del_response.json().get("ok"))
            self.assertEqual(((del_response.json().get("data") or {}).get("session_id")), "sid-a")

            self.assertEqual(missing_response.status_code, 404)
            self.assertFalse(missing_response.json().get("ok"))
            self.assertEqual((missing_response.json().get("error") or {}).get("code"), "SESSION_NOT_FOUND")
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
