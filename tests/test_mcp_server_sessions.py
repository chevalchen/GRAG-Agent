import json
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.app import mcp_server


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
    def __init__(self, db_path: str):
        self._db_path = db_path

    def invoke(self, state, config=None):
        sid = ((config or {}).get("configurable") or {}).get("thread_id", "")
        con = sqlite3.connect(self._db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel) "
            "VALUES (?, '', 'cp-from-fake', 'task', 0, 'answer')",
            (sid,),
        )
        con.commit()
        con.close()
        return {"answer": f"ok:{state.get('query', '')}"}


class TestMcpSessionTools(unittest.TestCase):
    def test_tcm_query_then_session_list_shows_session_id(self):
        db_path = _create_db_file()
        try:
            runtime = SimpleNamespace(
                config=SimpleNamespace(checkpointer_path=db_path),
                graph=_FakeGraph(db_path),
            )
            with patch("src.app.mcp_server._get_runtime", return_value=runtime):
                mcp_server.tcm_query("测试问题", session_id="sid-visible")
                payload = json.loads(mcp_server.session_list())
            ids = [item.get("session_id") for item in payload.get("sessions", [])]
            self.assertIn("sid-visible", ids)
        finally:
            os.unlink(db_path)

    def test_session_list_has_stable_minimal_field(self):
        db_path = _create_db_file()
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel) "
                "VALUES ('sid-a', '', 'cp-a', 'task', 0, 'answer')"
            )
            con.commit()
            con.close()
            sessions = mcp_server._list_history_sessions(db_path)
            self.assertTrue(sessions)
            self.assertTrue(all("session_id" in row for row in sessions))
        finally:
            os.unlink(db_path)

    def test_session_delete_returns_session_not_found(self):
        db_path = _create_db_file()
        try:
            result = mcp_server._delete_session(db_path, "missing")
            self.assertEqual(result.get("code"), "SESSION_NOT_FOUND")
        finally:
            os.unlink(db_path)

    def test_session_delete_invalid_argument(self):
        db_path = _create_db_file()
        try:
            runtime = SimpleNamespace(config=SimpleNamespace(checkpointer_path=db_path))
            with patch("src.app.mcp_server._get_runtime", return_value=runtime):
                payload = json.loads(mcp_server.session_delete("  "))
            self.assertEqual(payload.get("code"), "INVALID_ARGUMENT")
        finally:
            os.unlink(db_path)

    def test_session_delete_backend_error(self):
        bad_db_path = tempfile.mkdtemp(prefix="mcp-delete-bad-")
        try:
            result = mcp_server._delete_session(bad_db_path, "sid-any")
            self.assertEqual(result.get("code"), "INTERNAL_ERROR")
        finally:
            os.rmdir(bad_db_path)

    def test_session_delete_ok_contains_deleted_target(self):
        db_path = _create_db_file()
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id) VALUES ('sid-ok', '', 'cp-ok')"
            )
            con.commit()
            con.close()
            result = mcp_server._delete_session(db_path, "sid-ok")
            self.assertEqual(result.get("code"), "OK")
            self.assertEqual((result.get("detail") or {}).get("session_id"), "sid-ok")
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
