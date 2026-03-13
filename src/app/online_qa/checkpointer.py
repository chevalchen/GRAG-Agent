from __future__ import annotations

import atexit
from pathlib import Path

_checkpointer = None


def get_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    checkpoint_dir = Path(".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / "c9.db"
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()
        return _checkpointer
    if hasattr(SqliteSaver, "from_conn_string"):
        saver_obj = SqliteSaver.from_conn_string(str(db_path))
        if hasattr(saver_obj, "__enter__") and hasattr(saver_obj, "__exit__"):
            _checkpointer = saver_obj.__enter__()
            atexit.register(saver_obj.__exit__, None, None, None)
            return _checkpointer
        _checkpointer = saver_obj
        return _checkpointer
    _checkpointer = SqliteSaver()
    return _checkpointer
