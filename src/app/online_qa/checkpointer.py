from __future__ import annotations

import atexit
from pathlib import Path

_checkpointer = {}


def get_checkpointer(db_path: str = ".checkpoints/c9.db"):
    global _checkpointer
    if db_path in _checkpointer:
        return _checkpointer[db_path]
    p = Path(db_path)
    checkpoint_dir = p.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer[db_path] = MemorySaver()
        return _checkpointer[db_path]
    if hasattr(SqliteSaver, "from_conn_string"):
        saver_obj = SqliteSaver.from_conn_string(str(p))
        if hasattr(saver_obj, "__enter__") and hasattr(saver_obj, "__exit__"):
            _checkpointer[db_path] = saver_obj.__enter__()
            atexit.register(saver_obj.__exit__, None, None, None)
            return _checkpointer[db_path]
        _checkpointer[db_path] = saver_obj
        return _checkpointer[db_path]
    _checkpointer[db_path] = SqliteSaver()
    return _checkpointer[db_path]
