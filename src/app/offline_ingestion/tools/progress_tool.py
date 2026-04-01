from __future__ import annotations

from typing import Any


class ProgressStoreTool:
    def __init__(self, builder):
        self._builder = builder

    def save(self, state: dict[str, Any], path: str) -> None:
        self._builder.save_progress(
            path,
            state.get("total_files", 0),
            state.get("processed", 0),
        )

    def load(self, path: str) -> dict | None:
        self._builder.load_progress()
        return {
            "processed_files": sorted(self._builder.processed_files),
            "batch_id": self._builder.current_batch,
        }

    def is_done(self, file_path: str, path: str) -> bool:
        return file_path in self._builder.processed_files

