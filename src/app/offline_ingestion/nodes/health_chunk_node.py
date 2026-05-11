from __future__ import annotations

import json
import os
from collections.abc import Callable

from src.app.offline_ingestion.state import OfflineIngestionState


def _find_health_files(root_dir: str) -> list[str]:
    """查找 health 目录下的所有 jsonl 文件"""
    health_dir = os.path.join(root_dir, "health")
    if not os.path.isdir(health_dir):
        return []
    files = []
    for fname in os.listdir(health_dir):
        if fname.endswith(".jsonl"):
            files.append(os.path.join(health_dir, fname))
    return sorted(files)


def _flatten_questions(raw_questions) -> list[str]:
    if not isinstance(raw_questions, list):
        return []
    flattened: list[str] = []
    for item in raw_questions:
        values = item if isinstance(item, list) else [item]
        for value in values:
            text = str(value or "").strip()
            if text:
                flattened.append(text)
    return flattened


def make_health_chunk_node() -> Callable[[OfflineIngestionState], dict]:
    def health_chunk_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}

        root_dir = state.get("recipe_dir", "")
        health_files = _find_health_files(root_dir)

        if not health_files:
            metrics = {**(state.get("metrics") or {}), "health_files_found": 0, "health_questions_expanded": 0}
            return {"health_chunk_records": [], "metrics": metrics}

        chunk_records: list[dict] = []
        file_idx = 0
        lines_seen = 0
        invalid_json_lines = 0
        failed_files = 0

        for file_path in health_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        lines_seen += 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            invalid_json_lines += 1
                            continue

                        questions = _flatten_questions(row.get("questions"))
                        raw_answers = row.get("answers") or []
                        answers = raw_answers if isinstance(raw_answers, list) else [raw_answers]

                        for q_idx, q_text in enumerate(questions):
                            answer = str(answers[q_idx] if q_idx < len(answers) else "").strip()
                            chunk_records.append(
                                {
                                    "text": q_text,
                                    "chunk_type": "qa",
                                    "source": "health_popularization",
                                    "answer": answer,
                                    "node_id": f"health_{file_idx}_{line_idx}_{q_idx}",
                                    "doc_type": "health_popularization",
                                }
                            )
            except Exception:
                failed_files += 1
            file_idx += 1

        metrics = {
            **(state.get("metrics") or {}),
            "health_files_found": len(health_files),
            "health_files_failed": failed_files,
            "health_lines_seen": lines_seen,
            "health_invalid_json_lines": invalid_json_lines,
            "health_questions_expanded": len(chunk_records),
        }
        return {"health_chunk_records": chunk_records, "metrics": metrics}

    return health_chunk_node
