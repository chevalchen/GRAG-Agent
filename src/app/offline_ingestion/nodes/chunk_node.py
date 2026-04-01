from __future__ import annotations

import json
from collections.abc import Callable

from src.app.offline_ingestion.state import OfflineIngestionState


def _sliding_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean_text = (text or "").strip()
    if not clean_text:
        return []
    if chunk_size <= overlap:
        chunk_size = overlap + 1
    step = chunk_size - overlap
    chunks: list[str] = []
    for i in range(0, len(clean_text), step):
        part = clean_text[i : i + chunk_size].strip()
        if part:
            chunks.append(part)
        if i + chunk_size >= len(clean_text):
            break
    return chunks


def make_chunk_node(chunk_size: int = 500, overlap: int = 50) -> Callable[[OfflineIngestionState], dict]:
    def chunk_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}
        path = state.get("literature_path")
        if not path:
            return {"chunk_records": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            return {"error": f"chunk_node_read_error: {e}"}

        chunk_records: list[dict] = []
        source = "Traditional_Chinese_Medical_Literature"
        for row in rows or []:
            rid = row.get("id")
            text = row.get("text") or ""
            chunks = _sliding_chunks(text, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk_text in enumerate(chunks):
                chunk_records.append(
                    {
                        "text": chunk_text,
                        "chunk_type": "passage",
                        "source": source,
                        "answer": "",
                        "node_id": f"tcm_lit_{rid}_{idx}",
                    }
                )

            for qa_idx, qa in enumerate(row.get("annotations") or []):
                q = (qa.get("Q") or "").strip()
                a = (qa.get("A") or "").strip()
                if not q:
                    continue
                chunk_records.append(
                    {
                        "text": q,
                        "chunk_type": "qa",
                        "source": source,
                        "answer": a,
                        "node_id": f"tcm_lit_{rid}_qa_{qa_idx}",
                    }
                )
        return {"chunk_records": chunk_records}

    return chunk_node
