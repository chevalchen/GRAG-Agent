from __future__ import annotations

from collections.abc import Callable

from src.app.offline_ingestion.state import OfflineIngestionState

LABEL_MAP = {"药品": "Drug", "药物成分": "Ingredient"}


def make_normalize_node() -> Callable[[OfflineIngestionState], dict]:
    def normalize_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}
        parsed_records = list(state.get("parsed_records") or [])
        normalized: list[dict] = []
        for row in parsed_records:
            concept_type = LABEL_MAP.get(str(row.get("raw_label") or "").strip(), "Drug")
            normalized.append(
                {
                    **row,
                    "conceptType": concept_type,
                    "ingredients": list(row.get("ingredients") or []),
                    "effects": list(row.get("effects") or []),
                    "symptoms": list(row.get("symptoms") or []),
                    "diseases": list(row.get("diseases") or []),
                    "syndromes": list(row.get("syndromes") or []),
                    "populations": list(row.get("populations") or []),
                    "adverse_reactions": list(row.get("adverse_reactions") or []),
                }
            )
        return {"parsed_records": normalized}

    return normalize_node
