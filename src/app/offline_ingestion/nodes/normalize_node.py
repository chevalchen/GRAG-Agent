from __future__ import annotations

import hashlib
from collections.abc import Callable

from src.app.offline_ingestion.state import OfflineIngestionState
from src.core.utils.drug_name_normalizer import build_alias_norms, dedup_names, normalize_drug_name

LABEL_MAP = {"药品": "Drug", "药物成分": "Ingredient"}


def make_normalize_node() -> Callable[[OfflineIngestionState], dict]:
    def normalize_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}
        parsed_records = list(state.get("parsed_records") or [])
        normalized: list[dict] = []
        for row in parsed_records:
            concept_type = LABEL_MAP.get(str(row.get("raw_label") or "").strip(), "Drug")
            drug_name = str(row.get("drug_name") or "").strip()
            canonical_name = drug_name
            aliases = dedup_names([drug_name, *list(row.get("aliases") or [])])
            alias_norms = build_alias_norms([canonical_name, *aliases])
            node_id = f"drug_{hashlib.md5(canonical_name.encode('utf-8')).hexdigest()[:12]}" if canonical_name else ""
            normalized.append(
                {
                    **row,
                    "conceptType": concept_type,
                    "node_id": node_id,
                    "canonical_name": canonical_name,
                    "normalized_name": normalize_drug_name(canonical_name),
                    "aliases": aliases,
                    "alias_norms": alias_norms,
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
