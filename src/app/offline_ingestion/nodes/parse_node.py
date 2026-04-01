from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable

from src.app.offline_ingestion.state import OfflineIngestionState


def make_parse_node() -> Callable[[OfflineIngestionState], dict]:
    def parse_node(state: OfflineIngestionState) -> dict:
        if state.get("error"):
            return {}
        path = state.get("medical_entities_path")
        if not path:
            return {"parsed_records": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            return {"error": f"parse_node_read_error: {e}"}

        drug_profiles: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: {
                "ingredients": set(),
                "effects": set(),
                "symptoms": set(),
                "diseases": set(),
                "syndromes": set(),
                "populations": set(),
                "adverse_reactions": set(),
            }
        )
        label_field_map = {
            "药物成分": "ingredients",
            "中药功效": "effects",
            "症状": "symptoms",
            "疾病": "diseases",
            "证候": "syndromes",
            "人群": "populations",
            "不良反应": "adverse_reactions",
        }
        for row in rows or []:
            annotations = row.get("annotations") or []
            drugs: set[str] = set()
            row_fields: dict[str, set[str]] = defaultdict(set)
            for ann in annotations:
                label = (ann.get("label") or "").strip()
                entity = (ann.get("entity") or "").strip()
                if not entity:
                    continue
                if label == "药品":
                    drugs.add(entity)
                    continue
                field = label_field_map.get(label)
                if field:
                    row_fields[field].add(entity)
            for drug_name in drugs:
                profile = drug_profiles[drug_name]
                for field, values in row_fields.items():
                    profile[field].update(values)

        parsed_records = []
        for drug_name, profile in sorted(drug_profiles.items(), key=lambda x: x[0]):
            parsed_records.append(
                {
                    "drug_name": drug_name,
                    "ingredients": sorted(profile["ingredients"]),
                    "effects": sorted(profile["effects"]),
                    "symptoms": sorted(profile["symptoms"]),
                    "diseases": sorted(profile["diseases"]),
                    "syndromes": sorted(profile["syndromes"]),
                    "populations": sorted(profile["populations"]),
                    "adverse_reactions": sorted(profile["adverse_reactions"]),
                    "raw_label": "药品",
                }
            )
        return {"parsed_records": parsed_records}

    return parse_node
