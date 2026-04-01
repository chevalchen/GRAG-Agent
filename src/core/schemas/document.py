from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""


@dataclass
class QueryAnalysis:
    original_query: str
    keywords: list[str] = field(default_factory=list)
    intent: str = ""
    query_complexity: float = 0.0
    relationship_intensity: float = 0.0
    recommended_strategy: str = "hybrid"
