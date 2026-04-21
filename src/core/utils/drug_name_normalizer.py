from __future__ import annotations

import re

_TRANSLATION = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "、": "",
        "·": "",
        "・": "",
        "‧": "",
        "－": "-",
        "—": "-",
        "–": "-",
        "\u3000": "",
    }
)

_NOISE_RE = re.compile(r"[\s,\.;:!\?'\"]+")
_DASH_RE = re.compile(r"[-_/]+")


def normalize_drug_name(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = value.translate(_TRANSLATION)
    value = _NOISE_RE.sub("", value)
    value = _DASH_RE.sub("", value)
    return value.lower()


def dedup_names(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def build_alias_norms(names: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in dedup_names(names):
        normalized = normalize_drug_name(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
