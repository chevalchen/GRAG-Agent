from typing import List, Optional

from src.app.online_qa.state import OnlineQAState


def validate_has_query(state: OnlineQAState) -> Optional[str]:
    if not state.get("query"):
        return "missing_query"
    return None


def validate_has_docs(state: OnlineQAState) -> Optional[str]:
    if not state.get("docs_final"):
        return "missing_docs_final"
    return None


def append_error(state: OnlineQAState, error: str) -> List[str]:
    return (state.get("errors") or []) + [error]

