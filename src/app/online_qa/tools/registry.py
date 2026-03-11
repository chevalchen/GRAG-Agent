from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set


@dataclass(frozen=True)
class ToolContext:
    tools: Dict[str, Any]
    allowed: Set[str]

    def get(self, name: str) -> Any:
        if name not in self.allowed:
            raise PermissionError(f"Tool not allowed: {name}")
        return self.tools[name]


def make_tool_context(tools: Dict[str, Any], allowed: Iterable[str]) -> ToolContext:
    return ToolContext(tools=tools, allowed=set(allowed))

