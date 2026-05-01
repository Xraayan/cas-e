from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class RunContext:
    """Lightweight runtime context compatible with previous tool signatures."""

    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def function_tool() -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Tag async functions so they can be discovered by the local tool registry."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        setattr(func, "_is_function_tool", True)
        setattr(func, "_tool_name", func.__name__)
        setattr(func, "_tool_description", inspect.getdoc(func) or "")
        return func

    return decorator


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    has_query_arg: bool

    def to_gemini_declaration(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        if self.has_query_arg:
            properties["query"] = {
                "type": "string",
                "description": "User query text for the tool.",
            }
            required.append("query")

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, func: Callable[..., Awaitable[Any]]) -> None:
        if not getattr(func, "_is_function_tool", False):
            return

        signature = inspect.signature(func)
        has_query_arg = "query" in signature.parameters

        spec = ToolSpec(
            name=getattr(func, "_tool_name", func.__name__),
            description=getattr(func, "_tool_description", inspect.getdoc(func) or ""),
            func=func,
            has_query_arg=has_query_arg,
        )
        self._tools[spec.name] = spec

    def register_many(self, functions: list[Callable[..., Awaitable[Any]]]) -> None:
        for func in functions:
            self.register(func)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def to_gemini_tools(self) -> list[dict[str, Any]]:
        if not self._tools:
            return []
        declarations = [spec.to_gemini_declaration() for spec in self._tools.values()]
        return [{"function_declarations": declarations}]

    async def execute(self, name: str, arguments: dict[str, Any], context: RunContext) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            if spec.has_query_arg:
                result = await spec.func(context, str(arguments.get("query", "")))
            else:
                result = await spec.func(context)
        except Exception as exc:
            return json.dumps({"error": f"Tool execution failed: {exc}"})

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=True)
