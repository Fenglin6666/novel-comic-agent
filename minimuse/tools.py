"""工具骨架：BaseTool、safe_execute、ToolCollection。

对应 nanoMuse 的 nanomuse/tools/base.py。一个工具同时服务两个读者：
模型读 name/description/parameters（JSON Schema）；
（阶段 2 起）Sentinel 读 risk/assess()。
场景相关的具体工具在 comic_tools.py——这里只放与场景无关的部分。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from minimuse.schema import ToolResult


class BaseTool(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行工具。约定：可预期的失败返回 ToolResult.fail，不抛异常。"""

    def to_param(self) -> dict[str, Any]:
        """给模型的 OpenAI 风格 function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


async def safe_execute(tool: BaseTool, args: dict[str, Any]) -> ToolResult:
    """把工具执行的一切异常变成失败结果——失败也是数据，喂回给模型自纠。"""
    if "__raw__" in args:
        return ToolResult.fail(
            "arguments is not valid JSON (probably cut off in transit); "
            f"call the tool again with valid JSON: {str(args['__raw__'])[:200]}"
        )
    try:
        return await tool.execute(**args)
    except TypeError as exc:
        return ToolResult.fail(f"bad arguments for {tool.name}: {exc}")
    except Exception as exc:  # noqa: BLE001 — 工具失败必须变成结果，不许炸掉回合
        return ToolResult.fail(f"{tool.name} failed: {type(exc).__name__}: {exc}")


class ToolCollection:
    """一组工具：按名字查找、导出给模型的 schema 列表。"""

    def __init__(self, *tools: BaseTool) -> None:
        """收集工具；重名直接抛错（模型点名会有歧义）。"""
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """按名字找工具；找不到返回 None（调用方负责变成可纠错报错）。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """全部工具名（排序后），用于报错里列出可用工具。"""
        return sorted(self._tools)

    def to_params(self) -> list[dict[str, Any]]:
        """导出给模型的 function schema 列表。"""
        return [t.to_param() for t in self]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
