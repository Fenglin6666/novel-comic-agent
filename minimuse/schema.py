"""核心数据结构——整个 agent 的"线协议"。

对应 nanoMuse 的 nanomuse/schema.py。四种角色的消息、工具调用、工具结果。
这一层不依赖任何其他模块；先把它定稳，上面每一层都靠它说话。
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    """消息的四种角色，与 OpenAI 线协议一致。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


def new_id(prefix: str = "call") -> str:
    """生成一个短随机 id（工具调用/消息追踪用）。

    Args:
        prefix: id 前缀，便于日志里辨认来源。
    Returns:
        形如 "call_a1b2c3d4e5f6" 的唯一字符串。
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Function(BaseModel):
    """模型点名的工具调用：名字 + JSON 字符串形式的参数。"""

    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        """解析模型输出的参数 JSON。

        关键设计：解析失败不抛异常，而是放进 __raw__ 键。这样一次坏调用
        会变成一条"可纠正的失败结果"喂回给模型，而不是让整个回合崩溃——
        模型看到错误后通常自己就会重试并修好格式。
        """
        if not self.arguments or not self.arguments.strip():
            return {}
        try:
            data = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {"__raw__": self.arguments}
        if not isinstance(data, dict):
            return {"value": data}
        return data


class ToolCall(BaseModel):
    """一次工具调用：id + 函数名 + JSON 参数。id 用于把工具结果配对回调用。"""

    id: str = Field(default_factory=new_id)
    type: Literal["function"] = "function"
    function: Function

    @property
    def name(self) -> str:
        """快捷属性：工具名。"""
        return self.function.name

    @property
    def arguments(self) -> dict[str, Any]:
        """快捷属性：解析后的参数字典（解析失败落 __raw__，见 Function）。"""
        return self.function.parsed_arguments()


class ToolResult(BaseModel):
    """工具的结果。约定：失败也是数据（error 字段），工具永远不抛异常。"""

    output: str = ""
    error: str | None = None
    # 阶段 4 的占位：浏览器/截图工具会用到，先放进线协议
    images: list[str] | None = None
    # 工具宣告"任务完成"：循环收到后强制一轮无工具收尾（确定性终止）
    stop: bool = False

    @property
    def ok(self) -> bool:
        """是否成功（无 error 字段）。"""
        return self.error is None

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        """构造失败结果的工厂；错误文本截断到 4000 字符防止撑爆上下文。"""
        return cls(error=error[:4000])

    def for_model(self) -> str:
        """作为 tool 消息内容回填给模型的文本。"""
        return self.error if self.error is not None else self.output


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    # ---------------------------------------------------------------- factories
    @classmethod
    def system(cls, content: str) -> Message:
        """构造 system 消息。"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """构造 user 消息。"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        """构造 assistant 消息（可带 tool_calls）。"""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, content: str, call_id: str, name: str | None = None) -> Message:
        """构造 tool 结果消息；call_id 必须对应某次 ToolCall 的 id。"""
        return cls(role=Role.TOOL, content=content, tool_call_id=call_id, name=name)

    def to_wire(self) -> dict[str, Any]:
        """转成 OpenAI 兼容端点的消息格式（其他协议在各自的 LLM 适配层里转）。"""
        data: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    },
                }
                for c in self.tool_calls
            ]
        if self.role is Role.TOOL:
            data["tool_call_id"] = self.tool_call_id
            data["content"] = self.content or ""
            if self.name:
                data["name"] = self.name
        return data


class LLMResponse(BaseModel):
    """一次模型调用的返回：文本、工具调用、结束原因、用量。"""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] = Field(default_factory=dict)

    def to_message(self) -> Message:
        """把模型回复包成一条 assistant 消息，方便追加进对话历史。"""
        return Message.assistant(self.content, self.tool_calls or None)
