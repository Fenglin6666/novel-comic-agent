"""LLM 抽象层：一个方法走天下。

对应 nanoMuse 的 nanomuse/llm/base.py + openai_chat.py。
阶段 1 先不做流式（阶段 3 再加）。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from novel_comic_agent.schema import Function, LLMResponse, Message, ToolCall

DeltaCallback = Callable[[str], None]


class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    async def ask(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        """发一段对话，拿回一个 assistant 回合（文本和/或工具调用）。"""


class OpenAILLM(BaseLLM):
    """任何 OpenAI 兼容端点：OpenAI、DeepSeek、OpenRouter、本地 vLLM/Ollama。

    配置全走环境变量，代码里不放任何密钥：
      OPENAI_API_KEY   密钥
      OPENAI_BASE_URL  端点，如 https://api.deepseek.com/v1（省略则用官方默认）
      MINIMUSE_MODEL   模型名，如 deepseek-chat / gpt-4o-mini
    """

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """创建 OpenAI 兼容客户端；显式参数优先于环境变量。

        Args:
            model: 模型名，缺省读 MINIMUSE_MODEL（再缺省 deepseek-chat）。
            base_url: 端点，缺省读 OPENAI_BASE_URL（再缺省官方端点）。
            api_key: 密钥，缺省读 OPENAI_API_KEY。
        """
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model or os.environ.get("MINIMUSE_MODEL", "deepseek-chat")

    async def ask(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        """发一段对话，拿回一个 assistant 回复。

        Args:
            messages: 完整对话历史（含 system 与 tool 消息）。
            tools: OpenAI 风格 function schema 列表；None 表示禁用工具。
            on_delta: 文本增量回调（当前不做流式，整段回调一次）。
        Returns:
            LLMResponse：文本、工具调用、结束原因与用量。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
        }
        if tools:
            kwargs["tools"] = tools
        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCall(
                id=tc.id,
                function=Function(name=tc.function.name, arguments=tc.function.arguments or "{}"),
            )
            for tc in (msg.tool_calls or [])
        ]
        if msg.content and on_delta:
            on_delta(msg.content)  # 阶段 1 不做流式，整段回调一次
        usage: dict[str, Any] = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )


