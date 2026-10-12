"""循环终止语义的回归测试：stop 信号强制收尾 + 重复调用拦截。

背景：真实使用中出现过"任务完成后模型凭惯性重开工作流"，
以及 Web 端 SSE 断线自动重连导致同一任务反复执行。这两条测试
锁住循环层的对应防御。替身（ScriptedLLM）只存在于本文件。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from novel_comic_agent.agent import Agent
from novel_comic_agent.llm import BaseLLM
from novel_comic_agent.schema import Function, LLMResponse, Message, ToolCall, ToolResult
from novel_comic_agent.tools import BaseTool, ToolCollection


class ScriptedLLM(BaseLLM):
    """测试本地替身：按顺序弹出预设回复。"""

    name = "scripted"

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def ask(self, messages, tools=None, on_delta=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


class CountingEcho(BaseTool):
    """记录执行次数的回显工具——重复调用拦截靠它验证。"""

    name: str = "echo"
    description: str = "counts calls, returns text"
    parameters: dict[str, Any] = {"type": "object", "properties": {"text": {"type": "string"}}}
    executions: int = 0

    async def execute(self, text: str = "", **_: Any) -> ToolResult:
        self.executions += 1
        return ToolResult(output=text)


class Finish(BaseTool):
    """返回 stop=True 的工具：宣告任务完成。"""

    name: str = "finish"
    description: str = "signals completion"

    async def execute(self, **_: Any) -> ToolResult:
        return ToolResult(output="page composed -> fake.png", stop=True)


def _call(name: str, args: dict) -> ToolCall:
    return ToolCall(function=Function(name=name, arguments=json.dumps(args)))


def test_stop_flag_forces_toolless_wrapup():
    """stop=True 后：再问一轮（不带工具），用那轮文本收尾——确定性终止。"""
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[_call("finish", {})]),
        LLMResponse(content="页已完成：fake.png，5 格。"),
    )
    agent = Agent(llm, ToolCollection(Finish()), max_steps=10)
    answer = asyncio.run(agent.run("画一页"))
    assert answer == "页已完成：fake.png，5 格。"
    assert len(llm.calls) == 2
    assert llm.calls[1]["tools"] is None  # 收尾那轮物理上无法调工具


def test_identical_successful_call_is_replayed_not_reexecuted():
    """同参数成功调用第二次：回放缓存结果并附警告，工具本体不再执行。"""
    echo = CountingEcho()
    llm = ScriptedLLM(
        LLMResponse(tool_calls=[_call("echo", {"text": "hi"})]),
        LLMResponse(tool_calls=[_call("echo", {"text": "hi"})]),  # 原样重发
        LLMResponse(content="done"),
    )
    agent = Agent(llm, ToolCollection(echo), max_steps=10)
    assert asyncio.run(agent.run("x")) == "done"
    assert echo.executions == 1  # 第二次没有真正执行
    replayed = [
        m
        for m in agent.messages
        if m.content and "already succeeded" in m.content
    ]
    assert replayed  # 模型看到了"别重复"的警告


def test_debug_mode_echoes_full_tool_output():
    """debug=True：回显不截断，打印全量参数与全量返回。"""
    seen: list[str] = []

    class LongEcho(BaseTool):
        name: str = "echo"
        description: str = "returns a very long output"
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }

        async def execute(self, text: str = "", **_: Any) -> ToolResult:
            return ToolResult(output="x" * 500)  # 超过 _preview 的 200 截断上限

    llm = ScriptedLLM(
        LLMResponse(tool_calls=[_call("echo", {"text": "hi"})]),
        LLMResponse(content="done"),
    )
    agent = Agent(
        llm,
        ToolCollection(LongEcho()),
        max_steps=10,
        echo=seen.append,
        debug=True,
    )
    asyncio.run(agent.run("x"))
    joined = "\n".join(seen)
    assert "x" * 500 in joined  # 全量返回（500 字符一个不少）
    assert '"text"' in joined and "hi" in joined  # 全量参数原文也在


class MemorySink:
    """会话事务日志替身：内存版，接口与 SessionLog 一致。"""

    def __init__(self, preloaded: list[Message] | None = None) -> None:
        self.items = list(preloaded or [])
        self.appended: list[tuple[str, bool | None]] = []

    def messages(self) -> list[Message]:
        return [m.model_copy(deep=True) for m in self.items]

    def append_message(self, msg: Message, ok: bool | None = None) -> None:
        self.appended.append((msg.role.value, msg.content or "", ok))
        self.items.append(msg)


def _resp_text(text: str) -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop")


def _resp_call_tool(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[ToolCall(
        id="call_1", function=Function(name="echo", arguments='{"text": "hi"}'))])


def test_agent_sink_persists_every_message_and_discards_copy():
    """循环产出先落盘；副本随运行结束丢弃；重跑从日志加载接续。"""
    tool = CountingEcho()
    llm = ScriptedLLM(_resp_call_tool("我来试试"), _resp_text("hi 完成了"))
    sink = MemorySink()
    agent = Agent(llm, ToolCollection(tool), max_steps=6, sink=sink)
    answer = asyncio.run(agent.run("任务"))
    assert answer == "hi 完成了"
    # 四种 message 里的三种按序落盘：user → assistant(tool_calls) → tool(ok=True) → assistant(final)
    assert [(r, ok) for r, _, ok in sink.appended] == [
        ("user", None), ("assistant", None), ("tool", True), ("assistant", None)]
    assert sink.items[-1].content == "hi 完成了"
    assert tool.executions == 1

    # 第二轮：新的 agent 从同一个日志加载副本——上一轮对话还在，循环接着跑
    llm2 = ScriptedLLM(_resp_text("接续完成"))
    agent2 = Agent(llm2, ToolCollection(tool), max_steps=6, sink=sink)
    answer2 = asyncio.run(agent2.run("继续"))
    assert answer2 == "接续完成"
    # 第二轮的第一次调用里能看到上一轮的 user 消息（对话连续性）
    first_prompt_roles = [m.role.value for m in llm2.calls[0]["messages"]]
    assert first_prompt_roles[0] == "system"
    assert "任务" in [m.content for m in llm2.calls[0]["messages"]]
    assert "继续" in [m.content for m in llm2.calls[0]["messages"]]


def test_agent_without_sink_stays_in_memory():
    """不传 sink（CLI/测试）：行为不变，消息只在内存。"""
    llm = ScriptedLLM(_resp_text("直接回答"))
    agent = Agent(llm, ToolCollection(CountingEcho()), max_steps=6)
    assert asyncio.run(agent.run("任务")) == "直接回答"
    assert agent.messages[-1].content == "直接回答"
