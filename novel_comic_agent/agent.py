"""agent 循环：模型提议 → 执行工具 → 结果回填 → 重复。

对应 nanoMuse 的 nanomuse/agent/core.py::run()，只保留最必要的骨架，
但保留五件"真踩过坑"才有的防御：未知工具的报错里列出全部工具、
坏参数变成失败结果喂回模型、同参数成功调用不重复执行、
工具的 stop 信号触发无工具强制收尾、步数耗尽后的无工具收尾。
工具结果也会回显一行预览——真实场景里，看不见工具在干什么等于盲飞。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from novel_comic_agent.llm import BaseLLM
from novel_comic_agent.schema import Message, ToolResult
from novel_comic_agent.tools import ToolCollection, safe_execute

Echo = Callable[[str], None]

SYSTEM_PROMPT = """You are novel-comic-agent, a small task agent. You answer in the user's language.

You can call tools for facts you cannot know or compute yourself. After a tool
returns, read the result carefully and continue. When the task is done, reply
with the final answer in plain text and do not call any more tools.

Rules:
- One step at a time; check tool results before the next call.
- If a tool fails, read the error message: it usually tells you how to fix the call.
- Never invent tool results.
"""


class Agent:
    """最小 agent 循环：模型提议 → 执行工具 → 结果回填 → 重复。

    Attributes:
        messages: 本 agent 实例的完整对话历史（含工具结果）。
        _done: 成功调用缓存（"工具名:参数JSON" → 结果），重复调用直接回放。
        debug: 调试模式——回显不截断，打印工具的全量参数与全量返回。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolCollection,
        max_steps: int = 12,
        system_prompt: str = SYSTEM_PROMPT,
        echo: Echo | None = None,
        debug: bool = False,
    ) -> None:
        """装配一个 agent。

        Args:
            llm: 文本模型（任何 BaseLLM 实现）。
            tools: 可用工具集合；模型通过 function calling 点名调用。
            max_steps: 步数预算，防止失控循环烧钱。
            system_prompt: 任务指令（每个场景一条）。
            echo: 进度回调（终端 print 或 Web 的 SSE 队列都行）。
            debug: True 时回显工具的全量参数与返回（不截断）。
        """
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.echo = echo
        self.messages: list[Message] = []
        self._done: dict[str, ToolResult] = {}
        self.debug = debug

    def _say(self, text: str) -> None:
        """通过 echo 回调输出一行进度；未配置 echo 则静默。"""
        if self.echo is not None:
            self.echo(text)

    async def run(self, user_input: str) -> str:
        """执行一个任务：一条用户消息，做到模型给出最终文本为止。

        Args:
            user_input: 用户的原始输入（想法/问题）。
        Returns:
            模型的最终文本回答；步数耗尽时是强制收尾的总结。
        """
        self.messages.append(Message.user(user_input))
        step = 0
        empty_replies = 0
        while step < self.max_steps:
            step += 1
            context = [Message.system(self.system_prompt), *self.messages]
            response = await self.llm.ask(context, tools=self.tools.to_params())
            self.messages.append(response.to_message())

            # 没有工具调用：这就是最终回答
            if not response.tool_calls:
                if response.content:
                    return response.content
                # 空回复：给一次纠正机会，连续两次才放弃
                empty_replies += 1
                if empty_replies >= 2:
                    return ""
                self.messages.append(
                    Message.user(
                        "(Your reply was empty. Continue the task, or give the final answer.)"
                    )
                )
                continue

            empty_replies = 0
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                if tool is None:
                    # 报错里直接列出可用工具——模型下一次通常就改对了
                    result = ToolResult.fail(
                        f"unknown tool '{call.name}'. "
                        f"Available tools: {', '.join(self.tools.names())}"
                    )
                else:
                    if self.debug:
                        self._say(f"· {call.name}({call.function.arguments})")  # 全量参数
                    else:
                        self._say(f"· {call.name}({_short(call.arguments)})")
                    key = f"{call.name}:{call.function.arguments}"
                    cached = self._done.get(key)
                    if cached is not None:
                        # 同参数的成功调用直接回放，不再执行——防"原地转圈"重复烧钱
                        result = cached.model_copy(deep=True)
                        result.output = (result.output or "") + (
                            "\n(note: this exact call already succeeded; do NOT repeat it. "
                            "Continue with the next step, or give the final answer.)"
                        )
                        self._say("  ↳ (重复调用已拦截，回放上次结果)")
                    else:
                        result = await safe_execute(tool, call.arguments)
                        if result.ok:
                            self._done[key] = result.model_copy(deep=True)
                if self.debug:
                    self._say(f"  ↳ {result.for_model()}")  # 全量返回，不截断
                else:
                    self._say(f"  ↳ {_preview(result.for_model())}")
                self.messages.append(Message.tool_result(result.for_model(), call.id, call.name))
                if result.stop:
                    return await self._final_summary(result)
        else:
            # 步数预算耗尽（while 正常结束才进 else）：拿走工具，强制纯文本收尾
            self.messages.append(
                Message.user(
                    "(Step budget exhausted. Summarize the progress and the answer so far "
                    "in plain text. Do not call tools.)"
                )
            )
            context = [Message.system(self.system_prompt), *self.messages]
            response = await self.llm.ask(context, tools=None)
            self.messages.append(response.to_message())
            return response.content or "(step limit reached)"

    async def _final_summary(self, result: ToolResult) -> str:
        """stop 信号后的强制收尾：拿走工具再问一轮，模型物理上无法再调工具。

        Args:
            result: 触发 stop 的工具结果（模型答复为空时回落到它的 output）。
        Returns:
            模型的最终文本总结。
        """
        self.messages.append(
            Message.user(
                "(The task is complete — the last tool signalled the work is done. "
                "Reply now with the final answer: the file path, the panel count, "
                "and one sentence about the story. Do not call any more tools.)"
            )
        )
        context = [Message.system(self.system_prompt), *self.messages]
        response = await self.llm.ask(context, tools=None)
        self.messages.append(response.to_message())
        return response.content or result.output


def _short(args: dict[str, Any]) -> str:
    """把参数字典压成一行短文本（回显用，超 120 字符截断）。

    Args:
        args: 已解析的工具参数。
    Returns:
        可放进一行的 repr 文本。
    """
    text = repr(args)
    return text if len(text) <= 120 else text[:117] + "..."


def _preview(text: str, limit: int = 200) -> str:
    """工具结果的一行预览：取前几行拼起来。看得见它读到了什么，才谈得上核对。

    Args:
        text: 工具结果的完整文本。
        limit: 预览最大长度。
    Returns:
        压缩到一行、截断到 limit 的预览文本。
    """
    joined = " ⏎ ".join(part for part in text.splitlines()[:3] if part.strip())
    if not joined:
        joined = text.strip()
    return joined if len(joined) <= limit else joined[: limit - 1] + "…"
