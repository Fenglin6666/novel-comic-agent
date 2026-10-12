"""会话系统：状态机 + 事务日志 + 持久化 + 崩溃恢复。

一个会话一个事务日志（log.jsonl），会话产生的所有数据都先进日志再发给
消费者——消费者（前端消息窗、agent 循环）拿到的只是事件的副本：

  sessions/<sid>/log.jsonl     事务日志（append-only）：四种 message
                               （system/user/assistant/tool）、state、plan、
                               artifact、done/error……全部在这一个文件里
  sessions/<sid>/session.json  派生快照（最新状态，供列表/徽章快速查询）
  sessions/<sid>/events.jsonl  旧版事件流（只读兼容；新写入一律进 log.jsonl）

事件统一带序号：{"seq", "ts", "type", "data"}。先保存到本地，再由
消费者消费——消费者拿到的是落盘事件的副本，谁都不改写日志。

状态机（一次生成的一生）：
  created → style → script → storyboard → layout → panels → compose → done
  任何状态 ─→ failed / cancelled

恢复哲学：agent 循环每次从日志加载消息副本（不保留原件），只负责产出新
数据；确定性部分（补格/合成）按产物断点续跑。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from novel_comic_agent.comic_tools import ComposePage, GeneratePanel, missing_panels  # noqa: F401 — 再导出（旧导入路径兼容）
from novel_comic_agent.imagegen import BaseImageGen
from novel_comic_agent.schema import Message, ToolCall, ToolResult


class SessionLog:
    """会话事务日志：一个会话一个 log.jsonl，append-only。

    会话产生的所有数据先进日志再发给消费者——消费者拿到的只是事件的
    副本。四种 message（system/user/assistant/tool）以 msg 事件存储
    （与 OpenAI 线协议同构，另带 ok 标记工具成败），与 state/plan/
    artifact/done/error 等事件同住一个文件。
    """

    def __init__(self, root: Path, sid: str) -> None:
        self.root = Path(root)
        self.sid = sid
        self._seq = 0  # 惰性初始化：首次 append 时从文件行数推算

    @property
    def path(self) -> Path:
        return Path(self.root) / self.sid / "log.jsonl"

    def _next_seq(self) -> int:
        if self._seq:
            self._seq += 1
            return self._seq
        n = 0
        if self.path.is_file():
            with self.path.open("rb") as f:
                n = sum(1 for _ in f)
        self._seq = n + 1
        return self._seq

    def append(self, type_: str, data: Any) -> dict[str, Any]:
        """先保存到本地：追加一条事件并落盘，返回落盘事件（发给消费者的副本）。"""
        ev = {"seq": self._next_seq(), "ts": round(time.time(), 3),
              "type": type_, "data": data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
        return ev

    def events(self, after: int = 0) -> list[dict[str, Any]]:
        """读取全部事件（after 之后的部分）——返回副本，改动不影响日志。"""
        try:
            lines = self.path.read_text("utf-8").splitlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for ln in lines:
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if ev.get("seq", 0) > after:
                out.append(ev)
        return out

    @staticmethod
    def msg_payload(msg: Message, ok: bool | None = None) -> dict[str, Any]:
        """Message → msg 事件的负载（与 to_wire 同构，另带工具成败标记）。"""
        data: dict[str, Any] = msg.to_wire()
        if msg.role.value == "tool" and ok is not None:
            data["ok"] = ok
        return data

    def append_message(self, msg: Message, ok: bool | None = None) -> dict[str, Any]:
        """把四种 message 之一追加进事务日志，返回落盘事件。"""
        return self.append("msg", self.msg_payload(msg, ok=ok))

    def messages(self) -> list[Message]:
        """从日志还原 LLM 对话消息——返回副本，供 agent 循环加载使用。"""
        out: list[Message] = []
        for ev in self.events():
            if ev.get("type") != "msg":
                continue
            d = ev.get("data") or {}
            calls = None
            for c in d.get("tool_calls") or []:
                fn = c.get("function") or {}
                calls = calls or []
                calls.append(ToolCall(id=c.get("id") or "", type="function",
                                      function={"name": fn.get("name") or "",
                                               "arguments": fn.get("arguments") or "{}"}))
            out.append(Message(role=d.get("role") or "user",
                               content=d.get("content"),
                               tool_calls=calls or None,
                               tool_call_id=d.get("tool_call_id"),
                               name=d.get("name")))
        return out


class SessionState(str, Enum):
    """一次生成会话的全部状态。"""

    CREATED = "created"
    STYLE = "style"            # 正在选画师
    SCRIPT = "script"          # 剧本已写
    STORYBOARD = "storyboard"  # 分镜已存
    LAYOUT = "layout"          # 版式已排
    PANELS = "panels"          # 逐格作画（可能部分完成）
    COMPOSE = "compose"        # 成品已合成
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 正常路径的先后顺序（用于判断"状态只前进不后退"）
HAPPY_ORDER = [
    SessionState.CREATED,
    SessionState.STYLE,
    SessionState.SCRIPT,
    SessionState.STORYBOARD,
    SessionState.LAYOUT,
    SessionState.PANELS,
    SessionState.COMPOSE,
    SessionState.DONE,
]


class SessionStore:
    """会话档案的读写门面：目录布局 + 原子快照 + 事件追加。

    目录结构：
        <root>/<sid>/session.json   状态快照
        <root>/<sid>/events.jsonl   事件流（一行一个 JSON）
    """

    _last_ts: float = 0.0  # 类级单调钟：同一毫秒内连续写时保证 updated 严格递增

    @staticmethod
    def _now() -> float:
        SessionStore._last_ts = max(SessionStore._last_ts + 0.001, time.time())
        return SessionStore._last_ts

    def __init__(self, root: Path) -> None:
        """初始化存储根目录（sessions/）。

        Args:
            root: 会话根目录；不存在会在首次写入时创建。
        """
        self.root = root

    def _dir(self, sid: str) -> Path:
        """会话目录；sid 只允许安全字符，防路径逃逸。"""
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", sid or ""):
            raise ValueError(f"bad session id {sid!r}")
        return self.root / sid

    def create(self, idea: str, style: str = "") -> dict[str, Any]:
        """开一个新会话，写初始快照。

        Args:
            idea: 用户的想法原文。
            style: 显式指定的画师 id；"" 表示自动选择。
        Returns:
            完整快照 dict（含 sid）。
        """
        sid = f"s{int(time.time() * 1000):x}-{uuid.uuid4().hex[:6]}"
        snap: dict[str, Any] = {
            "sid": sid,
            "idea": idea,
            "style": style,
            "artist": "",
            "slug": "",
            "state": SessionState.CREATED.value,
            "error": "",
            "created": SessionStore._now(),
            "updated": SessionStore._now(),
        }
        d = self._dir(sid)
        d.mkdir(parents=True, exist_ok=True)
        self._write_snapshot(d, snap)
        self.append_event(sid, "session_created", {"idea": idea, "style": style})
        return snap

    @staticmethod
    def _write_snapshot(d: Path, snap: dict[str, Any]) -> None:
        """原子写快照：先写临时文件再替换，崩溃不留半截 JSON。"""
        tmp = d / "session.json.tmp"
        tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(d / "session.json")

    def update(self, sid: str, **fields: Any) -> dict[str, Any]:
        """合并更新快照字段（state/artist/slug/error…），并刷新 updated 时间。

        Args:
            sid: 会话 id。
            **fields: 要覆盖的字段。
        Returns:
            更新后的完整快照。
        Raises:
            KeyError: 会话不存在。
        """
        snap = self.get(sid)
        if snap is None:
            raise KeyError(f"no session {sid!r}")
        snap.update(fields)
        snap["updated"] = self._now()
        self._write_snapshot(self._dir(sid), snap)
        return snap

    def set_state(self, sid: str, state: SessionState, error: str = "") -> dict[str, Any]:
        """状态迁移：更新快照 + 记一条 state 事件（返回落盘事件给消费者转发）。"""
        self.update(sid, state=state.value, error=error)
        return self.append_event(sid, "state", {"state": state.value, "error": error})

    def get(self, sid: str) -> dict[str, Any] | None:
        """读快照；不存在返回 None。坏 JSON 不抛（崩溃现场），返回 None。"""
        f = self._dir(sid) / "session.json"
        try:
            return json.loads(f.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def delete(self, sid: str) -> bool:
        """删除整个会话目录（快照 + 事件流）。调用方负责确认没有产物。

        Args:
            sid: 会话 id。
        Returns:
            是否真的删了（不存在返回 False）。
        """
        import shutil

        try:
            d = self._dir(sid)
        except ValueError:
            return False
        if not d.is_dir():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出全部会话快照，按更新时间倒序（最新的在最前）。"""
        out: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            try:
                snap = json.loads((d / "session.json").read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append(snap)
        return sorted(out, key=lambda s: s.get("updated", 0), reverse=True)

    def append_event(self, sid: str, type_: str, data: Any) -> dict[str, Any]:
        """把一条事件写入会话事务日志（先落盘，返回的副本交给消费者）。

        旧会话只有 events.jsonl：首次追加时先把旧事件整体迁移进 log.jsonl，
        保证一个会话的完整历史始终在一个事务日志文件里。
        """
        log = SessionLog(self.root, sid)
        if not log.path.exists():
            legacy = self._dir(sid) / "events.jsonl"
            if legacy.is_file():
                log.path.parent.mkdir(parents=True, exist_ok=True)
                log.path.write_bytes(legacy.read_bytes())
        return log.append(type_, data)

    def events(self, sid: str, limit: int | None = None) -> list[dict[str, Any]]:
        """回放会话事务日志；新会话读 log.jsonl，老会话回退读 events.jsonl。"""
        d = self._dir(sid)
        f = d / "log.jsonl"
        if not f.exists():
            f = d / "events.jsonl"
        try:
            lines = f.read_text("utf-8").splitlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out[-limit:] if limit else out


class SessionTracker:
    """从 agent 的 echo 回显里解析状态迁移与 slug/画师，写入会话档案。

    为什么盯回显而不是改工具协议：零侵入——Agent/ToolResult 不用动，
    CLI 与 Web 共用同一套回显文案，追踪器两边都能挂。
    """

    _SLUG_RE = re.compile(r"scripts[\\/](strip-[a-z0-9-]+)\.json")
    _ARTIST_RE = re.compile(r"画师自选：(.+?)（(.+?)）")

    def __init__(self, store: SessionStore, sid: str) -> None:
        """绑定一个会话。

        Args:
            store: 会话存储。
            sid: 会话 id。
        """
        self.store = store
        self.sid = sid

    def feed(self, line: str) -> None:
        """喂一行回显；识别到已知里程碑就迁移状态（幂等：只前进不后退）。"""
        if "画师自选" in line:
            self._advance(SessionState.STYLE)
            m = self._ARTIST_RE.search(line)
            if m:
                self.store.update(self.sid, artist=m.group(2))
        elif "script saved:" in line:
            self._advance(SessionState.SCRIPT)
            m = self._SLUG_RE.search(line)
            if m:
                self.store.update(self.sid, slug=m.group(1))
        elif "storyboard saved:" in line:
            self._advance(SessionState.STORYBOARD)
        elif "layout planned:" in line:
            self._advance(SessionState.LAYOUT)
        elif "generated ->" in line:
            self._advance(SessionState.PANELS)
        elif "page composed ->" in line:
            self._advance(SessionState.COMPOSE)

    def _advance(self, state: SessionState) -> None:
        """只在状态"向前"时迁移（HAPPY_ORDER 里的序号变大才写）。"""
        snap = self.store.get(self.sid)
        if snap is None:
            return
        try:
            cur = SessionState(snap["state"])
        except ValueError:
            cur = SessionState.CREATED
        if state not in HAPPY_ORDER or cur not in HAPPY_ORDER:
            return
        if HAPPY_ORDER.index(state) > HAPPY_ORDER.index(cur):
            self.store.set_state(self.sid, state)


async def resume_pipeline(
    workspace: Path,
    slug: str,
    imagegen: BaseImageGen,
    panel_sizes: dict[str, str],
    echo: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """从产物断点确定性续跑：补画缺格 → 重合成。不经过 LLM。

    Args:
        workspace: 工作区（comic_out）。
        slug: 分镜 slug（storyboards/<slug>.json 必须存在）。
        imagegen: 画师实现。
        panel_sizes: 画师的像素档位表。
        echo: 进度回调（复用 Web 的 SSE 队列 / CLI 的 print）。
    Returns:
        (ok, message)：ok=True 表示补完全部格子并重新合成了成品页。
    """

    def say(text: str) -> None:
        if echo is not None:
            echo(text)

    if not (workspace / "storyboards" / f"{slug}.json").is_file():
        return False, f"no storyboard for '{slug}' — need a full agent run, not resume"
    gen = GeneratePanel(workspace=workspace, imagegen=imagegen, panel_sizes=panel_sizes)
    composer = ComposePage(workspace=workspace)
    try:
        todo = missing_panels(workspace, slug)
        say(f"[resume] {slug}: missing panels {todo or 'none'}")
        for n in todo:
            say(f"[resume] generating panel {n} …")
            r: ToolResult = await gen.execute(storyboard=slug, panel=n)
            if not r.ok:
                return False, f"panel {n} failed: {r.error}"
            say(f"[resume] {r.output.splitlines()[0]}")
        say("[resume] composing page …")
        r = await composer.execute(storyboard=slug)
        if not r.ok:
            return False, f"compose failed: {r.error}"
        say(f"[resume] {r.output.splitlines()[0]}")
        return True, r.output.splitlines()[0]
    except Exception as exc:  # noqa: BLE001 — 恢复过程也要给出可读失败
        return False, f"{type(exc).__name__}: {exc}"
