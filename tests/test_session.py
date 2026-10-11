"""会话系统（状态机/事件流/持久化/断点恢复）的离线测试。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from minimuse.imagegen import BaseImageGen
from minimuse.comic_tools import (
    ComposePage,
    GeneratePanel,
    PlanLayout,
    SaveScript,
    SaveStoryboard,
    size_map_for,
)
from minimuse.session import (
    HAPPY_ORDER,
    SessionState,
    SessionStore,
    SessionTracker,
    missing_panels,
    resume_pipeline,
)
SIZES = size_map_for("flat")


class FlatImageGen(BaseImageGen):
    """本地替身画师：写 64x64 纯色 PNG（resume 测试不碰网络）。"""

    name = "flat"

    def __init__(self) -> None:
        """记录调用次数与最近一次的参考图数量。"""
        self.calls: list[dict] = []

    async def generate(
        self,
        prompt: str,
        *,
        out_path: Path,
        size: str = "1024x1024",
        ref_images: list[Path] | None = None,
    ) -> Path:
        """替身实现：写一张纯色图并返回路径。"""
        from PIL import Image

        self.calls.append({"refs": len(ref_images or [])})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (10, 10, 10)).save(out_path)
        return out_path



def test_store_snapshot_and_events(tmp_path):
    """快照原子写 + 事件流 append-only + 列表排序。"""
    store = SessionStore(tmp_path)
    snap = store.create("一个想法", style="")
    sid = snap["sid"]
    assert snap["state"] == SessionState.CREATED.value

    store.set_state(sid, SessionState.SCRIPT)
    store.update(sid, slug="strip-abcd12", artist="ito-junji")
    got = store.get(sid)
    assert got["state"] == "script" and got["slug"] == "strip-abcd12"

    store.append_event(sid, "log", "hello")
    store.append_event(sid, "file", "/files/x.png")
    evs = store.events(sid)
    assert [e["type"] for e in evs] == ["session_created", "state", "log", "file"]
    assert store.events(sid, limit=2)[0]["type"] == "log"  # limit 保留末尾

    store.create("另一个", style="ghibli")
    assert store.list_sessions()[0]["idea"] == "另一个"  # 按更新时间倒序


def test_tracker_feeds_and_only_forward(tmp_path):
    """回显 → 状态迁移（含 slug/画师提取）；状态只前进不后退。"""
    store = SessionStore(tmp_path)
    sid = store.create("idea")["sid"]
    t = SessionTracker(store, sid)

    t.feed("🎨 画师自选：伊藤润二（ito-junji）—— 气质契合")
    t.feed("  ↳ script saved: D:\\x\\comic_out\\scripts\\strip-abc123.json")
    assert store.get(sid)["artist"] == "ito-junji"
    assert store.get(sid)["slug"] == "strip-abc123"
    assert store.get(sid)["state"] == SessionState.SCRIPT.value

    t.feed("page composed -> ...")  # 越级前进允许（崩溃恢复场景）
    assert store.get(sid)["state"] == SessionState.COMPOSE.value
    t.feed("  ↳ script saved: ...")  # 后退不允许
    assert store.get(sid)["state"] == SessionState.COMPOSE.value
    assert HAPPY_ORDER[-1] == SessionState.DONE


def test_missing_panels_and_no_storyboard(tmp_path):
    """缺格对照；无分镜时恢复直接报"需要完整重跑"。"""
    sb_dir = tmp_path / "storyboards"
    sb_dir.mkdir(parents=True)
    (sb_dir / "strip-x.json").write_text(
        json.dumps({"panels": [{"n": 1}, {"n": 2}, {"n": 3}]}), "utf-8"
    )
    pdir = tmp_path / "panels"
    pdir.mkdir()
    (pdir / "strip-x_p01.png").write_bytes(b"1")
    (pdir / "strip-x_p03.png").write_bytes(b"3")
    assert missing_panels(tmp_path, "strip-x") == [2]

    ok, msg = asyncio.run(
        resume_pipeline(tmp_path, "strip-none", FlatImageGen(), SIZES)
    )
    assert not ok and "run anew" in msg or "no storyboard" in msg


def test_resume_pipeline_rebuilds_missing_panels(tmp_path):
    """端到端恢复：完整工具链产物 → 删一格与成品 → 断点续跑只补缺。"""
    gen = FlatImageGen()
    slug = asyncio.run(_make_full_strip(tmp_path, gen))

    panels = sorted((tmp_path / "panels").glob("*.png"))
    panels[1].unlink()  # 模拟崩溃：第 2 格没画完
    (tmp_path / "pages" / f"{slug}.png").unlink()  # 成品也没合成
    assert missing_panels(tmp_path, slug) == [2]

    gen2 = FlatImageGen()
    echo_log: list[str] = []
    ok, msg = asyncio.run(
        resume_pipeline(tmp_path, slug, gen2, SIZES, echo=echo_log.append)
    )
    assert ok, msg
    assert len(gen2.calls) == 1  # 只补了缺失的 1 格，没重画其他格
    assert gen2.calls[0]["refs"] == 0  # 无参考图时不传
    assert (tmp_path / "pages" / f"{slug}.png").is_file()
    assert any("missing panels [2]" in ln for ln in echo_log)


async def _make_full_strip(workspace: Path, gen: FlatImageGen) -> str:
    """用真实工具链（画师替身）跑出一份完整产物，返回 slug。"""
    r = await SaveScript(workspace=workspace).execute(
        title="恢复测试",
        style="测试风格卡，高反差黑白",
        characters=[{"name": "小测", "look": "短发女孩"}],
        synopsis="一句话",
    )
    assert r.ok, r.error

    r = await SaveStoryboard(workspace=workspace).execute(
        script="恢复测试",
        layout="duo",
        panels=[
            {"n": 1, "scene": "格一", "prompt": "中景：测试画面一", "dialogue": []},
            {
                "n": 2,
                "scene": "格二",
                "prompt": "特写：测试画面二",
                "dialogue": [{"speaker": "小测", "text": "你好", "anchor": "tl"}],
            },
        ],
    )
    assert r.ok, r.error
    m = re.search(r"storyboards[\\/](strip-[a-z0-9-]+)\.json", r.output)
    assert m
    slug = m.group(1)

    r = await PlanLayout(workspace=workspace, panel_sizes=SIZES).execute(
        storyboard=slug
    )
    assert r.ok, r.error
    for n in (1, 2):
        r = await GeneratePanel(
            workspace=workspace, imagegen=gen, panel_sizes=SIZES
        ).execute(storyboard=slug, panel=n)
        assert r.ok, r.error
    r = await ComposePage(workspace=workspace).execute(storyboard=slug)
    assert r.ok, r.error
    return slug