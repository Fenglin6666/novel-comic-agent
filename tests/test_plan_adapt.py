"""规划层（plan.py）与逐页流水线（adapt.py）的离线测试。

模型替身只活在测试文件里（约定同 test_comic.py）：
  FakeLLM —— 按脚本返回分镜 JSON，记录收到的 prompt（验前情/角色表注入）；
  FlatImageGen —— 写纯色 PNG（沿用 test_comic 的替身思路）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from minimuse.adapt import generate_pages, write_page
from minimuse.comic_tools import SaveScript, missing_panels, storyboard_slug
from minimuse.imagegen import BaseImageGen
from minimuse.plan import _extract_json, build_plan, chunk_text, cost_estimate, normalize_plan
from minimuse.schema import Message


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- plan 纯函数


def test_chunk_text_splits_at_paragraphs_and_pads():
    text = "\n\n".join(f"第{i}段，内容若干。" for i in range(5))
    chunks = chunk_text(text, 3)
    assert len(chunks) == 3
    assert "".join(chunks).replace("\n\n", "") == text.replace("\n\n", "")
    # 原文比页数少：空串补位（该页靠拍点创作）
    assert chunk_text("只有一段。", 3) == ["只有一段。", "", ""]
    assert chunk_text("单页", 1) == ["单页"]


def test_chunk_text_falls_back_to_sentences():
    text = "第一句。第二句。第三句。第四句。第五句。第六句。"
    chunks = chunk_text(text, 2)
    assert len(chunks) == 2
    assert all(c for c in chunks)
    # 块间补了段落分隔，正文本身不丢
    assert "".join(c.replace("\n\n", "") for c in chunks) == text


def test_cost_estimate_scales_and_sums_image_plus_text():
    one, four = cost_estimate(1, image_model="doubao-seedream-5-0-flash-260915",
                              text_model="deepseek-chat"), cost_estimate(
        4, image_model="doubao-seedream-5-0-flash-260915", text_model="deepseek-chat")
    assert one["panels_est"] == 4
    assert four["panels_est"] == 16
    assert four["cny"] > one["cny"] > 0
    assert one["lo"] <= one["cny"] <= one["hi"]
    # Seedream flash 官方价 ¥0.12/张：图像部分 = 格数 × 单价
    assert one["image_unit"] == 0.12
    assert abs(one["image_cny"] - 0.48) < 1e-9
    # 总成本 = 生图 + 文本 token（文本不为零）
    assert one["text_cny"] > 0
    assert abs(one["cny"] - (one["image_cny"] + one["text_cny"])) < 1e-9
    # 估算里的 token 数量随页数增长
    assert four["tokens_est"]["in"] > one["tokens_est"]["in"]


def test_extract_json_tolerates_fences_and_prose():
    text = '好的，这是规划：\n```json\n{"kind": "short", "pages": 1}\n```\n以上。'
    assert _extract_json(text) == {"kind": "short", "pages": 1}
    assert _extract_json("没有 JSON") is None


def _plan_raw(**over: Any) -> dict[str, Any]:
    raw = {
        "kind": "long",
        "title": "测试作品标题很长很长很长很长很长很长很长很长",
        "synopsis": "梗概",
        "style_card": "风格卡",
        "characters": [
            {"name": "阿一", "look": "少年"},
            {"name": "阿二", "look": "少女"},
            {"name": "", "look": "坏角色"},
            {"name": "阿三", "look": "路人"},
            {"name": "阿四", "look": "多余"},
        ],
        "pages": 99,
        "outline": [{"page": 7, "beat": f"拍点{i}"} for i in range(9)],
        "skill": "ghibli",
        "skill_reason": "气质对",
        "continuation": True,
        "note": "n",
    }
    raw.update(over)
    return raw


def test_normalize_plan_keeps_cast_map():
    p = normalize_plan(_plan_raw(cast_map=[
        {"from": "萧炎", "to": "熊二"},
        {"from": "萧薰儿", "to": "熊大"},
        {"from": "", "to": "坏条目"},
        {"from": "同名", "to": "同名"},       # 自映射无意义，丢弃
        "坏条目",
    ]), idea="x", session_ctx=None, valid_skills=["ghibli"])
    assert p["cast_map"] == [{"from": "萧炎", "to": "熊二"},
                             {"from": "萧薰儿", "to": "熊大"}]
    # 没给 cast_map 时是空列表，字段始终存在
    assert normalize_plan(_plan_raw(), idea="x", session_ctx=None,
                          valid_skills=[])["cast_map"] == []


def test_normalize_plan_clamps_and_renumbers():
    p = normalize_plan(_plan_raw(), idea="x" * 99, session_ctx=None, valid_skills=["ghibli"])
    # 页数被 outline 收敛并封顶；拍点从 1 连续编号；角色截到 4 且跳过坏项
    assert p["pages"] == 9
    assert [o["page"] for o in p["outline"]] == list(range(1, 10))
    assert len(p["characters"]) == 4
    assert [c["name"] for c in p["characters"]] == ["阿一", "阿二", "阿三", "阿四"]
    assert p["continuation"] is False  # 没有会话前情不允许续写
    assert p["skill"] == "ghibli"
    assert len(p["title"]) <= 24


def test_normalize_plan_short_and_fallbacks():
    p = normalize_plan(
        _plan_raw(kind="short", pages=5, outline=[{"beat": "x"}], skill="不存在"),
        idea="一句想法", session_ctx={"title": "旧作"}, valid_skills=["ghibli"],
    )
    assert p["kind"] == "short" and p["pages"] == 1 and p["outline"] == []
    assert p["skill"] == ""  # 非法画师回落为空
    assert p["title"].startswith("旧作") or p["title"]  # 标题有回落
    # 长篇宣称却没给拍点：按输入体量确定性定页数
    p2 = normalize_plan(
        _plan_raw(kind="long", outline=[], pages=0),
        idea="x" * 3600, session_ctx=None, valid_skills=[],
    )
    assert p2["pages"] == 3
    assert p2["kind"] == "long"


class FakeResp:
    """LLM 返回替身：只带 content（write_page 只用 content 与 to_message）。"""

    def __init__(self, content: str) -> None:
        self.content = content

    def to_message(self) -> Message:
        return Message.assistant(content=self.content)


class FakeLLM:
    """按脚本依次出话；记录每次 prompt，供断言前情/角色表是否注入。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def ask(self, messages, tools=None, on_delta=None):  # noqa: ANN001
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return FakeResp(self.replies.pop(0) if self.replies else "{}")


def test_build_plan_parses_and_corrects():
    reply = json.dumps(_plan_raw(pages=3, outline=[
        {"page": 1, "beat": "开场"}, {"page": 2, "beat": "推进"}, {"page": 3, "beat": "收尾"},
    ]), ensure_ascii=False)
    llm = FakeLLM([f"```json\n{reply}\n```"])
    p = run(build_plan("一个三页的小故事", llm, valid_skills=["ghibli"]))
    assert p["kind"] == "long" and p["pages"] == 3
    assert [o["beat"] for o in p["outline"]] == ["开场", "推进", "收尾"]


def test_build_plan_raises_after_two_failures():
    llm = FakeLLM(["不是 JSON", "还不是 JSON"])
    try:
        run(build_plan("想法", llm, valid_skills=[]))
        raise AssertionError("should raise")
    except ValueError as exc:
        assert "JSON" in str(exc)


# ---------------------------------------------------------------- 逐页流水线


class FlatImageGen(BaseImageGen):
    """写纯色 PNG 的本地替身（同 test_comic）。"""

    name = "flat"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt, *, out_path, size="1024x1024", ref_images=None):  # noqa: ANN001
        self.calls += 1
        w, h = (int(x) for x in size.split("x"))
        PILImage.new("RGB", (w, h), (240, 240, 240)).save(out_path)
        return out_path


def _storyboard_reply(beats: str) -> str:
    return json.dumps({
        "layout": "duo",
        "panels": [
            {"n": 1, "scene": beats + "上", "caption": "", "cast": ["阿一"],
             "dialogue": [{"speaker": "阿一", "text": "台词一", "anchor": "tl"}],
             "prompt": "黑白线稿，少年站在天台，逆光，风掀起衣角"},
            {"n": 2, "scene": beats + "下", "caption": "旁白一句", "cast": [],
             "dialogue": [], "prompt": "空镜，城市黄昏，电线杆剪影，暖橙色天空"},
        ],
    }, ensure_ascii=False)


def _setup_workspace(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    r = run(SaveScript(workspace=tmp_path).execute(
        title="连载测试", style="水墨淡彩，留白多",
        characters=[{"name": "阿一", "look": "黑发少年"}], synopsis="测试梗概"))
    assert r.ok
    slug = r.output.split("slug=")[1].split()[0]
    script = json.loads((tmp_path / "scripts" / f"{slug}.json").read_text("utf-8"))
    return slug, script


def test_generate_pages_end_to_end_and_resume(tmp_path: Path):
    base, script = _setup_workspace(tmp_path)
    beats = [{"beat": "第一页拍点"}, {"beat": "第二页拍点"}]
    chunks = chunk_text("第一页的原文。\n\n第二页的原文。", 2)
    llm = FakeLLM([_storyboard_reply("一"), _storyboard_reply("二")])
    gen = FlatImageGen()
    echo_log, on_pages = [], []
    ok, msg, done = run(generate_pages(
        tmp_path, base=base, beats=beats, chunks=chunks, start_no=1, script=script,
        digest_prefix="", llm=llm, imagegen=gen, panel_sizes={
            "square": "1024x1024", "wide": "1344x768", "tall": "768x1344"},
        echo=echo_log.append, on_page=lambda n, s, b, sk: on_pages.append((n, s, b, sk)),
    ))
    assert ok and done == 2 and "2 页" in msg
    assert on_pages == [(1, base + "-p01", "第一页拍点", False),
                        (2, base + "-p02", "第二页拍点", False)]
    for slug in (f"{base}-p01", f"{base}-p02"):
        assert (tmp_path / "storyboards" / f"{slug}.json").is_file()
        assert (tmp_path / "layouts" / f"{slug}.json").is_file()
        assert not missing_panels(tmp_path, slug)
        assert (tmp_path / "pages" / f"{slug}.png").is_file()
    # 前情与拍点确实进了编剧的 prompt；页码/角色表/原文块也在
    second_prompt = llm.prompts[1]
    assert "第1页：第一页拍点" in second_prompt
    assert "第二页拍点" in second_prompt and "第二页的原文" in second_prompt
    assert "阿一" in second_prompt
    # 恢复 = 原样重跑：全部跳过，不再花一分钱图片费
    calls_before = gen.calls
    ok2, msg2, done2 = run(generate_pages(
        tmp_path, base=base, beats=beats, chunks=chunks, start_no=1, script=script,
        digest_prefix="", llm=FakeLLM([]), imagegen=gen, panel_sizes={
            "square": "1024x1024", "wide": "1344x768", "tall": "768x1344"},
        echo=echo_log.append, on_page=lambda n, s, b, sk: None,
    ))
    assert ok2 and done2 == 2
    assert gen.calls == calls_before
    assert sum("已完成，跳过" in line for line in echo_log) == 2


def test_generate_pages_continues_numbering(tmp_path: Path):
    base, script = _setup_workspace(tmp_path)
    llm = FakeLLM([_storyboard_reply("续")])
    ok, msg, done = run(generate_pages(
        tmp_path, base=base, beats=[{"beat": "续写页"}], chunks=["续写原文"], start_no=2,
        script=script, digest_prefix="第1页：开场", llm=llm, imagegen=FlatImageGen(),
        panel_sizes={"square": "1024x1024", "wide": "1344x768", "tall": "768x1344"},
    ))
    assert ok and done == 1
    slug = f"{base}-p02"
    assert (tmp_path / "pages" / f"{slug}.png").is_file()
    assert "第1页：开场" in llm.prompts[0]  # 续写把前情带进编剧
    # 同一剧本，两个分镜 slug（跨页共用角色表/风格卡的根据）
    assert (tmp_path / "storyboards" / f"{base}-p02.json").is_file()


def test_write_page_retries_with_tool_error(tmp_path: Path):
    base, script = _setup_workspace(tmp_path)
    bad = json.dumps({"layout": "duo", "panels": [{"n": 1, "prompt": "只有一格"}]},
                     ensure_ascii=False)
    good = _storyboard_reply("修正")
    llm = FakeLLM([bad, good])  # 第一次格数不符 → 工具报错喂回 → 第二次成功
    r = run(write_page(
        llm, tmp_path, base=base, page_no=1, title="连载测试", style_card="水墨",
        characters=script["characters"], synopsis="梗概", beat="拍点", digest="", chunk="原文"))
    assert r.ok
    assert "只有一格" in llm.prompts[1]  # 错误信息确实喂回了模型
    assert (tmp_path / "storyboards" / f"{base}-p01.json").is_file()
    assert storyboard_slug("连载测试") == base  # slug 沿用剧本


def test_write_page_injects_cast_map(tmp_path: Path):
    base, script = _setup_workspace(tmp_path)
    llm = FakeLLM([_storyboard_reply("换角")])
    r = run(write_page(
        llm, tmp_path, base=base, page_no=1, title="连载测试", style_card="水墨",
        characters=script["characters"], synopsis="梗概", beat="拍点", digest="",
        chunk="原文里的萧炎握紧了拳头。",
        cast_map=[{"from": "萧炎", "to": "熊二"}]))
    assert r.ok
    prompt = llm.prompts[0]
    assert "角色替换" in prompt
    assert "「萧炎」就是「熊二」" in prompt
    assert "萧炎握紧了拳头" in prompt  # 原文块原样保留（替换由模型执行）
    # 分镜落盘：数据里没有强约束 cast_map，但写盘的剧本角色表仍是参考图角色
    data = json.loads((tmp_path / "storyboards" / f"{base}-p01.json").read_text("utf-8"))
    assert [c["name"] for c in data["characters"]] == ["阿一"]
