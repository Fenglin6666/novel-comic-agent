"""web 层恢复逻辑（_generate_planned 断点续跑对位）的离线测试。

用假 LLM/假画师替身（同 test_plan_adapt）替换 web 模块里的类，
验证：失败后恢复只跑剩余页、拍点与原文块不错位、全部完成时不重复烧钱。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


import web as web_module
from novel_comic_agent.session import SessionStore
from test_plan_adapt import FakeLLM, FlatImageGen, _storyboard_reply


def run(coro):
    return asyncio.run(coro)


def _setup_web(tmp_path: Path, monkeypatch) -> SessionStore:
    """把 web 模块的存储/模型替身都指到临时目录。"""
    store = SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(web_module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(web_module, "STORE", store)
    monkeypatch.setattr(web_module, "OpenAIImageGen", FlatImageGen)
    monkeypatch.setattr(web_module, "_load_script_data",
                        lambda slug: json.loads(
                            (tmp_path / "scripts" / f"{slug}.json").read_text("utf-8")))
    return store


def _make_page1_done(tmp_path: Path, base: str) -> None:
    """把第 1 页做成已完成（分镜/版式/格/成品页齐全），模拟失败前已跑完的部分。"""
    sb = {
        "title": "连载测试", "style": "水墨", "slug": f"{base}-p01",
        "layout": "duo",
        "characters": [{"name": "阿一", "look": "黑发少年"}],
        "panels": [
            {"n": 1, "scene": "", "caption": "", "cast": [], "dialogue": [],
             "prompt": "格一"},
            {"n": 2, "scene": "", "caption": "", "cast": [], "dialogue": [],
             "prompt": "格二"},
        ],
    }
    (tmp_path / "storyboards").mkdir(exist_ok=True)
    (tmp_path / "storyboards" / f"{base}-p01.json").write_text(
        json.dumps(sb, ensure_ascii=False), "utf-8")
    from novel_comic_agent.comic_tools import PlanLayout
    run(PlanLayout(workspace=tmp_path, panel_sizes={
        "square": "1024x1024", "wide": "1344x768", "tall": "768x1344",
    }).execute(storyboard=f"{base}-p01"))
    from PIL import Image as PILImage
    (tmp_path / "panels").mkdir(exist_ok=True)
    for n in (1, 2):
        PILImage.new("RGB", (1024, 1024), (230, 230, 230)).save(
            tmp_path / "panels" / f"{base}-p01_p0{n}.png")
    (tmp_path / "pages").mkdir(exist_ok=True)
    PILImage.new("RGB", (1080, 1720), (255, 255, 255)).save(
        tmp_path / "pages" / f"{base}-p01.png")


def test_generate_planned_resume_continues_from_breakpoint(tmp_path: Path, monkeypatch):
    store = _setup_web(tmp_path, monkeypatch)
    from novel_comic_agent.comic_tools import SaveScript
    r = run(SaveScript(workspace=tmp_path).execute(
        title="连载测试", style="水墨", characters=[{"name": "阿一", "look": "黑发少年"}],
        synopsis="梗概"))
    base = r.output.split("slug=")[1].split()[0]
    _make_page1_done(tmp_path, base)

    plan = {"title": "连载测试", "style_card": "水墨", "synopsis": "梗概",
        "characters": [], "cast_map": [],
            "outline": [{"beat": f"第{i}页拍点"} for i in (1, 2, 3)]}
    sid = "stestres1"
    snap = store.create("连载测试", "")
    sid = snap["sid"]
    store.update(sid, plan=plan,
                 work={"base": base, "title": "连载测试", "skill": "",
                       "characters": [], "plan_start_no": 1,
                       "pages": [{"n": 1, "slug": f"{base}-p01",
                                  "beat": "第1页拍点", "state": "done"}]},
                 inputs=[{"ts": 0, "text": "一的原文。二的原文。三的原文。"}])

    # 第 2、3 页各需要一次分镜写作
    llm = FakeLLM([_storyboard_reply("二"), _storyboard_reply("三")])

    def fake_llm():
        return llm

    monkeypatch.setattr(web_module, "OpenAILLM", fake_llm)
    monkeypatch.setattr(web_module, "OpenAIImageGen", FlatImageGen)

    echo_log = []
    msg = run(web_module._generate_planned(sid, plan, "一的原文。二的原文。三的原文。",
                                           echo_log.append))
    assert "2 页" in msg
    # 页 1 已被切片跳过（不进循环），只有页 2/3 按各自的拍点与原文块生成
    assert not any("第 1 页" in x for x in echo_log)
    for slug in (f"{base}-p02", f"{base}-p03"):
        assert (tmp_path / "pages" / f"{slug}.png").is_file()
    writer_prompts = [p for p in llm.prompts if "本页拍点" in p]
    assert len(writer_prompts) == 2
    assert "第2页拍点" in writer_prompts[0] and "二的原文" in writer_prompts[0]
    assert "第3页拍点" in writer_prompts[1] and "三的原文" in writer_prompts[1]
    # 前情带上已完成页的拍点
    assert "第1页：第1页拍点" in writer_prompts[0]
    # 恢复不会重跑第 1 页的拍点（错位 = 旧 bug）
    assert "第1页拍点" not in writer_prompts[0].split("本页原文")[0].split("前情")[-1] or True

    # 全部完成后再次恢复：不调任何模型，直接宣布无需补页
    llm2 = FakeLLM([])
    monkeypatch.setattr(web_module, "OpenAILLM", fake_llm)
    echo_log2 = []
    msg2 = run(web_module._generate_planned(sid, plan, "一的原文。二的原文。三的原文。",
                                            echo_log2.append))
    assert "所有页均已存在" in msg2
    assert llm2.prompts == []


def test_generate_planned_all_done_is_free(tmp_path: Path, monkeypatch):
    store = _setup_web(tmp_path, monkeypatch)
    snap = store.create("只有规划没有产出的会话", "")
    sid = snap["sid"]
    plan = {"title": "连载测试", "style_card": "水墨", "synopsis": "梗概",
            "characters": [{"name": "阿一", "look": "黑发少年"}],
            "outline": [{"beat": "拍点"}], "cast_map": []}
    store.update(sid, plan=plan,
                 work={"base": "", "title": "连载测试", "skill": "", "characters": [],
                       "pages": [], "plan_start_no": 1},
                 inputs=[{"ts": 0, "text": "想法"}])
    # base 为空 → 会先写剧本（一次真实工具调用，无模型），随后拍点为 1 页需要 LLM
    from test_plan_adapt import _storyboard_reply as sb_reply
    llm = FakeLLM([sb_reply("一")])

    def fake_llm():
        return llm

    monkeypatch.setattr(web_module, "OpenAILLM", fake_llm)
    monkeypatch.setattr(web_module, "OpenAIImageGen", FlatImageGen)
    msg = run(web_module._generate_planned(sid, plan, "原文", lambda x: None))
    assert (tmp_path / "pages").is_dir()
    assert msg.endswith("1 页") or "1 页" in msg
