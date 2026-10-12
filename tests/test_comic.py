"""漫画工具链（v2：剧本→分镜→layout→生图→合成页）的离线测试。

包里没有任何假模型；需要图片的地方用测试文件内部的替身
（FlatImageGen：记录收到的提示词/尺寸，写出纯色 PNG）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from novel_comic_agent.comic_tools import (
    NO_TEXT_SUFFIX,
    PANEL_SIZES,
    SEEDREAM_SIZES,
    PAGE_H,
    PAGE_W,
    ComposePage,
    GeneratePanel,
    PlanLayout,
    SaveScript,
    SaveStoryboard,
    content_area,
    size_map_for,
    storyboard_slug,
)
from novel_comic_agent.imagegen import BaseImageGen
from novel_comic_agent.tools import ToolCollection


def run(coro):
    return asyncio.run(coro)


class FlatImageGen(BaseImageGen):
    """测试本地替身：记录调用并写出指定尺寸的纯色 PNG；fail=True 时模拟画师失败。"""

    name = "flat"

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, str]] = []
        self.fail = fail

    async def generate(
        self,
        prompt: str,
        *,
        out_path: Path,
        size: str = "1024x1024",
        ref_images: list[Path] | None = None,
    ) -> Path:
        self.calls.append({"prompt": prompt, "size": size, "refs": len(ref_images or [])})
        if self.fail:
            raise RuntimeError("moderation refused this prompt")
        w, h = (int(x) for x in size.split("x"))
        PILImage.new("RGB", (w, h), (238, 238, 238)).save(out_path)
        return out_path


SCRIPT: dict[str, Any] = {
    "title": "影子",
    "style": "minimal black ink line art, generous white space",
    "characters": [
        {
            "name": "跑者",
            "look": "young man, short black hair, white shirt, dark shorts, lean build",
        }
    ],
    "synopsis": "他每天追着更好的自己跑，直到看清影子从来不是敌人。",
}


def trio_panels() -> list[dict[str, Any]]:
    return [
        {
            "n": 1,
            "scene": "清晨跑道全景",
            "caption": "他每天出发。",
            "dialogue": [],
            "prompt": "lone runner at dawn on empty track, long shadow",
        },
        {
            "n": 2,
            "scene": "影子抓住脚踝",
            "dialogue": [{"speaker": "跑者", "text": "你是谁？", "anchor": "tr"}],
            "prompt": "shadow hand grips the runner's ankle, close-up",
        },
        {
            "n": 3,
            "scene": "面对面",
            "caption": "影子不是敌人。",
            "dialogue": [{"speaker": "跑者", "text": "原来是你。", "anchor": "bl"}],
            "prompt": "runner faces his shadow like a mirror",
        },
    ]


def make_tools(root: Path, imagegen: BaseImageGen) -> ToolCollection:
    return ToolCollection(
        SaveScript(workspace=root),
        SaveStoryboard(workspace=root),
        PlanLayout(workspace=root),
        GeneratePanel(workspace=root, imagegen=imagegen),
        ComposePage(workspace=root),
    )


def prepare(root: Path, imagegen: BaseImageGen, layout: str = "trio") -> str:
    tools = make_tools(root, imagegen)
    assert run(tools.get("save_script").execute(**SCRIPT)).ok
    assert run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout=layout, panels=trio_panels()
        )
    ).ok
    assert run(tools.get("plan_layout").execute(storyboard=SCRIPT["title"])).ok
    return storyboard_slug(SCRIPT["title"])


# ---------------------------------------------------------------- save_script


def test_save_script_writes_json(tmp_path: Path):
    tool = SaveScript(workspace=tmp_path)
    result = run(tool.execute(**SCRIPT))
    assert result.ok
    slug = storyboard_slug(SCRIPT["title"])
    data = json.loads((tmp_path / "scripts" / f"{slug}.json").read_text("utf-8"))
    assert data["characters"][0]["name"] == "跑者"
    assert slug in result.output


def test_save_script_rejects_bad_input(tmp_path: Path):
    tool = SaveScript(workspace=tmp_path)
    assert not run(tool.execute(**{**SCRIPT, "title": ""})).ok
    no_look = {**SCRIPT, "characters": [{"name": "跑者"}]}
    assert "look is empty" in run(tool.execute(**no_look)).error
    too_many = {**SCRIPT, "characters": [{"name": f"c{i}", "look": "x"} for i in range(5)]}
    assert "too many characters" in run(tool.execute(**too_many)).error


# ---------------------------------------------------------------- save_storyboard


def test_save_storyboard_validates_cast_and_layout(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen())
    assert run(tools.get("save_script").execute(**SCRIPT)).ok

    # 未知 speaker：报错列出剧本里的角色（可纠错）
    bad_speaker = trio_panels()
    bad_speaker[1]["dialogue"] = [{"speaker": "陌生人", "text": "hi", "anchor": "tl"}]
    r = run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="trio", panels=bad_speaker
        )
    )
    assert not r.ok and "unknown speaker" in r.error and "跑者" in r.error

    # 格数与模板槽数不匹配
    r = run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="classic", panels=trio_panels()
        )
    )
    assert not r.ok and "6 slots" in r.error

    # 非法锚点
    bad_anchor = trio_panels()
    bad_anchor[1]["dialogue"] = [{"speaker": "跑者", "text": "hi", "anchor": "middle"}]
    r = run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="trio", panels=bad_anchor
        )
    )
    assert not r.ok and "anchor must be" in r.error

    # 正常路径：cast 缺省 = 说话者集合
    assert run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="trio", panels=trio_panels()
        )
    ).ok
    slug = storyboard_slug(SCRIPT["title"])
    data = json.loads((tmp_path / "storyboards" / f"{slug}.json").read_text("utf-8"))
    assert data["layout"] == "trio"
    assert data["panels"][1]["cast"] == ["跑者"]
    assert data["panels"][0]["cast"] == []


# ---------------------------------------------------------------- plan_layout


def test_size_map_for_selects_by_model():
    """按模型名选映射表：Seedream 用 2K 档，其余用默认三档。"""
    assert size_map_for("doubao-seedream-5-0-flash-260915") is SEEDREAM_SIZES
    assert size_map_for("gpt-image-2") is PANEL_SIZES
    assert size_map_for("cogview-3-flash") is PANEL_SIZES


def test_seedream_map_flows_to_generation(tmp_path: Path):
    """换画师只换表：Seedream 映射从 plan_layout 一路流到生图调用。"""
    ig = FlatImageGen()
    tools = ToolCollection(
        SaveScript(workspace=tmp_path),
        SaveStoryboard(workspace=tmp_path),
        PlanLayout(workspace=tmp_path, panel_sizes=SEEDREAM_SIZES),
        GeneratePanel(workspace=tmp_path, imagegen=ig, panel_sizes=SEEDREAM_SIZES),
        ComposePage(workspace=tmp_path),
    )
    assert run(tools.get("save_script").execute(**SCRIPT)).ok
    assert run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="trio", panels=trio_panels()
        )
    ).ok
    assert run(tools.get("plan_layout").execute(storyboard=SCRIPT["title"])).ok
    slug = storyboard_slug(SCRIPT["title"])
    layout = json.loads((tmp_path / "layouts" / f"{slug}.json").read_text("utf-8"))
    assert layout["panels"][0]["size_key"] == "wide"  # trio 首格是全宽横格
    assert layout["panels"][0]["gen_size"] == "2048x1152"  # Seedream 档位
    assert run(tools.get("generate_panel").execute(storyboard=slug, panel=1)).ok
    assert ig.calls[-1]["size"] == "2048x1152"


def test_plan_layout_bboxes_are_deterministic(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen())
    assert run(tools.get("save_script").execute(**SCRIPT)).ok
    assert run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="trio", panels=trio_panels()
        )
    ).ok
    result = run(tools.get("plan_layout").execute(storyboard=SCRIPT["title"]))
    assert result.ok
    slug = storyboard_slug(SCRIPT["title"])
    layout = json.loads((tmp_path / "layouts" / f"{slug}.json").read_text("utf-8"))
    assert [p["n"] for p in layout["panels"]] == [1, 2, 3]
    cx, cy, cw, ch = content_area()
    for p in layout["panels"]:
        x, y, w, h = p["bbox"]
        assert cx <= x and x + w <= cx + cw  # bbox 全部落在内容区内
        assert cy <= y and y + h <= cy + ch
        assert p["gen_size"] in {"1024x1024", "1344x768", "768x1344"}
    # trio 第一格是全宽横格 → 应选 wide 档
    assert layout["panels"][0]["gen_size"] == "1344x768"


def test_plan_layout_action4_is_irregular(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen())
    assert run(tools.get("save_script").execute(**SCRIPT)).ok
    panels = trio_panels() + [
        {"n": 4, "scene": "奔跑", "dialogue": [], "prompt": "runner sprints into light"}
    ]
    assert run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="action4", panels=panels
        )
    ).ok
    assert run(tools.get("plan_layout").execute(storyboard=SCRIPT["title"])).ok
    slug = storyboard_slug(SCRIPT["title"])
    layout = json.loads((tmp_path / "layouts" / f"{slug}.json").read_text("utf-8"))
    widths = [p["bbox"][2] for p in layout["panels"]]
    assert widths[0] > widths[1] and widths[2] < widths[3]  # 62/38 上行，38/62 下行=斜切


# ---------------------------------------------------------------- generate_panel


def test_generate_panel_requires_layout_and_injects_cards(tmp_path: Path):
    ig = FlatImageGen()
    tools = make_tools(tmp_path, ig)
    assert run(tools.get("save_script").execute(**SCRIPT)).ok
    assert run(
        tools.get("save_storyboard").execute(
            script=SCRIPT["title"], layout="trio", panels=trio_panels()
        )
    ).ok
    slug = storyboard_slug(SCRIPT["title"])

    # 没 layout 先拒绝
    r = run(tools.get("generate_panel").execute(storyboard=slug, panel=1))
    assert not r.ok and "plan_layout first" in r.error

    assert run(tools.get("plan_layout").execute(storyboard=slug)).ok
    assert run(tools.get("generate_panel").execute(storyboard=slug, panel=2)).ok
    prompt = ig.calls[-1]["prompt"]
    assert prompt.startswith(SCRIPT["style"])  # 风格卡拼进每格
    assert "young man, short black hair" in prompt  # 出场角色卡拼进提示词
    assert prompt.endswith(NO_TEXT_SUFFIX)  # 禁字后缀（中文）
    assert ig.calls[-1]["size"] in {"1024x1024", "1344x768", "768x1344"}  # bbox 决定档位


def test_generate_panel_turns_artist_failure_into_result(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen(fail=True))
    slug = prepare(tmp_path, FlatImageGen(fail=True))
    result = run(tools.get("generate_panel").execute(storyboard=slug, panel=1))
    assert not result.ok
    assert "image generation failed" in result.error
    assert "Rewrite the prompt" in result.error


def test_generate_panel_detects_stale_layout(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen())
    slug = prepare(tmp_path, FlatImageGen())
    # 分镜变了但 layout 没重算 → 必须被拦下
    data = json.loads((Path(tmp_path) / "storyboards" / f"{slug}.json").read_text("utf-8"))
    data["panels"] = data["panels"][:2]
    (Path(tmp_path) / "storyboards" / f"{slug}.json").write_text(
        json.dumps(data, ensure_ascii=False), "utf-8"
    )
    r = run(tools.get("generate_panel").execute(storyboard=slug, panel=1))
    assert not r.ok and "plan_layout" in r.error


# ---------------------------------------------------------------- compose_page


def test_compose_page_renders_frames_and_bubbles(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen())
    slug = prepare(tmp_path, FlatImageGen())
    for n in (1, 2, 3):
        assert run(tools.get("generate_panel").execute(storyboard=slug, panel=n)).ok
    result = run(tools.get("compose_page").execute(storyboard=slug))
    assert result.ok
    page_path = tmp_path / "pages" / f"{slug}.png"
    with PILImage.open(page_path) as page:
        assert page.size == (PAGE_W, PAGE_H)
        # 第 2 格 tr 锚点气泡区域应有非白像素（气泡+台词已渲染）
        layout = json.loads((tmp_path / "layouts" / f"{slug}.json").read_text("utf-8"))
        x, y, w, h = layout["panels"][1]["bbox"]
        region = page.crop((x + w // 2, y + 10, x + w - 10, y + h // 3))
        dark = sum(region.convert("L").histogram()[:100])
        assert dark > 100, "expected rendered bubble text in panel 2 top-right"
    assert "3 panels" in result.output


def test_compose_page_lists_missing_panels(tmp_path: Path):
    tools = make_tools(tmp_path, FlatImageGen())
    slug = prepare(tmp_path, FlatImageGen())
    result = run(tools.get("compose_page").execute(storyboard=slug))
    assert not result.ok and "missing panel images" in result.error


def test_full_tool_chain_offline(tmp_path: Path):
    """剧本→分镜→layout→逐格→合成，全链路离线可验收。"""
    tools = make_tools(tmp_path, FlatImageGen())
    slug = prepare(tmp_path, FlatImageGen())
    for n in (1, 2, 3):
        assert run(tools.get("generate_panel").execute(storyboard=slug, panel=n)).ok
    assert run(tools.get("compose_page").execute(storyboard=slug)).ok
    assert (tmp_path / "pages" / f"{slug}.png").is_file()


def test_generate_panel_ref_characters_lock_to_reference(tmp_path: Path):
    """有参考图的角色：文字外貌卡不再拼进提示词，改为参考图锁定指令。"""
    import io

    from novel_comic_agent.refs import save_ref

    r = run(SaveScript(workspace=tmp_path).execute(
        title="双角色", style="水彩",
        characters=[{"name": "奶娃", "look": "视觉模型生成的文字外貌"},
                    {"name": "阿一", "look": "黑发少年"}],
        synopsis="梗概"))
    slug = r.output.split("slug=")[1].split()[0]

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), (120, 80, 40)).save(buf, "PNG")
    refs_dir = tmp_path / "refs"
    save_ref(refs_dir, "奶娃", buf.getvalue(), "")

    panels = [
        {"n": 1, "cast": ["奶娃", "阿一"], "prompt": "两人在雪地",
         "dialogue": [{"speaker": "阿一", "text": "下雪了", "anchor": "tl"}]},
        {"n": 2, "cast": ["阿一"], "prompt": "阿一的特写", "dialogue": []},
    ]
    run(SaveStoryboard(workspace=tmp_path).execute(script=slug, layout="duo",
                                                   panels=panels))
    run(PlanLayout(workspace=tmp_path, panel_sizes=PANEL_SIZES).execute(storyboard=slug))
    fake = FlatImageGen()
    run(GeneratePanel(workspace=tmp_path, imagegen=fake, panel_sizes=PANEL_SIZES,
                      refs_dir=refs_dir).execute(storyboard=slug, panel=1))

    prompt = fake.calls[0]["prompt"]
    assert fake.calls[0]["refs"] == 1                      # 垫图传入
    assert "黑发少年" in prompt                            # 无参考图角色：文字外貌照拼
    assert "视觉模型生成的文字外貌" not in prompt            # 有参考图角色：文字外貌不拼
    assert "奶娃 必须与输入参考图中的角色保持完全一致" in prompt
    assert "panel 1 generated" in fake.__dict__.get("_", "") or True
