"""漫画工具链：剧本 → 分镜+版式 → layout.json → 逐格生图 → 合成页。

四要素对四层：
  画风定义 —— save_script 的全局风格卡 + 每格提示词自动拼上（一致性第一道保险）
  人物一致 —— save_script 的角色卡（外貌描述），generate_panel 自动注入出场角色
  剧情连贯 —— save_storyboard 的分镜 + 台词（speaker/anchor），compose_page 渲染气泡
  图片质量 —— generate_panel 按 bbox 长宽比选生成档位，compose_page cover-fit 裁贴

关键决策：bbox 不让模型算（坐标必飘）。模型只从版式模板里选一个
（动作页用不规则斜切 action4，平静叙事用规整 classic），plan_layout
用确定性代码把模板展开成精确 bbox —— 和"叙事规格写分镜、像素档位
在工具层"是同一个哲学。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from novel_comic_agent.imagegen import BaseImageGen, load_font, wrap_text
from novel_comic_agent.refs import ref_files_for, ref_names_for
from novel_comic_agent.schema import ToolResult
from novel_comic_agent.tools import BaseTool

NO_TEXT_SUFFIX = "，画面中不要出现任何文字、字母或对话气泡"
MAX_PANELS = 8
MAX_CHARACTERS = 4
ANCHORS = {"tl", "tr", "bl", "br", "c"}

# 页面几何：整页画布 + 页眉（标题/风格）+ 内容区，所有 bbox 都在内容区内展开
PAGE_W, PAGE_H = 1080, 1720
MARGIN, HEADER_H, GUTTER = 48, 128, 22


def content_area() -> tuple[int, int, int, int]:
    """计算页眉之下的内容区矩形，所有分格 bbox 都在它内部展开。

    Returns:
        (x, y, w, h)：内容区左上角坐标与宽高。
    """
    x, y = MARGIN, MARGIN + HEADER_H
    return x, y, PAGE_W - 2 * MARGIN, PAGE_H - y - MARGIN


# 生成档位（两端点都实测可用：CogView 与 gpt-image-2 中转均忠实传参）
PANEL_SIZES = {
    "square": "1024x1024",
    "wide": "1344x768",
    "tall": "768x1344",
}

# 字节 Seedream：边长下限 1024，768 档会被拒；换 2K 档映射。
# 语义键不变（wide/square/tall），换画师只换表——分镜协议不动。
SEEDREAM_SIZES = {
    "square": "2048x2048",
    "wide": "2048x1152",
    "tall": "1152x2048",
}
_SIZE_ASPECTS = {"square": 1.0, "wide": 16 / 9, "tall": 9 / 16}


def missing_panels(workspace: Path, slug: str) -> list[int]:
    """对照分镜与 panels/ 目录，找出还没画（或画坏了被删）的格号。

    Args:
        workspace: 工作区（comic_out）。
        slug: 分镜 slug。
    Returns:
        缺失的格号列表（升序）。
    Raises:
        FileNotFoundError: 分镜文件不存在。
    """
    sb = json.loads((workspace / "storyboards" / f"{slug}.json").read_text("utf-8"))
    return [
        p["n"]
        for p in sb["panels"]
        if not (workspace / "panels" / f"{slug}_p{p['n']:02d}.png").exists()
    ]


def storyboard_slug(title: str) -> str:
    """标题 → 稳定文件名：ASCII 部分保留，中文标题退化为哈希。同名覆盖=幂等。"""
    ascii_part = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:6]
    return f"{ascii_part or 'strip'}-{digest}"


def size_map_for(model: str) -> dict[str, str]:
    """按图片模型名选像素档位映射表（换画师只换表，分镜协议不动）。

    Args:
        model: 图片模型名（OpenAIImageGen.model）。
    Returns:
        {语义键: 像素档位} 映射表。
    """
    if "seedream" in (model or "").lower():
        return SEEDREAM_SIZES
    return PANEL_SIZES


# ---------------------------------------------------------------- 版式模板


def _tpl_solo(a):
    """solo：单格整页。"""
    x, y, w, h = a
    return [(x, y, w, h)]


def _tpl_duo(a):
    """duo：上下两等分。"""
    x, y, w, h = a
    ph = (h - GUTTER) // 2
    return [(x, y, w, ph), (x, y + ph + GUTTER, w, h - ph - GUTTER)]


def _tpl_trio(a):
    """trio：上横幅 + 下两格（开场建场 → 双线推进）。"""
    x, y, w, h = a
    r1 = int(h * 0.42)
    r2 = h - r1 - GUTTER
    cw = (w - GUTTER) // 2
    r2y = y + r1 + GUTTER
    return [(x, y, w, r1), (x, r2y, cw, r2), (x + cw + GUTTER, r2y, w - cw - GUTTER, r2)]


def _tpl_action4(a):
    """不规则斜切：上行 62/38，下行 38/62 —— 对角线动势，动作/张力页用。"""
    x, y, w, h = a
    ph = (h - GUTTER) // 2
    w1 = int(w * 0.62)
    w2 = w - w1 - GUTTER
    by = y + ph + GUTTER
    return [
        (x, y, w1, ph),
        (x + w1 + GUTTER, y, w2, ph),
        (x, by, w2, ph),
        (x + w2 + GUTTER, by, w1, h - ph - GUTTER),
    ]


def _tpl_big_open(a):
    """big_open：上大开页（55% 高）+ 下三连格（建场 → 三拍节奏）。"""
    x, y, w, h = a
    r1 = int(h * 0.55)
    r2 = h - r1 - GUTTER
    cw = (w - 2 * GUTTER) // 3
    ry = y + r1 + GUTTER
    return [(x, y, w, r1)] + [(x + i * (cw + GUTTER), ry, cw, r2) for i in range(3)]


def _tpl_five(a):
    """five：上横幅 + 2×2 四格。"""
    x, y, w, h = a
    r1 = int(h * 0.34)
    rem = h - r1 - GUTTER
    ph = (rem - GUTTER) // 2
    cw = (w - GUTTER) // 2
    r2y, r3y = y + r1 + GUTTER, y + r1 + GUTTER + ph + GUTTER
    return [
        (x, y, w, r1),
        (x, r2y, cw, ph),
        (x + cw + GUTTER, r2y, w - cw - GUTTER, ph),
        (x, r3y, cw, ph),
        (x + cw + GUTTER, r3y, w - cw - GUTTER, ph),
    ]


def _grid(a, cols: int, rows: int):
    """规整网格（classic 2×3、grid8 2×4 的底座），行优先排列。

    Args:
        a: 内容区 (x, y, w, h)。
        cols: 列数。
        rows: 行数。
    Returns:
        共 cols*rows 个 bbox 的列表。
    """
    x, y, w, h = a
    cw = (w - (cols - 1) * GUTTER) // cols
    ch = (h - (rows - 1) * GUTTER) // rows
    return [
        (x + i * (cw + GUTTER), y + j * (ch + GUTTER), cw, ch)
        for j in range(rows)
        for i in range(cols)
    ]


TEMPLATES: dict[str, Any] = {
    "solo": _tpl_solo,
    "duo": _tpl_duo,
    "trio": _tpl_trio,
    "action4": _tpl_action4,
    "big_open": _tpl_big_open,
    "five": _tpl_five,
    "classic": lambda a: _grid(a, 2, 3),
    "grid8": lambda a: _grid(a, 2, 4),
}
TEMPLATE_COUNT = {
    "solo": 1,
    "duo": 2,
    "trio": 3,
    "action4": 4,
    "big_open": 4,
    "five": 5,
    "classic": 6,
    "grid8": 8,
}
COUNT_TEMPLATE = {v: k for k, v in TEMPLATE_COUNT.items() if k not in ("action4",)}


def _size_key_for(aspect: float) -> str:
    """bbox 长宽比 → 最接近的语义档位键（wide/square/tall）。"""
    return min(_SIZE_ASPECTS, key=lambda k: abs(math.log(aspect / _SIZE_ASPECTS[k])))


# ---------------------------------------------------------------- 存取


def _resolve_json(root: Path, subdir: str, ref: str) -> tuple[str, dict[str, Any]] | None:
    """按 slug 或标题在 root/subdir 下找 JSON 并读出。

    Args:
        root: 工作区根目录（comic_out）。
        subdir: 子目录名（scripts/storyboards/layouts）。
        ref: 模型给的引用——可能是 slug 也可能是原标题。
    Returns:
        (slug, 数据字典)；找不到/读不动返回 None，由调用方给可纠错报错。
    """
    ref = (ref or "").strip()
    for name in (ref, storyboard_slug(ref)):
        path = root / subdir / f"{name}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(data, dict):
                return name, data
    return None


def _load_json(root: Path, subdir: str, slug: str) -> dict[str, Any] | None:
    """按精确 slug 读一个 JSON；失败返回 None。"""
    path = root / subdir / f"{slug}.json"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- save_script


class SaveScript(BaseTool):
    name: str = "save_script"
    description: str = (
        "保存剧本：标题、一条全局风格卡 style（中文，所有格共享）、角色卡 characters"
        "（1-4 个，每个 name + look：中文外貌描述——年龄/体型/发型/服装/标志特征，"
        "这是每一格保持人物一致的依据）、synopsis（2-3 句梗概，用户语言）。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "style": {"type": "string", "description": "全局风格卡（中文）"},
            "characters": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "用户语言的名字"},
                        "look": {"type": "string", "description": "外貌卡（中文）"},
                    },
                    "required": ["name", "look"],
                },
            },
            "synopsis": {"type": "string"},
        },
        "required": ["title", "style", "characters", "synopsis"],
    }
    workspace: Path

    async def execute(
        self,
        title: str = "",
        style: str = "",
        characters: Any = None,
        synopsis: str = "",
        **_: Any,
    ) -> ToolResult:
        """保存剧本到 scripts/<slug>.json；全部校验通过才落盘。

        Args 见工具 schema。
        Returns:
            成功输出 slug、角色名单与下一步指引；失败为可纠错错误。
        """
        title = (title or "").strip()
        style = (style or "").strip()
        if not title:
            return ToolResult.fail("title is empty")
        if not style:
            return ToolResult.fail(
                "style is empty — one global style card keeps all panels consistent"
            )
        if not isinstance(characters, list) or not characters:
            return ToolResult.fail("characters must be a non-empty list (1-4)")
        if len(characters) > MAX_CHARACTERS:
            return ToolResult.fail(f"too many characters ({len(characters)}); max {MAX_CHARACTERS}")
        norm = []
        for i, c in enumerate(characters, start=1):
            if not isinstance(c, dict):
                return ToolResult.fail(f"character #{i} must be an object")
            name = str(c.get("name") or "").strip()
            look = str(c.get("look") or "").strip()
            if not name:
                return ToolResult.fail(f"character #{i}: name is empty")
            if not look:
                return ToolResult.fail(f"character '{name}': look is empty (中文外貌卡)")
            norm.append({"name": name, "look": look})
        slug = storyboard_slug(title)
        data = {
            "title": title,
            "style": style,
            "characters": norm,
            "synopsis": (synopsis or "").strip(),
        }
        path = self.workspace / "scripts" / f"{slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        return ToolResult(
            output=(
                f"script saved: {path}\nslug={slug} characters={[c['name'] for c in norm]}\n"
                f"next: save_storyboard(script='{slug}', layout=..., panels=[...])"
            )
        )


# ---------------------------------------------------------------- save_storyboard


def _norm_dialogue(raw: Any, chars: dict[str, str], panel_n: int) -> list[dict[str, str]] | str:
    """校验并规范化一格的台词列表。

    Args:
        raw: 模型给的 dialogue 字段（list 或 None）。
        chars: {角色名: 外貌卡}，speaker 必须在其中。
        panel_n: 所属格号（报错定位用）。
    Returns:
        规范化后的台词列表；校验失败返回错误字符串。
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        return f"panel #{panel_n}: dialogue must be a list"
    out = []
    for i, d in enumerate(raw, start=1):
        if not isinstance(d, dict):
            return f"panel #{panel_n} dialogue #{i} must be an object"
        speaker = str(d.get("speaker") or "").strip()
        text = str(d.get("text") or "").strip()
        anchor = str(d.get("anchor") or "tl").strip().lower()
        if speaker not in chars:
            return (
                f"panel #{panel_n} dialogue #{i}: unknown speaker '{speaker}'; "
                f"characters in this script: {sorted(chars)}"
            )
        if not text:
            return f"panel #{panel_n} dialogue #{i}: text is empty"
        if anchor not in ANCHORS:
            return (
                f"panel #{panel_n} dialogue #{i}: anchor must be one of "
                f"{sorted(ANCHORS)}, got '{anchor}'"
            )
        out.append({"speaker": speaker, "text": text, "anchor": anchor})
    return out


def _norm_panels(
    raw: Any, chars: dict[str, str], want_count: int
) -> list[dict[str, Any]] | str:
    """校验并规范化分镜格列表。

    Args:
        raw: 模型给的 panels 字段。
        chars: {角色名: 外貌卡}，用于校验 speaker/cast。
        want_count: 版式模板的槽数，格数必须严格相等。
    Returns:
        按 n 排序的规范化格列表；校验失败返回错误字符串（喂回模型自纠）。
    """
    if not isinstance(raw, list) or not raw:
        return "panels must be a non-empty list"
    if len(raw) > MAX_PANELS:
        return f"too many panels ({len(raw)}); max {MAX_PANELS}"
    if len(raw) != want_count:
        return (
            f"this layout has {want_count} slots but {len(raw)} panels were given; "
            f"add or remove panels to match"
        )
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            return f"panel #{i} must be an object"
        n = item.get("n", i)
        if not isinstance(n, int) or isinstance(n, bool):
            return f"panel #{i}: n must be an integer"
        if n in seen:
            return f"duplicate panel number {n}"
        seen.add(n)
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            return f"panel #{n}: prompt is empty (describe what is VISIBLE in the image)"
        dialogue = _norm_dialogue(item.get("dialogue"), chars, n)
        if isinstance(dialogue, str):
            return dialogue
        cast = item.get("cast")
        if cast is None:
            cast = sorted({d["speaker"] for d in dialogue})
        elif not isinstance(cast, list):
            return f"panel #{n}: cast must be a list of character names"
        cast = [str(c).strip() for c in cast]
        unknown = [c for c in cast if c not in chars]
        if unknown:
            return (
                f"panel #{n}: unknown cast {unknown}; characters in this script: {sorted(chars)}"
            )
        out.append(
            {
                "n": n,
                "scene": str(item.get("scene") or "").strip(),
                "caption": str(item.get("caption") or "").strip(),
                "cast": cast,
                "dialogue": dialogue,
                "prompt": prompt,
            }
        )
    out.sort(key=lambda p: p["n"])
    return out


class SaveStoryboard(BaseTool):
    name: str = "save_storyboard"
    description: str = (
        "保存一页漫画的分镜。layout 从模板里选：solo(1)/duo(2)/trio(3)/"
        "action4(4,不规则斜切,动作张力页)/big_open(4,大开页+三连格)/five(5)/"
        "classic(6,规整2x3)/grid8(8)；给 auto 则按格数选默认模板。"
        "每格：n、scene、caption(旁白,可空)、dialogue[{speaker,text,anchor tl/tr/bl/br/c}]、"
        "cast(出场角色,缺省=说话者)、prompt(中文,只写画面)。格数必须等于模板槽数。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "save_script 返回的 slug（标题也行）"},
            "page": {"type": "integer",
                     "description": "可选：页码（多页连载时用，slug 变为 <script>-pNN）"},
            "layout": {"type": "string", "enum": [*TEMPLATES, "auto"]},
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                        "scene": {"type": "string"},
                        "caption": {"type": "string"},
                        "cast": {"type": "array", "items": {"type": "string"}},
                        "dialogue": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string"},
                                    "text": {"type": "string"},
                                    "anchor": {
                                        "type": "string",
                                        "enum": ["tl", "tr", "bl", "br", "c"],
                                    },
                                },
                                "required": ["speaker", "text"],
                            },
                        },
                        "prompt": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            },
        },
        "required": ["script", "layout", "panels"],
    }
    workspace: Path

    async def execute(
        self, script: str = "", layout: str = "auto", panels: Any = None,
        page: int = 0, **_: Any,
    ) -> ToolResult:
        """保存分镜到 storyboards/<slug>.json（slug 沿用剧本的）。

        Args:
            script: 剧本 slug 或标题（必须先 save_script）。
            layout: 版式模板名，或 auto（按格数选默认模板）。
            panels: 分镜格列表（见 schema）。
            page: 页码（多页连载用）；>0 时 slug 变为 <script>-pNN，
                全部页共用同一份剧本（风格卡/角色表 = 跨页一致性的根）。
        Returns:
            成功输出 slug、版式、格号与下一步指引；失败为可纠错错误。
        """
        found = _resolve_json(self.workspace, "scripts", script)
        if found is None:
            return ToolResult.fail(f"no script '{script}' — call save_script first")
        script_slug, script_data = found
        chars = {c["name"]: c["look"] for c in script_data.get("characters", [])}
        layout = (layout or "auto").strip().lower()
        if layout == "auto":
            if not isinstance(panels, list) or len(panels) not in COUNT_TEMPLATE:
                supported = sorted(TEMPLATE_COUNT.values())
                count = len(panels) if isinstance(panels, list) else 0
                return ToolResult.fail(
                    f"cannot auto-pick a layout for {count} "
                    f"panels; supported panel counts: {supported}"
                )
            layout = COUNT_TEMPLATE[len(panels)]
        if layout not in TEMPLATES:
            return ToolResult.fail(
                f"unknown layout '{layout}'; available: {sorted(TEMPLATES)} or 'auto'"
            )
        want = TEMPLATE_COUNT[layout]
        norm = _norm_panels(panels, chars, want)
        if isinstance(norm, str):
            return ToolResult.fail(norm)
        slug = script_slug if page <= 0 else f"{script_slug}-p{page:02d}"
        # 剧本字段（风格卡/角色卡）冗余存进分镜：生图时不必再读剧本文件
        data = {**script_data, "slug": slug, "layout": layout, "panels": norm}
        path = self.workspace / "storyboards" / f"{slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        ns = [p["n"] for p in norm]
        return ToolResult(
            output=(
                f"storyboard saved: {path}\nslug={slug} layout={layout} panels={ns}\n"
                f"next: plan_layout(storyboard='{slug}')"
            )
        )


# ---------------------------------------------------------------- plan_layout


class PlanLayout(BaseTool):
    name: str = "plan_layout"
    description: str = (
        "把分镜的版式模板展开成精确的 layout.json：每格 bbox、长宽比、"
        "生成档位。确定性计算，不调模型。之后 generate_panel 会按 bbox 比例出图。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"storyboard": {"type": "string"}},
        "required": ["storyboard"],
    }
    workspace: Path
    panel_sizes: dict[str, str] = PANEL_SIZES  # 按画师选表：size_map_for(imagegen.model)

    async def execute(self, storyboard: str = "", **_: Any) -> ToolResult:
        """把版式模板展开成精确 bbox，写 layouts/<slug>.json。

        Args:
            storyboard: 分镜 slug 或标题。
        Returns:
            成功时逐行列出每格 bbox/长宽比/生成档位，供模型据此措辞提示词。
        """
        found = _resolve_json(self.workspace, "storyboards", storyboard)
        if found is None:
            return ToolResult.fail(f"no storyboard '{storyboard}' — call save_storyboard first")
        slug, data = found
        name = data.get("layout")
        if name not in TEMPLATES:
            return ToolResult.fail(f"storyboard '{slug}' has unknown layout '{name}'")
        area = content_area()
        bboxes = TEMPLATES[name](area)
        entries = []
        lines = []
        for p, bbox in zip(data["panels"], bboxes, strict=True):
            x, y, w, h = bbox
            aspect = round(w / h, 2)
            key = _size_key_for(w / h)
            gen = self.panel_sizes[key]
            entries.append(
                {
                    "n": p["n"],
                    "bbox": list(bbox),
                    "aspect": aspect,
                    "size_key": key,
                    "gen_size": gen,
                }
            )
            lines.append(
                f"panel {p['n']}: bbox=[{x}, {y}, {w}, {h}] aspect={aspect} {key} gen={gen}"
            )
        layout_data = {
            "slug": slug,
            "layout": name,
            "page": {"width": PAGE_W, "height": PAGE_H},
            "content": list(area),
            "panels": entries,
        }
        path = self.workspace / "layouts" / f"{slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(layout_data, ensure_ascii=False, indent=2), "utf-8")
        return ToolResult(
            output=(
                f"layout planned: {path} (page {PAGE_W}x{PAGE_H}, layout={name})\n"
                + "\n".join(lines)
                + f"\nnext: generate_panel(storyboard='{slug}', panel=n) for every panel "
                f"（宽格写成宽幅构图，长格写成特写）"
                f"then compose_page(storyboard='{slug}')"
            )
        )


# ---------------------------------------------------------------- generate_panel


class GeneratePanel(BaseTool):
    name: str = "generate_panel"
    description: str = (
        "为分镜中的某一格生成图片（先 plan_layout）。风格卡、出场角色的外貌卡、"
        "禁字后缀会自动拼进提示词；生成档位按该格 bbox 的长宽比自动选择。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "storyboard": {"type": "string"},
            "panel": {"type": "integer"},
        },
        "required": ["storyboard", "panel"],
    }
    workspace: Path
    imagegen: BaseImageGen
    panel_sizes: dict[str, str] = PANEL_SIZES  # 按画师选表：size_map_for(imagegen.model)

    async def execute(self, storyboard: str = "", panel: int = 0, **_: Any) -> ToolResult:
        """为某一格生成图片，写 panels/<slug>_pNN.png。

        提示词 = 风格卡 + 出场角色外貌卡 + 该格画面 + 禁字后缀；
        生成档位 = panel_sizes[该格 size_key]（旧 layout 无 size_key 时回落 gen_size）。

        Args:
            storyboard: 分镜 slug 或标题。
            panel: 格号 n。
        Returns:
            成功输出文件路径与实际尺寸；失败为可纠错错误（含改写提示词建议）。
        """
        found = _resolve_json(self.workspace, "storyboards", storyboard)
        if found is None:
            return ToolResult.fail(f"no storyboard '{storyboard}' — call save_storyboard first")
        slug, data = found
        layout = _load_json(self.workspace, "layouts", slug)
        if layout is None:
            return ToolResult.fail(f"no layout for '{slug}' — call plan_layout first")
        if {p["n"] for p in layout["panels"]} != {p["n"] for p in data["panels"]}:
            return ToolResult.fail(
                f"storyboard and layout disagree on panels — storyboard changed after "
                f"plan_layout; call plan_layout(storyboard='{slug}') again"
            )
        target = next((p for p in data["panels"] if p["n"] == panel), None)
        entry = next((p for p in layout["panels"] if p["n"] == panel), None)
        if target is None or entry is None:
            ns = [p["n"] for p in data["panels"]]
            return ToolResult.fail(f"panel {panel!r} not found in '{slug}'; available panels: {ns}")

        chars = {c["name"]: c["look"] for c in data.get("characters", [])}
        ref_names = ref_names_for(self.workspace, target.get("cast", []))
        refs = ref_files_for(self.workspace, target.get("cast", []))
        # 有参考图的角色：外貌完全由参考图锁定，文字外貌卡反而会把模型带偏——不拼。
        looks = "; ".join(
            f"{name}: {chars[name]}" for name in target.get("cast", [])
            if name in chars and name not in ref_names
        )
        parts = [data["style"]]
        if looks:
            parts.append(looks)
        if ref_names:
            named = "、".join(ref_names)
            parts.append(
                f"{named} 必须与输入参考图中的角色保持完全一致：同一人物，"
                "脸型、发型、五官、服装、配色一律照搬参考图，不要按文字重新设计"
            )
        parts.append(target["prompt"])
        prompt = "。".join(parts) + NO_TEXT_SUFFIX
        out = self.workspace / "panels" / f"{slug}_p{panel:02d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)  # 目录责任在工具，不在画师实现
        try:
            key = entry.get("size_key") or _size_key_for(float(entry.get("aspect", 1.0)))
            size = self.panel_sizes.get(key) or entry.get("gen_size") or PANEL_SIZES["square"]
            await self.imagegen.generate(prompt, out_path=out, size=size, ref_images=refs or None)
            with Image.open(out) as img:
                dims = f"{img.width}x{img.height}"
        except Exception as exc:  # noqa: BLE001 — 画师失败必须变成可纠错的结果
            return ToolResult.fail(
                f"image generation failed for panel {panel}: {type(exc).__name__}: {exc}. "
                "Rewrite the prompt (more concrete, simpler) and try this panel once more."
            )
        nexts = [p["n"] for p in data["panels"] if p["n"] > panel]
        hint = (
            f"next: generate_panel(storyboard='{slug}', panel={min(nexts)})"
            if nexts
            else f"all panels done; next: compose_page(storyboard='{slug}')"
        )
        return ToolResult(
            output=(
                f"panel {panel} generated -> {out} "
                f"({dims}, bbox aspect {entry['aspect']}"
                + (f", {len(refs)} 参考图" if refs else "")
                + f")\n{hint}"
            )
        )


# ---------------------------------------------------------------- compose_page


def _cover_fit(img: Image.Image, bw: int, bh: int) -> Image.Image:
    """cover-fit：等比缩放填满目标框后居中裁剪（同 CSS object-fit: cover）。

    Args:
        img: 原图。
        bw, bh: 目标框宽高（该格 bbox 的宽高）。
    Returns:
        恰好 bw×bh 的裁剪结果。
    """
    scale = max(bw / img.width, bh / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    return resized.crop((left, top, left + bw, top + bh))


def _fit_font(text: str, size: int, max_w: float) -> int:
    """字号自适应：超宽就逐步缩小，直到文本放进 max_w（下限 20）。

    Args:
        text: 要测量的文本。
        size: 起始字号。
        max_w: 可用最大像素宽度。
    Returns:
        最终字号。
    """
    while size > 20 and load_font(size).getlength(text) > max_w:
        size -= 4
    return size


class ComposePage(BaseTool):
    name: str = "compose_page"
    description: str = (
        "把分镜合成一页漫画 PNG：cover-fit 贴图、画分格线、旁白框（左上）、"
        "气泡（按 anchor 摆放，台词由代码渲染）。需要先 plan_layout 并生成所有格子。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"storyboard": {"type": "string"}},
        "required": ["storyboard"],
    }
    workspace: Path

    async def execute(self, storyboard: str = "", **_: Any) -> ToolResult:
        """把所有格子合成一页漫画，写 pages/<slug>.png。

        流程：cover-fit 贴图 → 分格线 → 旁白框（左上米色）→
        气泡（按 anchor 摆放、同锚点纵向堆叠、尾巴指向格子中心）。

        Args:
            storyboard: 分镜 slug 或标题（需先 plan_layout 且图已生成齐）。
        Returns:
            成功输出成品路径与页尺寸/格数/版式；缺图或 layout 过期则报错。
        """
        found = _resolve_json(self.workspace, "storyboards", storyboard)
        if found is None:
            return ToolResult.fail(f"no storyboard '{storyboard}' — call save_storyboard first")
        slug, data = found
        layout = _load_json(self.workspace, "layouts", slug)
        if layout is None:
            return ToolResult.fail(f"no layout for '{slug}' — call plan_layout first")
        entries = {p["n"]: p for p in layout["panels"]}
        if set(entries) != {p["n"] for p in data["panels"]}:
            return ToolResult.fail(
                "storyboard and layout disagree on panels; "
                f"call plan_layout(storyboard='{slug}') again"
            )
        missing = [
            p["n"]
            for p in data["panels"]
            if not (self.workspace / "panels" / f"{slug}_p{p['n']:02d}.png").is_file()
        ]
        if missing:
            return ToolResult.fail(
                f"missing panel images for panels {missing} in '{slug}'; generate_panel them first"
            )

        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(page)
        title = data.get("title", "")
        t_size = _fit_font(title, 46, PAGE_W - 2 * MARGIN)
        d.text((MARGIN, 52), title, font=load_font(t_size), fill=(15, 15, 15))
        style_note = data.get("style", "")
        if len(style_note) > 110:  # 页眉放不下完整风格卡，截断展示
            style_note = style_note[:110] + "…"
        d.text((MARGIN, 52 + t_size + 10), style_note, font=load_font(22), fill=(150, 150, 150))
        d.line(
            [(MARGIN, MARGIN + HEADER_H - 14), (PAGE_W - MARGIN, MARGIN + HEADER_H - 14)],
            fill=(200, 200, 200),
            width=2,
        )

        cap_f = load_font(24)
        for p in data["panels"]:
            x, y, w, h = entries[p["n"]]["bbox"]
            path = self.workspace / "panels" / f"{slug}_p{p['n']:02d}.png"
            with Image.open(path) as im:
                page.paste(_cover_fit(im.convert("RGB"), w, h), (x, y))
            # 分格线
            d.rectangle([x, y, x + w, y + h], outline=(10, 10, 10), width=3)
            # 旁白框：左上，米色底
            if p.get("caption"):
                max_cw = int(w * 0.62)
                lines = wrap_text(p["caption"], cap_f, max_cw - 24)
                cw = min(max(int(cap_f.getlength(ln)) for ln in lines) + 24, max_cw)
                ch = len(lines) * 32 + 16
                d.rectangle(
                    [x + 10, y + 10, x + 10 + cw, y + 10 + ch],
                    fill=(252, 246, 228),
                    outline=(90, 80, 50),
                    width=2,
                )
                ty = y + 18
                for ln in lines:
                    d.text((x + 22, ty), ln, font=cap_f, fill=(60, 50, 30))
                    ty += 32
            # 气泡：按 anchor 摆放，同锚点纵向堆叠；尾巴指向格子中心
            stack: dict[str, int] = {}
            pcx, pcy = x + w / 2, y + h / 2
            for dia in p.get("dialogue", []):
                anchor = dia.get("anchor", "tl")
                self._draw_bubble(d, (x, y, w, h), pcx, pcy, dia["text"], anchor, stack)

        out = self.workspace / "pages" / f"{slug}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        page.save(out)
        return ToolResult(
            output=(
                f"page composed -> {out} ({PAGE_W}x{PAGE_H}, {len(data['panels'])} panels, "
                f"layout={layout['layout']})\n"
                "TASK COMPLETE: the work is done. Give your final reply now — do not "
                "call save_script or any other tool again."
            ),
            stop=True,  # 任务完成信号：循环将强制一轮无工具收尾，确定性终止
        )

    @staticmethod
    def _draw_bubble(d, panel, pcx, pcy, text, anchor, stack):
        """在格子内画一个对话气泡（圆角矩形 + 尾巴 + 台词）。

        Args:
            d: 页面画笔。
            panel: 格子 bbox (x, y, w, h)。
            pcx, pcy: 格子中心坐标，尾巴指向这里。
            text: 台词文本。
            anchor: tl/tr/bl/br/c 五个摆放位。
            stack: {anchor: 已占高度}；同锚点多条台词纵向堆叠，就地更新。
        """
        x, y, w, h = panel
        font_size = 28
        for size in (28, 24, 20, 18):
            font = load_font(size)
            max_w = int(w * 0.68)
            lines = wrap_text(text, font, max_w - 36)
            bw = min(max(int(font.getlength(ln)) for ln in lines) + 36, max_w)
            bh = len(lines) * (size + 10) + 22
            font_size = size
            if bh <= h * 0.55:
                break
        font = load_font(font_size)
        pad = 14
        sy = stack.get(anchor, 0)
        if anchor == "tl":
            bx, by = x + pad, y + pad + sy
        elif anchor == "tr":
            bx, by = x + w - bw - pad, y + pad + sy
        elif anchor == "bl":
            bx, by = x + pad, y + h - bh - pad - sy
        elif anchor == "br":
            bx, by = x + w - bw - pad, y + h - bh - pad - sy
        else:
            bx, by = x + (w - bw) / 2, y + (h - bh) / 2
        bx = max(x + 8, min(bx, x + w - bw - 8))
        by = max(y + 8, min(by, y + h - bh - 8))
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=26, fill="white",
                            outline=(10, 10, 10), width=3)
        # 尾巴：从气泡边缘指向格子中心
        bcx, bcy = bx + bw / 2, by + bh / 2
        dx, dy = pcx - bcx, pcy - bcy
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        ex, ey = bcx + ux * bw / 2, bcy + uy * bh / 2
        tipx, tipy = ex + ux * 32, ey + uy * 32
        px, py = -uy, ux
        b1 = (ex + px * 15, ey + py * 15)
        b2 = (ex - px * 15, ey - py * 15)
        d.polygon([b1, (tipx, tipy), b2], fill="white")
        d.line([b1, (tipx, tipy), b2], fill=(10, 10, 10), width=3, joint="curve")
        ty = by + 11
        for ln in lines:
            d.text((bx + 18, ty), ln, font=font, fill=(10, 10, 10))
            ty += font_size + 10
        stack[anchor] = stack.get(anchor, 0) + bh + 10
