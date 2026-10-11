"""画师风格 Skill：索引层（frontmatter）+ 正文注入 + 模型自动选择。

skills/<id>.md 分两层（对应真 Agent Skill 的渐进式披露）：
  索引层 —— YAML 头（title/description），常驻"选择器"上下文，正文不进，
            几十个画师挂着也只花几百 token；
  正文层 —— 完整画师卡（视觉基因/叙事气质/适配题材/避开），选中才全文注入。

选择权默认交给模型（choose_skill：读索引 → 按剧本气质挑一位），
用户显式指定画师时以用户为准（apply_skill 直接注入）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from novel_comic_agent.schema import Message

if TYPE_CHECKING:
    from novel_comic_agent.llm import BaseLLM

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "skills"
_SAFE_ID = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff-]*$")
MAX_SKILL_BYTES = 20_000
_DESC_MAX = 80  # 索引层一句话简介的截断长度，防止上传文件把索引撑爆


def skills_dir() -> Path:
    """画师 skill 目录（项目根下 skills/）。"""
    return _DEFAULT_DIR


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """分离文件最前面的 YAML 头与正文。

    Args:
        text: md 全文；头必须是首行 '---' 开始的键: 值单行块。
    Returns:
        (meta, body)：meta 为 {键: 值}；没有合法头时 meta 为空、body 原样返回。
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), 0)
    if end == 0:  # 只有开头的 --- 没有闭合，视为普通正文
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, "\n".join(lines[end + 1 :]).strip()


def _fallback_description(body: str) -> str:
    """无 frontmatter 文件（用户上传）的简介回落：取第一段非标题、非列表符号文字。"""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.lstrip(">- ").strip()[:_DESC_MAX]
    return ""


def _title_of(path: Path, meta: dict[str, str], body: str) -> str:
    """显示名优先级：frontmatter title > 正文首个 '# ' 标题 > 文件名主干。"""
    if meta.get("title"):
        return meta["title"]
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _read_skill(path: Path) -> dict[str, str]:
    """读一个 skill 文件，解析出索引层与正文层。

    Args:
        path: skills/<id>.md。
    Returns:
        {"id", "title", "description", "body"}。
    """
    meta, body = _split_frontmatter(path.read_text("utf-8", errors="replace").strip())
    return {
        "id": path.stem,
        "title": _title_of(path, meta, body),
        "description": meta.get("description", "")[:_DESC_MAX] or _fallback_description(body),
        "body": body,
    }


def list_skills() -> list[dict[str, str]]:
    """列出全部画师 skill——只有索引层，不含正文。

    Returns:
        [{id, title, description}]，按 id 排序。
    """
    return [
        {k: s[k] for k in ("id", "title", "description")}
        for s in sorted((_read_skill(p) for p in skills_dir().glob("*.md")),
                        key=lambda s: s["id"])
    ]


def load_skill(skill_id: str) -> str | None:
    """读一个 skill，返回可注入系统提示词的完整画师卡（正文层）。

    Args:
        skill_id: skills/ 下的文件名去扩展名。
    Returns:
        指令头 + 正文；不存在或文件名非法返回 None。
    """
    if not _SAFE_ID.match(skill_id or ""):
        return None
    path = skills_dir() / f"{skill_id}.md"
    if not path.is_file():
        return None
    s = _read_skill(path)
    if not s["body"]:
        return None
    return (
        f"【画师风格 Skill：{s['title']}】\n"
        "本作启用该画师风格卡，它优先于默认的风格指引：\n"
        "- save_script 的 style 字段必须提炼这张卡的「视觉基因」撰写"
        "（保留线条/人物/色彩/影调的关键特征，遵守「避开」清单，可按故事微调措辞）；\n"
        "- style 字段结尾必须原样保留该卡「提示词配方」里的质量锚点"
        "（它们会拼进每一格的生图提示词，是画面质量的下限）；\n"
        "- 每格 prompt 按该卡「提示词配方」的骨架组织，措辞密度向「示范」看齐——"
        "具体名词多过形容词；\n"
        "- 「避开」在分镜 prompt 里一律改用其对应的正向替代词（如 ❌明快糖果色→✅高反差黑白），"
        "不要直接写\"不要X\"——负向词会被画面模型吸进画面；\n"
        "- characters 的 look 卡与每格 prompt 的措辞都要与该画师气质一致；\n"
        "- 叙事节奏可参考「叙事气质」，但故事本身仍以用户的想法为准。\n\n"
        f"{s['body']}"
    )


def save_skill(filename: str, content: str) -> dict[str, str]:
    """保存用户上传的自定义 skill（Web 上传入口）。

    Args:
        filename: 上传文件名（必须 .md）。
        content: md 正文（可带 frontmatter，没有则标题/简介自动回落）。
    Returns:
        {"id", "title"}。
    Raises:
        ValueError: 扩展名不对 / 内容为空 / 超大 / 文件名无法转成安全 id。
    """
    if not (filename or "").lower().endswith(".md"):
        raise ValueError("skill 文件必须是 .md")
    stem = re.sub(r"[^\w\u4e00-\u9fff-]", "_", Path(filename).stem).strip("_")
    if not _SAFE_ID.match(stem):
        raise ValueError("文件名无法转成合法 skill id（请用字母/数字/中文/连字符）")
    text = (content or "").strip()
    if not text:
        raise ValueError("skill 内容为空")
    if len(text.encode("utf-8")) > MAX_SKILL_BYTES:
        raise ValueError(f"skill 太大（上限 {MAX_SKILL_BYTES // 1000}KB）")
    skills_dir().mkdir(parents=True, exist_ok=True)
    path = skills_dir() / f"{stem}.md"
    path.write_text(text + "\n", "utf-8")
    s = _read_skill(path)
    return {"id": s["id"], "title": s["title"]}


def apply_skill(base_prompt: str, skill_id: str | None) -> str:
    """把选中的画师卡拼到基础系统提示词后面（正文层注入点）。

    Args:
        base_prompt: 场景提示词（COMIC_PROMPT）。
        skill_id: 画师 skill id；空/None 表示不启用，返回原文。
    Returns:
        拼接后的系统提示词。
    Raises:
        ValueError: skill_id 不存在（报错里列出全部可选项）。
    """
    if not skill_id:
        return base_prompt
    block = load_skill(skill_id)
    if block is None:
        available = ", ".join(s["id"] for s in list_skills()) or "（无，可上传 .md）"
        raise ValueError(f"unknown style skill '{skill_id}'; available: {available}")
    return f"{base_prompt}\n\n{block}"


# ---------------------------------------------------------------------------
# 索引层：模型自选画师
# ---------------------------------------------------------------------------


def skill_index_text() -> str:
    """把索引层排成给模型看的一段文字（无正文，省 token）。"""
    lines = ["可用画师索引（id｜画师：一句话简介）："]
    lines += [f"- {s['id']}｜{s['title']}：{s['description']}" for s in list_skills()]
    return "\n".join(lines)


def selection_messages(idea: str) -> list[Message]:
    """构造画师选择器的一次对话：索引层进上下文，正文不进。

    Args:
        idea: 用户的漫画想法。
    Returns:
        [system 选择规则, user 想法+索引]。
    """
    system = (
        "你是漫画画师选择器。读用户的漫画想法，从画师索引里选出气质最匹配的一位。\n"
        "标准：题材落在画师的适配范围内；故事情绪与画师叙事气质同频；"
        "避开明显不适配的（如温暖童话别选恐怖画师）。\n"
        '只输出一个 JSON 对象：{"skill": "画师id", "reason": "25字以内理由"}，'
        "不要输出任何其他内容。"
    )
    return [
        Message.system(system),
        Message.user(f"漫画想法：{idea}\n\n{skill_index_text()}"),
    ]


def parse_selection(reply: str, valid_ids: list[str]) -> dict[str, str] | None:
    """从模型回复里抠出选择结果。

    Args:
        reply: 模型原始回复（容忍 ```json 围栏与前后废话）。
        valid_ids: 索引里的合法 id 列表。
    Returns:
        {"id", "reason"}；解析失败或 id 不合法返回 None。
    """
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    sid = str(data.get("skill", "")).strip()
    if sid not in valid_ids:
        return None
    return {"id": sid, "reason": str(data.get("reason", "")).strip() or "按剧本气质匹配"}


async def choose_skill(idea: str, llm: BaseLLM) -> dict[str, str] | None:
    """让模型按剧本气质自选一位画师（索引层一次调用，失败纠正一次）。

    Args:
        idea: 用户的漫画想法。
        llm: 文本模型（纯文本对话，不带工具）。
    Returns:
        {"id", "title", "reason"}；两次都无法解析返回 None（调用方回落为不启用）。
    """
    items = list_skills()
    if not items:
        return None
    valid = [s["id"] for s in items]
    messages = selection_messages(idea)
    for _ in range(2):  # 第一次失败给一次纠正机会，别多——选择不值得烧钱
        resp = await llm.ask(messages)
        picked = parse_selection(resp.content or "", valid)
        if picked:
            picked["title"] = next(s["title"] for s in items if s["id"] == picked["id"])
            return picked
        messages.append(resp.to_message())
        messages.append(
            Message.user(
                "你返回的内容无法解析或 id 不在索引里。"
                f'可选 id：{", ".join(valid)}。重新只输出 JSON：'
                '{"skill": "...", "reason": "..."}'
            )
        )
    return None
