"""规划层：输入分析 → 页数 / 角色 / 拍点 / 画师 / 预估花费 → 用户确认后再烧图。

三种输入都收（对应三类真实用法）：
  一句想法 —— kind=short，1 页，走原有 agent 流水线；
  几段/整章 —— kind=long，切成 N 页（每页一块原文 + 一个拍点），一页一页生成；
  会话续写 —— 带 sid 来，规划器看前情判断"接续本作品"还是"另起新作"。

超长作品的一致性策略（不靠把全部历史塞进上下文）：
  风格 —— 规划期写好全局风格卡，全部页共用一份剧本（生图时自动拼卡）；
  人物 —— 规划期定下角色表（name + look 外貌卡），全部页共用；
  剧情 —— 规划期给出全部拍点 outline，第 k 页的编剧能看到前 k-1 页拍点当"前情"。
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from minimuse.schema import Message
from minimuse.skills import list_skills

MAX_PLAN_PAGES = 20  # 单次生成的页数上限；整本小说分批，发后续文本继续追加
SHORT_IDEA_CHARS = 1500  # 超过视为长文本（影响规划失败的确定性兜底）
MAX_INPUT_CHARS = 30000  # 规划入口的长度保护
_PANELS_PER_PAGE_EST = 4  # 每页平均格数（1-8 槽的版式的经验均值）


def price_per_panel() -> float:
    """单格生图单价（CNY）。中转/画师计价不同，用环境变量覆盖。"""
    try:
        return float(os.environ.get("MINIMUSE_PRICE_PANEL", "0.3"))
    except ValueError:
        return 0.3


def cost_estimate(pages: int) -> dict[str, Any]:
    """按页数估花费：图片是大头（格数 × 单价），文本调用可忽略。

    Returns:
        {pages, panels_est, panel_price, cny, lo, hi, note}；lo/hi 按每页
        3-6 格的波动给出区间——规划只是预估，实际以 API 账单为准。
    """
    price = price_per_panel()
    panels = max(1, pages) * _PANELS_PER_PAGE_EST
    text_fee = 0.01
    return {
        "pages": pages,
        "panels_est": panels,
        "panel_price": price,
        "cny": round(panels * price + text_fee, 2),
        "lo": round(max(1, pages) * 3 * price + text_fee, 2),
        "hi": round(max(1, pages) * 6 * price + text_fee, 2),
        "note": (
            f"按每格 ¥{price} 估算（环境变量 MINIMUSE_PRICE_PANEL 可改），"
            "文本调用合计约 ¥0.01；实际以 API 账单为准"
        ),
    }


def chunk_text(text: str, n: int) -> list[str]:
    """把长文本近似均分成 n 块（段落边界优先，句子次之），对应 n 页。

    Args:
        text: 原文。
        n: 页数。
    Returns:
        长度恰为 n 的块列表；原文不够分时用空串补位（该页靠拍点创作）。
    """
    text = (text or "").strip()
    if n <= 1:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < n:  # 段落不够按页分，退到句子边界
        parts = re.split(r"(?<=[。！？!?…])", text)
        paras = [p.strip() for p in parts if p.strip()]
    if not paras:
        return [""] * n
    if len(paras) <= n:
        return paras + [""] * (n - len(paras))
    target = math.ceil(len(text) / n)
    chunks: list[str] = []
    buf = ""
    for p in paras:
        buf += (("\n\n" + p) if buf else p)
        if len(buf) >= target and len(chunks) < n - 1:
            chunks.append(buf)
            buf = ""
    chunks.append(buf)
    while len(chunks) > n:  # 兜底：合并多出来的尾块
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()
    while len(chunks) < n:
        chunks.append("")
    return chunks


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型回复里抠出第一个 JSON 对象（容忍 ```json 围栏与前后废话）。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def session_context(snap: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """从会话快照提取给规划器的"前情"（标题/角色表/拍点/梗概）。

    Args:
        snap: 会话快照（session.json 内容）。
        workspace: 工作区（读剧本梗概用）。
    Returns:
        前情 dict；没有可用的作品信息时字段尽量为空。
    """
    work = snap.get("work") or {}
    pages = work.get("pages") or []
    base = work.get("base") or snap.get("slug", "")
    synopsis = ""
    if base:
        try:
            data = json.loads(
                (workspace / "scripts" / f"{base}.json").read_text("utf-8")
            )
            synopsis = str(data.get("synopsis", ""))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "title": work.get("title") or snap.get("idea", ""),
        "pages_done": len(pages),
        "beats": [f"第{p['n']}页：{p.get('beat', '')}" for p in pages[-10:]],
        "characters": work.get("characters") or [],
        "synopsis": synopsis,
    }


_PLAN_SYSTEM = """你是漫画连载规划器。读用户的输入（可能是一句话、零散的几句、
一段文字，也可能是整章小说），决定如何把它改编成 1080x1720 的竖版条漫页。
只输出一个 JSON 对象，不要输出任何其他内容。

输出格式：
{
  "kind": "short 或 long",
  "title": "作品标题（不超过 16 个字）",
  "synopsis": "2-3 句故事梗概",
  "style_card": "全局风格卡：60-120 字中文连续指令（线条/色板/光影/构图/质感），给生图模型看",
  "characters": [{"name": "角色名", "look": "中文外貌卡：年龄/体型/发型/服装/标志性特征"}],
  "pages": 3,
  "outline": [{"page": 1, "beat": "这一页的核心事件（1-2 句）", "characters": ["出场角色名"]}],
  "skill": "画师 id 或空串",
  "skill_reason": "20 字以内理由",
  "note": "给用户看的一句话说明（可空）"
}

规则：
- 输入是一句话或零散想法：kind=short，pages=1，outline 给空数组。
- 输入是成段文字/章节/小说：kind=long；pages 按情节容量定（约每 1200 字一页），
  上限 20 页——装不下的写进 note（提示"后续可继续发送原文追加页面"）；
  outline 必须恰好 pages 条，page 从 1 连续编号，每条 beat 是该页的核心事件。
- characters 最多 4 个（工具层上限）：选真正的主角与关键配角，次要角色并进 look 描述；
- 若提供了【已上传角色参考图】：参考图角色是本作的"演员"，由他们出演原文里的角色。
  此时输出里必须包含 "cast_map" 字段（没有参考图时给空数组 []）：
  "cast_map": [{"from": "原文角色名", "to": "参考图角色名"}]，最多 4 条。
  例：参考图角色是"熊二"，原文主角是"萧炎"，就写 {"from": "萧炎", "to": "熊二"}。
  被替换的原文角色不再进 characters；characters 以参考图角色打头
  （look 卡与参考图描述一致），原文里未被映射的配角可正常立卡；
  synopsis 与 outline 拍点提到被替换角色时，写「熊二（饰 萧炎）」或直接用熊二；
  look 卡要具体到年龄/发型/服装/标志特征——这是全部页面人物一致性的唯一依据。
- 拍点连成一条完整故事线：第 1 页建场，中间推进转折，最后一页留余韵。
- skill 从画师索引里选气质最匹配的一位；都不合适给空串。
- 若提供了【会话前情】：判断用户新输入是"本作品的后续"（continuation=true：
  outline 的 page 从 已完成页数+1 开始编号，沿用并按需扩充前情角色表），
  还是一个不相关的新故事（continuation=false：按新作品规划）。加一个
  "continuation_reason": "20 字以内判断依据"。
"""


def _truncate_for_plan(text: str, head: int = 6000, tail: int = 2000) -> str:
    """长文本给规划器看的形态：头部 + 省略标记 + 尾部（保住开头与结局）。"""
    if len(text) <= head + tail + 200:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n……（中间省略 {omitted} 字）……\n{text[-tail:]}"


def normalize_plan(
    raw: dict[str, Any],
    *,
    idea: str,
    session_ctx: dict[str, Any] | None,
    valid_skills: list[str],
) -> dict[str, Any]:
    """把规划器的原始输出修整成可信形状：越界值收敛、缺项回落、编号连续。

    Args:
        raw: 模型给的 JSON。
        idea: 用户原文（推断 kind 的兜底依据）。
        session_ctx: 会话前情；None 表示没有可续的作品。
        valid_skills: 合法画师 id 列表。
    Returns:
        修整后的 plan dict。
    """

    def _s(v: Any) -> str:
        return str(v or "").strip()

    kind = _s(raw.get("kind")).lower()
    if kind not in ("short", "long"):
        kind = "long" if len(idea) > SHORT_IDEA_CHARS else "short"
    outline_raw = raw.get("outline") if isinstance(raw.get("outline"), list) else []
    outline = []
    for item in outline_raw:
        if not isinstance(item, dict):
            continue
        beat = _s(item.get("beat"))
        if not beat:
            continue
        chars = item.get("characters")
        outline.append({
            "beat": beat,
            "characters": [str(c).strip() for c in chars] if isinstance(chars, list) else [],
        })
    if kind == "short":
        outline = []
        pages = 1
    else:
        pages = raw.get("pages")
        pages = pages if isinstance(pages, int) and not isinstance(pages, bool) else 0
        # long：pages 缺失先看拍点数，再看输入体量（约每 1200 字一页）；封顶保护
        if pages < 1:
            pages = len(outline) or max(2, min(MAX_PLAN_PAGES, round(len(idea) / 1200) or 2))
        pages = min(pages, MAX_PLAN_PAGES)
        if not outline:
            # 长篇却没给拍点：造占位拍点（编剧仍会拿到对应原文块）
            outline = [{"beat": f"按原文顺序改编本页内容（第 {i} 页）",
                        "characters": []} for i in range(1, pages + 1)]
        elif len(outline) != pages:
            pages = min(len(outline), MAX_PLAN_PAGES)
        outline = outline[:pages]
        for i, item in enumerate(outline, start=1):
            item["page"] = i
    characters = []
    for c in raw.get("characters") if isinstance(raw.get("characters"), list) else []:
        if not isinstance(c, dict):
            continue
        name, look = _s(c.get("name")), _s(c.get("look"))
        if name and look:
            characters.append({"name": name, "look": look})
        if len(characters) >= 4:
            break
    skill = _s(raw.get("skill"))
    if skill not in valid_skills:
        skill = ""
    continuation = bool(session_ctx) and bool(raw.get("continuation"))
    cast_map = []
    raw_map = raw.get("cast_map") if isinstance(raw.get("cast_map"), list) else []
    for m in raw_map:
        if not isinstance(m, dict):
            continue
        src, dst = _s(m.get("from")), _s(m.get("to"))
        if src and dst and src != dst:
            cast_map.append({"from": src, "to": dst})
        if len(cast_map) >= 4:
            break
    return {
        "kind": kind,
        "title": _s(raw.get("title"))[:24] or (session_ctx or {}).get("title", "")[:24]
                 or "未命名作品",
        "synopsis": _s(raw.get("synopsis")),
        "style_card": _s(raw.get("style_card")),
        "characters": characters,
        "pages": pages,
        "outline": outline,
        "cast_map": cast_map,
        "skill": skill,
        "skill_reason": _s(raw.get("skill_reason")),
        "continuation": continuation,
        "continuation_reason": _s(raw.get("continuation_reason")),
        "note": _s(raw.get("note")),
    }


async def build_plan(
    idea: str,
    llm: Any,
    *,
    session_ctx: dict[str, Any] | None = None,
    valid_skills: list[str] | None = None,
) -> dict[str, Any]:
    """跑一次规划调用（失败纠正一次），返回修整后的 plan。

    Args:
        idea: 用户输入原文（任意长度）。
        llm: 文本模型。
        session_ctx: 会话前情（续写判断用）；None = 全新作品。
        valid_skills: 合法画师 id；None 则读 skills/ 索引。
    Returns:
        normalize_plan 之后的 plan dict。
    Raises:
        ValueError: 两次都无法解析出合法 JSON（别瞎猜，让用户看到真实失败）。
    """
    valid = valid_skills if valid_skills is not None else [s["id"] for s in list_skills()]
    index = "\n".join(f"- {s['id']}｜{s['title']}：{s['description']}" for s in list_skills())
    ctx_block = ""
    if session_ctx:
        chars = "\n".join(
            f"- {c['name']}：{c['look']}" for c in session_ctx.get("characters", [])
        )
        beats = "\n".join(session_ctx.get("beats", []))
        ctx_block = (
            f"【会话前情】用户在这个会话里已有作品《{session_ctx.get('title', '')}》"
            f"（已完成 {session_ctx.get('pages_done', 0)} 页），新输入见下文。\n"
            f"角色表：\n{chars or '（无）'}\n前情拍点：\n{beats or '（无）'}\n"
            f"剧本梗概：{session_ctx.get('synopsis', '') or '（无）'}\n"
            "请判断新输入是后续（continuation=true，页码从 "
            f"{session_ctx.get('pages_done', 0) + 1} 开始）还是新故事（continuation=false）。\n\n"
        )
    messages = [
        Message.system(_PLAN_SYSTEM),
        Message.user(
            f"{ctx_block}【画师索引】\n{index or '（无）'}\n\n"
            f"【用户输入】\n{_truncate_for_plan(idea)}"
        ),
    ]
    for _ in range(2):
        resp = await llm.ask(messages)
        raw = _extract_json(resp.content or "")
        if raw is not None:
            return normalize_plan(raw, idea=idea, session_ctx=session_ctx, valid_skills=valid)
        messages.append(resp.to_message())
        messages.append(Message.user("你返回的内容无法解析。重新只输出一个 JSON 对象。"))
    raise ValueError("规划器两次未能给出合法 JSON，请重试或换更清晰的输入")
