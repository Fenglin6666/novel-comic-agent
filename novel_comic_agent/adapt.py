"""长文本 → 分页漫画：一页一页生成，页间共享同一份剧本。

页内流水线（创作的给模型，确定的归代码）：
  第 k 页原文块 + 拍点 + 前情拍点 → LLM 写分镜 JSON → 校验落盘
  → plan_layout → 补缺格（已存在的格子跳过 = 断点可续）→ compose_page

一致性（对应"超长会话怎么不崩"）：
  风格 / 人物 —— 全部页共用一份剧本（风格卡 + 角色表），生图时自动拼进每格提示词；
  剧情 —— 规划期的全部拍点是脊柱，第 k 页只看"前情拍点摘要 + 本页拍点 + 本页原文"，
          不需要把整部小说或全部历史塞进上下文。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from novel_comic_agent.comic_tools import (
    ComposePage,
    GeneratePanel,
    PlanLayout,
    SaveStoryboard,
    missing_panels,
)
from novel_comic_agent.imagegen import BaseImageGen
from novel_comic_agent.llm import BaseLLM
from novel_comic_agent.plan import _extract_json
from novel_comic_agent.schema import Message, ToolResult

CHUNK_MAX = 2400  # 单页原文块给编剧的上限（拍点负责剧情，原文负责细节）

ADAPT_SYSTEM = """你是条漫分镜作家，为《{title}》写第 {page_no} 页的分镜
（1080x1720 竖版条漫；台词由代码渲染，你只管画面与台词文本）。
只输出一个 JSON 对象，格式：
{{"layout": "版式名", "panels": [ ... ]}}

layout 可选（括号内是槽数，panels 数组长度必须等于槽数）：
solo(1) duo(2) trio(3) action4(4,不规则斜切,动作张力页) big_open(4,大开页+三连格)
five(5) classic(6,规整2x3) grid8(8)

每格字段：
- n：从 1 连续编号
- scene：一句话，这格发生了什么
- caption：旁白，一句以内，可空
- dialogue：[{{"speaker": "...", "text": "...", "anchor": "tl/tr/bl/br/c"}}]，
  每句不超过 16 字、每格最多 3 句；可空
- cast：这格出场角色的名字列表（缺省=说话者）
- prompt：40-100 字中文，只描述画面可见之物，供生图模型使用

prompt 写法（每格都要遵守）：
- 按"镜头→主体动作→环境→光色→质量"组织；先定景别与机位（特写/中景/全景 +
  平视/俯仰/倾斜/贴地），再填主体；
- 主体写身体的精确状态（"她攥紧车票，指节发白"），不写情绪形容词；
- 环境至少一个可数细节（物件/材质/天气）；光线写方向+色温+光比；
- 用专业词汇：轮廓光、伦勃朗光、逆光剪影、浅景深、贴地平视、斜切构图；
- 正向框架：写想要的，不写"没有/不要X"——负向词会被画面模型吸进画面；
- 画面中不出现任何文字；风格卡与角色外貌卡会自动拼进提示词，不要在 prompt 里重复。

节奏：
- 每页是一个呼吸单位：第一格承接上一页结尾（或建场），
  最大/最不规则的一格留给本页情绪最高点，最后一格落在可感的画面上；
- 只画本页拍点：剧情不许超前到后面的拍点，也不许滞留重复。

人物一致性：
- speaker/cast 只能用【角色表】里的名字；
- 角色表里标注（有参考图）的角色：prompt 只写名字、动作与构图，
  不要描写其长相与服装——外貌由输入参考图锁定，文字描写反而会让模型偏离参考图；
- 无参考图的角色：远景或多人时，在 prompt 里重复其标志性特征（跨页保持同一人）。"""


def remaining_work(beats, chunks, offset):
    """恢复/续跑时截取还没完成的部分。

    本计划从 plan_start 页开始跑，start_no 是当前应跑的页码——
    前面的页已完成（或属于上一批），把它们的拍点与原文块切掉，
    避免恢复时把旧拍点当成新页重跑。
    """
    offset = max(0, offset)
    return beats[offset:], chunks[offset:]


def digest_text(prefix: str, beats: list[str], start_no: int) -> str:
    """前情提要：已完成页的摘要 + 本批之前页的拍点，一行串起（控制长度）。"""
    parts = [prefix.strip("；;")] if prefix.strip("；;") else []
    parts += [f"第{start_no + i}页：{beat}" for i, beat in enumerate(beats)]
    text = "；".join(p for p in parts if p)
    return text[-1200:]  # 兜底截断：提要只增不减，别把上下文撑爆


async def write_page(
    llm: BaseLLM,
    workspace: Path,
    *,
    base: str,
    page_no: int,
    title: str,
    style_card: str,
    characters: list[dict[str, str]],
    synopsis: str,
    beat: str,
    digest: str,
    chunk: str,
    cast_map: list[dict[str, str]] | None = None,
    ref_names: list[str] | None = None,
) -> ToolResult:
    """让模型写第 page_no 页的分镜 JSON，校验落盘（失败带错误纠正一次）。

    Args:
        base: 剧本 slug（分镜会校验 speaker/cast 与剧本角色表一致）。
        chunk: 本页原文块；空串则纯靠拍点创作。
        cast_map: 原文角色 → 参考图角色的替换映射（改写小说不改剧情用）。
        ref_names: 有参考图的角色名——prompt 勿描写其外貌，由垫图锁定。
    Returns:
        SaveStoryboard 的成功结果；两次都失败返回可读错误。
    """
    ref_set = set(ref_names or [])
    chars_block = "\n".join(
        f"- {c['name']}：{c['look']}"
        + ("（有参考图，勿描写外貌）" if c["name"] in ref_set else "")
        for c in characters
    ) or "（无）"
    cast_block = ""
    if cast_map:
        maps = "；".join(f"原文里的「{m['from']}」就是「{m['to']}」" for m in cast_map)
        cast_block = (
            f"【角色替换】{maps}。台词的 speaker、每格 cast、prompt 里的称呼"
            "一律用替换后的名字，原名字不得出现在本页任何字段中。\n"
        )
    user = (
        f"【作品】《{title}》——{synopsis or '（无梗概）'}\n"
        f"【全局风格卡】{style_card or '（沿用默认风格指引）'}\n"
        f"【角色表】\n{chars_block}\n"
        f"{cast_block}"
        f"【前情】{digest or '（本页是第一页）'}\n"
        f"【本页拍点 · 第 {page_no} 页】{beat}\n"
        f"【本页原文】\n{chunk[:CHUNK_MAX] or '（原文没有对应段落，请依据拍点与角色表创作这一页）'}"
    )
    messages = [
        Message.system(ADAPT_SYSTEM.format(title=title, page_no=page_no)),
        Message.user(user),
    ]
    sb = SaveStoryboard(workspace=workspace)
    for _ in range(2):
        resp = await llm.ask(messages)
        data = _extract_json(resp.content or "")
        if data is not None:
            r = await sb.execute(
                script=base,
                page=page_no,
                layout=str(data.get("layout") or "auto"),
                panels=data.get("panels"),
            )
            if r.ok:
                return r
            err = r.error
        else:
            err = "回复不是合法 JSON"
        messages.append(resp.to_message())
        messages.append(Message.user(f"{err}\n请修正后重新只输出一个 JSON 对象。"))
    return ToolResult.fail("分镜写作两次未通过校验")


async def generate_pages(
    workspace: Path,
    *,
    base: str,
    beats: list[dict[str, Any]],
    chunks: list[str],
    start_no: int,
    script: dict[str, Any],
    digest_prefix: str,
    llm: BaseLLM,
    imagegen: BaseImageGen,
    panel_sizes: dict[str, str],
    cast_map: list[dict[str, str]] | None = None,
    ref_names: list[str] | None = None,
    echo: Any = None,
    on_page: Any = None,
) -> tuple[bool, str, int]:
    """逐页跑完整流水线；已完成页/格自动跳过（这就是断点续跑）。

    Args:
        base: 剧本 slug。
        beats: 本批拍点（plan.outline）。
        chunks: 与 beats 对齐的原文块。
        start_no: 本批第一页的页码（续写时 > 1）。
        script: 剧本数据（title/style/characters/synopsis）。
        digest_prefix: 本批之前的前情（已完成页的拍点摘要）。
        cast_map: 原文角色 → 参考图角色的替换映射（逐页透传给编剧）。
        echo: 进度回调（接 SSE 队列）。
        on_page: 每页完成回调 (page_no, slug, beat, skipped)。
    Returns:
        (ok, message, done_pages)。
    """

    def say(text: str) -> None:
        if echo is not None:
            echo(text)

    planner = PlanLayout(workspace=workspace, panel_sizes=panel_sizes)
    gen = GeneratePanel(workspace=workspace, imagegen=imagegen, panel_sizes=panel_sizes)
    composer = ComposePage(workspace=workspace)
    done = 0
    for i, item in enumerate(beats):
        page_no = start_no + i
        slug = f"{base}-p{page_no:02d}"  # 流水线内页页带页码；base 只留给剧本与旧的单页作品
        beat = str(item.get("beat", ""))
        # 断点续跑：成品页已在且格无缺 → 跳过
        if (workspace / "pages" / f"{slug}.png").is_file():
            try:
                missing = missing_panels(workspace, slug)
            except Exception:  # noqa: BLE001 — 分镜损坏视同未完成
                missing = [0]
            if not missing:
                say(f"[第 {page_no} 页] 已完成，跳过")
                done += 1
                if on_page is not None:
                    on_page(page_no, slug, beat, True)
                continue
        # 1) 分镜：缺了才写（写过的页保持原样，重画格子不重写剧情）
        if (workspace / "storyboards" / f"{slug}.json").is_file():
            say(f"[第 {page_no} 页] 分镜已存在，沿用")
        else:
            r = await write_page(
                llm,
                workspace,
                base=base,
                page_no=page_no,
                title=str(script.get("title", "")),
                style_card=str(script.get("style", "")),
                characters=script.get("characters", []),
                synopsis=str(script.get("synopsis", "")),
                beat=beat,
                digest=digest_text(digest_prefix, [b.get("beat", "") for b in beats[:i]], start_no),
                chunk=chunks[i] if i < len(chunks) else "",
                cast_map=cast_map,
                ref_names=ref_names,
            )
            if not r.ok:
                return False, f"第 {page_no} 页分镜失败：{r.error}", done
            say(r.output.splitlines()[0])
        # 2) 版式
        if not (workspace / "layouts" / f"{slug}.json").is_file():
            r = await planner.execute(storyboard=slug)
            if not r.ok:
                return False, f"第 {page_no} 页版式失败：{r.error}", done
        # 3) 补缺格
        for n in missing_panels(workspace, slug):
            r = await gen.execute(storyboard=slug, panel=n)
            if not r.ok:  # 同参数重试一次（瞬时抖动常见，画师失败别直接弃整页）
                r = await gen.execute(storyboard=slug, panel=n)
                if not r.ok:
                    return False, f"第 {page_no} 页第 {n} 格失败：{r.error}", done
            say(r.output.splitlines()[0])
        # 4) 合成
        r = await composer.execute(storyboard=slug)
        if not r.ok:
            return False, f"第 {page_no} 页合成失败：{r.error}", done
        say(r.output.splitlines()[0])
        done += 1
        if on_page is not None:
            on_page(page_no, slug, beat, False)
    return True, f"本次完成 {done} 页", done
