r"""novel-comic-agent 的 Web 界面。

启动：.\run-web.ps1（自动带密钥并打开浏览器）
或手动配好环境变量后：.\.venv\Scripts\python -m uvicorn web:app --port 8765

接口一览：
  GET  /                          单页界面（web/index.html）
  POST /api/plan                  规划：页数/角色/拍点/画师 + 预估花费（不烧图）
  GET  /api/generate?sid=         按已确认的规划生成本次全部页（SSE；页幂等可续）
  GET  /api/run?idea=             旧的一次性直通生成（SSE；前端已改走 plan→generate）
  GET/POST /api/config            模型配置：文本/生图的密钥、端点、模型名（密钥只回尾号）
  GET  /api/works                 历史作品（comic_out/pages）
  /files/*                        comic_out 静态文件
  /static/*                       前端静态资源（web/ 下的本地 vue 等）
  GET/POST/DELETE /api/refs       角色参考图（垫图）清单 / 上传 / 删除
  GET  /api/sessions              会话列表（含可恢复标记）
  GET/PATCH/DELETE /api/sessions/{sid}  详情+回放 / 重命名 / 删除无产物会话
  GET  /api/sessions/{sid}/resume SSE 流：单页断点续跑（不重放模型决策）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comic import COMIC_PROMPT
from novel_comic_agent.adapt import generate_pages, remaining_work
from novel_comic_agent.agent import Agent
from novel_comic_agent.comic_tools import (
    ComposePage,
    GeneratePanel,
    PlanLayout,
    SaveScript,
    SaveStoryboard,
    size_map_for,
    storyboard_slug,
)
from novel_comic_agent.imagegen import OpenAIImageGen
from novel_comic_agent.llm import OpenAILLM
from novel_comic_agent.plan import (
    MAX_INPUT_CHARS,
    build_plan,
    chunk_text,
    cost_estimate,
    session_context,
)
from novel_comic_agent.refs import (
    decode_data_url,
    delete_ref,
    load_refs,
    refs_dir,
    refs_prompt,
    save_ref,
    update_desc,
)
from novel_comic_agent.session import SessionState, SessionStore, SessionTracker, resume_pipeline
from novel_comic_agent.skills import apply_skill, choose_skill, list_skills, load_skill, save_skill
from novel_comic_agent.tools import ToolCollection
from novel_comic_agent.vision import describe_image
from novel_comic_agent.webconfig import apply_config, public_view, save_config

OUT_DIR = (Path(__file__).parent / "comic_out").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = Path(__file__).parent / "web"
STORE = SessionStore(OUT_DIR / "sessions")
apply_config(OUT_DIR)  # 启动即应用前端保存的模型配置（覆盖在 run-*.ps1 之上）

app = FastAPI(title="novel-comic-agent")
app.mount("/files", StaticFiles(directory=OUT_DIR), name="files")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")  # 前端静态资源（本地 vue 等）

_busy = False  # 生图花钱，一次只允许跑一个；FastAPI 单事件循环，同步检查无竞态
_WIN_PNG = re.compile(r"[A-Za-z]:\\[^\s]+?\.png")


@app.get("/")
def index() -> FileResponse:
    """返回单页界面 web/index.html。"""
    return FileResponse(WEB_DIR / "index.html")


# ---------------------------------------------------------------- 画师 skill


class ModelConfig(BaseModel):
    """一组模型的配置项；均为可选，空/缺省 = 保持现状。"""

    api_key: str = ""
    base_url: str = ""
    model: str = ""


class ConfigUpdate(BaseModel):
    """前端模型配置请求体：文本与生图两组，互相独立。"""

    text: ModelConfig = ModelConfig()
    image: ModelConfig = ModelConfig()


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """当前模型配置视图：端点/模型明文，密钥只给尾号掩码（永不回传明文）。"""
    return public_view(OUT_DIR)


@app.post("/api/config")
def update_config(body: ConfigUpdate) -> dict[str, Any]:
    """保存前端提交的模型配置：写环境变量 + 落盘 web_config.json，立即生效。

    空/缺省字段保持现状；保存会覆盖 run-*.ps1 的同名变量（重启后由本文件恢复）。
    """
    save_config(OUT_DIR, body.text.model_dump(), body.image.model_dump())
    return public_view(OUT_DIR)


@app.get("/api/skills")
def skills() -> list[dict[str, str]]:
    """列出可选画师 skill（索引层：id/title/description）。"""
    return list_skills()


class SkillUpload(BaseModel):
    """上传自定义风格 skill 的请求体。"""

    filename: str
    content: str


@app.post("/api/skills")
def upload_skill(body: SkillUpload) -> dict[str, str]:
    """保存上传的自定义风格 skill 到 skills/ 目录，即时可选。"""
    try:
        return save_skill(body.filename, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------- 角色参考图


@app.get("/api/refs")
def get_refs() -> list[dict[str, str]]:
    """列出已上传的角色参考图（file 是落盘文件名，与角色名可能不同）。"""
    return [
        {"name": n, "file": e.get("file", ""), "desc": e.get("desc", "")}
        for n, e in load_refs(OUT_DIR).items()
    ]


class RefUpload(BaseModel):
    """上传角色参考图的请求体。"""

    name: str
    image: str  # data:image/...;base64,...
    desc: str = ""  # 可空：留空则尝试视觉模型自动补


@app.post("/api/refs")
async def upload_ref(body: RefUpload) -> dict[str, Any]:
    """保存角色参考图；描述留空时用视觉模型（如配置）自动生成一句话外貌卡。"""
    try:
        raw = decode_data_url(body.image)
        entry = save_ref(OUT_DIR, body.name, raw, body.desc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not entry["desc"]:
        desc = await describe_image(refs_dir(OUT_DIR) / entry["file"])
        if desc:
            update_desc(OUT_DIR, entry["name"], desc)
            entry["desc"] = desc
    return entry


@app.delete("/api/refs/{name}")
def remove_ref(name: str) -> dict[str, bool]:
    """删除一个角色参考图。"""
    return {"deleted": delete_ref(OUT_DIR, name)}


# ---------------------------------------------------------------- 作品列表


@app.get("/api/works")
def works() -> list[dict[str, Any]]:
    """列出历史作品：pages 下的成品页 + 从分镜里带出标题/版式。

    Returns:
        [{slug, title, layout, page, mtime}]，按修改时间倒序。
    """
    pages = sorted(
        (OUT_DIR / "pages").glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    items = []
    for page in pages:
        slug = page.stem
        title, layout = slug, ""
        sb = OUT_DIR / "storyboards" / f"{slug}.json"
        if sb.is_file():
            try:
                data = json.loads(sb.read_text("utf-8"))
                title, layout = data.get("title", slug), data.get("layout", "")
            except (OSError, json.JSONDecodeError):
                pass
        items.append(
            {
                "slug": slug,
                "title": title,
                "layout": layout,
                "page": f"/files/pages/{page.name}",
                "mtime": page.stat().st_mtime,
            }
        )
    return items


# ---------------------------------------------------------------- 会话


def _enrich(snap: dict[str, Any]) -> dict[str, Any]:
    """给会话快照补 UI 需要的派生字段：是否可恢复/继续、成品页链接、页数进度。"""
    slug = snap.get("slug", "")
    work = snap.get("work") or {}
    has_storyboard = bool(slug and (OUT_DIR / "storyboards" / f"{slug}.json").is_file())
    has_pages = bool(work.get("pages"))
    has_plan = bool(snap.get("plan"))
    snap["resumable"] = (
        (has_storyboard or has_pages or has_plan) and snap.get("state") != "done"
    )
    snap["page"] = f"/files/pages/{slug}.png" if slug else ""
    pages = work.get("pages") or []
    snap["pages_done"] = len(pages)
    snap["pages_total"] = len(pages) or (snap.get("plan") or {}).get("pages", 0)
    return snap


@app.get("/api/sessions")
def sessions() -> list[dict[str, Any]]:
    """列出全部会话（最新在前），带可恢复标记。"""
    return [_enrich(s) for s in STORE.list_sessions()]


@app.get("/api/sessions/{sid}")
def session_detail(sid: str) -> dict[str, Any]:
    """会话详情 + 事件流回放。"""
    snap = STORE.get(sid)
    if snap is None:
        raise HTTPException(status_code=404, detail="no such session")
    return _enrich(snap) | {"events": STORE.events(sid)}


class SessionRename(BaseModel):
    """会话PATCH请求体：改标题（idea）和/或清除未确认的规划（clear_plan）。"""

    idea: str = ""
    clear_plan: bool = False


@app.patch("/api/sessions/{sid}")
def update_session(sid: str, body: SessionRename) -> dict[str, Any]:
    """改会话标题 / 清除未确认的规划（取消规划时用，防止误触恢复烧钱）。"""
    snap = STORE.get(sid)
    if snap is None:
        raise HTTPException(status_code=404, detail="no such session")
    fields: dict[str, Any] = {}
    title = (body.idea or "").strip()
    if title:
        fields["idea"] = title
    if body.clear_plan:
        fields["plan"] = None
    if not fields:
        raise HTTPException(status_code=400, detail="nothing to update")
    return _enrich(STORE.update(sid, **fields))


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str) -> dict[str, bool]:
    """删除一个没有产物的会话（规划确认前的"取消"入口）。

    已有分镜/成品的会话拒绝删除——那是花过钱的作品，不走静默清理。
    Raises:
        HTTPException: 404 不存在；409 已有产物。
    """
    snap = STORE.get(sid)
    if snap is None:
        raise HTTPException(status_code=404, detail="no such session")
    work = snap.get("work") or {}
    slug = snap.get("slug", "")
    if work.get("pages") or (slug and (OUT_DIR / "storyboards" / f"{slug}.json").is_file()):
        raise HTTPException(status_code=409, detail="session has artifacts")
    return {"deleted": STORE.delete(sid)}


def _png_urls(text: str) -> list[str]:
    """从工具回显里抠出图片路径，换成 /files/ 链接（逐格预览靠这个）。"""
    urls = []
    for hit in _WIN_PNG.findall(text):
        try:
            rel = Path(hit).resolve().relative_to(OUT_DIR)
        except ValueError:
            continue
        urls.append("/files/" + rel.as_posix())
    return urls


def _sse(worker: Any) -> StreamingResponse:
    """公共 SSE 包装：queue 里的事件 → event-stream；断开时取消后台任务。

    Args:
        worker: 接收 asyncio.Queue 的协程工厂；结束时放 None 哨兵。
    Returns:
        StreamingResponse（text/event-stream）。
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def stream():
        task = asyncio.create_task(worker(queue))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():  # 浏览器断开：取消后台任务，别让它继续烧钱
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(stream(), media_type="text/event-stream")


def _echo_factory(sid: str, queue: asyncio.Queue, tracker: SessionTracker | None = None):
    """echo 双写工厂：SSE 队列 + 会话事件流（+ 可选的状态机追踪）+ 图片链接事件。"""

    def echo(text: str) -> None:
        queue.put_nowait({"type": "log", "sid": sid, "text": text})
        if tracker is not None:
            tracker.feed(text)
        STORE.append_event(sid, "log", text)
        for url in _png_urls(text):
            queue.put_nowait({"type": "file", "sid": sid, "url": url})
            STORE.append_event(sid, "file", url)

    return echo


def _agent_worker(sid: str, idea: str, skill: str, on_done: Any = None):
    """agent 流水线 worker 工厂（/api/run 与 /api/generate 的单页路径共用）。

    Args:
        on_done: agent 成功后的回调（如回填 work 骨架）；返回值替换最终 answer。
    """
    tracker = SessionTracker(STORE, sid)
    ref_block = refs_prompt(OUT_DIR)
    user_idea = f"{ref_block}\n\n想法：{idea}" if ref_block else idea

    async def worker(queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        """后台跑 agent；状态机由 tracker 驱动；结束释放忙碌闸。"""
        echo = _echo_factory(sid, queue, tracker)
        try:
            llm = OpenAILLM()
            style_id = skill
            if skill == "":  # auto：索引层交给模型判断，选择结果实时推给前端
                picked = await choose_skill(user_idea, llm)
                if picked:
                    style_id = picked["id"]
                    echo(f"🎨 画师自选：{picked['title']}（{picked['id']}）—— {picked['reason']}")
                else:
                    style_id = ""
                    echo("🎨 画师自选失败，使用默认风格指引")
            elif skill == "none":
                style_id = ""
            system_prompt = apply_skill(COMIC_PROMPT, style_id)
            imagegen = OpenAIImageGen()
            sizes = size_map_for(imagegen.model)
            tools = ToolCollection(
                SaveScript(workspace=OUT_DIR),
                SaveStoryboard(workspace=OUT_DIR),
                PlanLayout(workspace=OUT_DIR, panel_sizes=sizes),
                GeneratePanel(workspace=OUT_DIR, imagegen=imagegen, panel_sizes=sizes),
                ComposePage(workspace=OUT_DIR),
            )
            agent = Agent(
                llm,
                tools,
                max_steps=24,
                system_prompt=system_prompt,
                echo=echo,
                debug=bool(os.environ.get("MINIMUSE_DEBUG")),  # MINIMUSE_DEBUG=1 开全量回显
            )
            answer = await agent.run(user_idea)
            if on_done is not None:
                answer = await on_done(answer) or answer
            STORE.set_state(sid, SessionState.DONE)
            STORE.append_event(sid, "done", {"answer": answer})
            queue.put_nowait({"type": "done", "sid": sid, "answer": answer})
        except asyncio.CancelledError:
            STORE.set_state(sid, SessionState.CANCELLED, "client disconnected")
            STORE.append_event(sid, "cancelled", {})
            raise
        except Exception as exc:  # noqa: BLE001 — 失败也要变成事件流给前端
            msg = f"{type(exc).__name__}: {exc}"
            STORE.set_state(sid, SessionState.FAILED, msg)
            STORE.append_event(sid, "error", {"message": msg})
            queue.put_nowait({"type": "error", "sid": sid, "message": msg})
        finally:
            global _busy  # noqa: PLW0603 — 单事件循环内的任务互斥闸
            _busy = False
            queue.put_nowait(None)

    return worker


def _load_script_data(base: str) -> dict[str, Any] | None:
    """读剧本 JSON；不存在/坏 JSON 返回 None。"""
    p = OUT_DIR / "scripts" / f"{base}.json"
    try:
        data = json.loads(p.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def _generate_planned(
    sid: str, plan: dict[str, Any], idea: str, echo: Any
) -> str:
    """按规划逐页生成长文本改编（页与格都幂等，中断后重调即续跑）。

    一致性做法：全部页共用同一份剧本（风格卡 + 角色表，续写只合并新角色、
    不动风格卡）；剧情靠"前情拍点摘要 + 本页拍点 + 本页原文块"逐页接力。
    Returns:
        完成消息（喂给 done 事件）。
    Raises:
        RuntimeError: 任一页失败（已完成页保留，恢复时自动跳过）。
    """
    snap = STORE.get(sid) or {}
    work = snap.get("work") or {}
    base = work.get("base") or snap.get("slug", "")
    beats = plan.get("outline") or []
    pages = work.get("pages") or []
    start_no = (pages[-1]["n"] if pages else (1 if base else 0)) + 1
    # 本计划从第几页开始跑：首次执行时记入 work；失败恢复沿用——
    # 拍点/原文块按 offset 切片，只跑真正缺的页，不会把前面的拍点当成后面的重跑
    plan_start = work.get("plan_start_no") or start_no
    offset = max(0, start_no - plan_start)
    run_beats, run_chunks = remaining_work(beats, chunk_text(idea, len(beats)), offset)

    if base:
        script_data = _load_script_data(base)
        if script_data is None:
            raise RuntimeError(f"剧本 {base} 缺失 — 请重新规划（POST /api/plan）")
        # 续写：规划新增的角色并进同一份剧本；标题/风格卡保持原样（一致性不漂移）
        known = {c["name"] for c in script_data.get("characters", [])}
        merged = script_data.get("characters", []) + [
            c for c in plan.get("characters", []) if c.get("name") not in known
        ]
        if len(merged) > 4:
            merged = merged[:4]  # 工具层上限；规划器被要求只带关键角色
        if len(merged) > len(script_data.get("characters", [])):
            r = await SaveScript(workspace=OUT_DIR).execute(
                title=script_data.get("title", ""),
                style=script_data.get("style", ""),
                characters=merged,
                synopsis=script_data.get("synopsis", ""),
            )
            if not r.ok:
                raise RuntimeError(r.error)
            script_data["characters"] = merged
            echo(f"🔧 角色表合并完成（{len(merged)} 人），沿用原风格卡")
    else:
        r = await SaveScript(workspace=OUT_DIR).execute(
            title=plan.get("title") or "未命名作品",
            style=plan.get("style_card", ""),
            characters=plan.get("characters") or [{"name": "主角", "look": "见原文描述"}],
            synopsis=plan.get("synopsis", ""),
        )
        if not r.ok:
            raise RuntimeError(r.error)
        m = re.search(r"slug=(\S+)", r.output)
        base = m.group(1) if m else storyboard_slug(plan.get("title") or "未命名作品")
        script_data = _load_script_data(base) or {}
        names = [c["name"] for c in script_data.get("characters", [])]
        echo(f"📖 剧本就绪：{base}（角色 {names}）")

    STORE.update(
        sid,
        work={
            "base": base,
            "title": script_data.get("title") or plan.get("title", ""),
            "skill": work.get("skill") or plan.get("skill", ""),
            "characters": script_data.get("characters", []),
            "pages": pages,
            "plan_start_no": plan_start,
        },
    )
    STORE.set_state(sid, SessionState.SCRIPT)
    if not run_beats:
        echo("✓ 规划内的页面均已完成，无需补页")
        return "所有页均已存在"
    end_no = start_no + len(run_beats) - 1
    echo(f"📚 开始逐页生成：第 {start_no}-{end_no} 页"
         f"（本次 {len(run_beats)} 页，每页约 30-60 秒/格）")

    digest_prefix = "；".join(f"第{p['n']}页：{p.get('beat', '')}" for p in pages[-10:])
    llm = OpenAILLM()
    imagegen = OpenAIImageGen()
    sizes = size_map_for(imagegen.model)

    def on_page(no: int, slug: str, beat: str, skipped: bool) -> None:
        cur = STORE.get(sid) or {}
        w = dict(cur.get("work") or {})
        plist = [p for p in (w.get("pages") or []) if p.get("slug") != slug]
        plist.append({"n": no, "slug": slug, "beat": beat, "state": "done"})
        plist.sort(key=lambda p: p.get("n", 0))
        w["pages"] = plist
        STORE.update(sid, work=w)
        STORE.set_state(sid, SessionState.PANELS)

    ok, msg, _ = await generate_pages(
        OUT_DIR,
        base=base,
        beats=run_beats,
        chunks=run_chunks,
        start_no=start_no,
        script=script_data,
        digest_prefix=digest_prefix,
        cast_map=plan.get("cast_map") or [],
        llm=llm,
        imagegen=imagegen,
        panel_sizes=sizes,
        echo=echo,
        on_page=on_page,
    )
    if not ok:
        raise RuntimeError(msg)
    return msg


def _planned_worker(sid: str, plan: dict[str, Any], idea: str):
    """按已确认规划逐页生成的 worker（长文本路径）。"""

    async def worker(queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        echo = _echo_factory(sid, queue)
        try:
            echo(f"📋 按已确认的规划生成《{plan.get('title', '')}》")
            msg = await _generate_planned(sid, plan, idea, echo)
            STORE.set_state(sid, SessionState.DONE)
            STORE.append_event(sid, "done", {"answer": msg})
            queue.put_nowait({"type": "done", "sid": sid, "answer": msg})
        except asyncio.CancelledError:
            STORE.set_state(sid, SessionState.CANCELLED, "client disconnected")
            STORE.append_event(sid, "cancelled", {})
            raise
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            STORE.set_state(sid, SessionState.FAILED, msg)
            STORE.append_event(sid, "error", {"message": msg})
            queue.put_nowait({"type": "error", "sid": sid, "message": msg})
        finally:
            global _busy  # noqa: PLW0603
            _busy = False
            queue.put_nowait(None)

    return worker


class PlanRequest(BaseModel):
    """生成规划请求：idea 必填；带 sid 按"会话续写"处理；skill 可显式指定画师。"""

    idea: str
    sid: str = ""
    skill: str = ""


@app.post("/api/plan")
async def make_plan(body: PlanRequest) -> dict[str, Any]:
    """规划：输入分析 → 页数/角色表/拍点/画师 + 预估花费（只花几分钱文本费）。

    带 sid 时按"会话续写"处理：规划器看到前情，判断接续（页码接着排、沿用
    角色表）还是另起新作。确认后用 GET /api/generate?sid= 真正开始生成。
    Returns:
        {sid, plan, cost, continuing, pages_before, start_no}。
    Raises:
        HTTPException: 400 输入为空/超长/画师非法；404 会话不存在；502 规划失败。
    """
    idea = (body.idea or "").strip()
    if not idea:
        raise HTTPException(status_code=400, detail="idea is empty")
    if len(idea) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"输入超过 {MAX_INPUT_CHARS} 字，请分段发送（规划器按批改编）",
        )
    if body.skill and body.skill != "none" and load_skill(body.skill) is None:
        available = ", ".join(s["id"] for s in list_skills())
        raise HTTPException(status_code=400, detail=f"unknown skill; available: {available}")
    snap = STORE.get(body.sid) if body.sid else None
    if body.sid and snap is None:
        raise HTTPException(status_code=404, detail="no such session")

    ctx = session_context(snap, OUT_DIR) if snap else None
    ref_block = refs_prompt(OUT_DIR)
    plan_idea = f"{ref_block}\n\n用户输入：\n{idea}" if ref_block else idea
    try:
        plan = await build_plan(plan_idea, OpenAILLM(), session_ctx=ctx)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"规划失败：{exc}") from exc
    if body.skill:  # 用户显式指定画师 > 规划器自选
        plan["skill"] = "" if body.skill == "none" else body.skill
        plan["skill_source"] = "user"
    else:
        plan["skill_source"] = "auto"

    continuing = bool(plan.get("continuation") and ctx)
    if continuing:
        sid = body.sid
        work = snap.get("work") or {}
        base = work.get("base") or snap.get("slug", "")
        pages_before = len(work.get("pages") or []) or (1 if base else 0)
        # 新一批拍点从"已完成页数+1"页起跑（恢复同一批时该值已在 work 里，不覆盖语义：
        # pages_before 含本批已完成页时 plan_start 保持不变，见 _generate_planned）
        STORE.update(sid, plan=plan, inputs=_append_input(snap, idea),
                     work={**work, "plan_start_no": pages_before + 1})
    else:
        fresh = STORE.create(idea[:80], plan.get("skill", ""))
        sid = fresh["sid"]
        pages_before = 0
        STORE.update(
            sid,
            plan=plan,
            idea=plan.get("title") or idea[:40],
            inputs=[{"ts": time.time(), "text": idea[:500]}],
            work={
                "base": "",
                "title": plan.get("title", ""),
                "skill": plan.get("skill", ""),
                "characters": plan.get("characters", []),
                "pages": [],
            },
        )
    STORE.append_event(sid, "user", {"text": idea})   # 用户消息进对话存档（回放像聊天）
    STORE.append_event(sid, "plan", plan)
    return {
        "sid": sid,
        "plan": plan,
        "user_text": idea,
        "cost": cost_estimate(plan["pages"], image_model=os.environ.get("IMAGE_MODEL", ""),
                       text_model=os.environ.get("MINIMUSE_MODEL", "")),
        "continuing": continuing,
        "had_sid": bool(body.sid),
        "pages_before": pages_before,
        "start_no": pages_before + 1,
        "has_pages": bool((snap.get("work") or {}).get("pages")) if continuing else False,
    }


def _append_input(snap: dict[str, Any], idea: str) -> list[dict[str, Any]]:
    """把这次输入追加进会话的输入历史（截断存，只做档案不喂模型）。"""
    inputs = list(snap.get("inputs") or [])
    inputs.append({"ts": time.time(), "text": idea[:500]})
    return inputs


@app.get("/api/generate")
async def generate(sid: str):
    """按已确认的规划生成本次全部页（SSE；单页路径复用 agent，多页路径逐页跑）。

    幂等：已完成页/格自动跳过——中断后再次调用即为断点续跑。
    Raises:
        HTTPException: 404 会话不存在或没有规划；409 已有任务在跑。
    """
    snap = STORE.get(sid)
    if snap is None:
        raise HTTPException(status_code=404, detail="no such session")
    plan = snap.get("plan")
    if not plan:
        raise HTTPException(status_code=404, detail="no plan — POST /api/plan first")
    global _busy
    if _busy:
        raise HTTPException(status_code=409, detail="another comic is already generating")
    _busy = True

    idea = ((snap.get("inputs") or [{}])[0].get("text") or "").strip() or snap.get("idea", "")
    if plan.get("kind") == "short":
        slug = snap.get("slug", "")
        sb_exists = bool(slug and (OUT_DIR / "storyboards" / f"{slug}.json").is_file())

        async def on_done(answer: str) -> str:
            # 单页路径：agent 里产生的 slug 回填成 work 骨架（后续可续写成长篇）
            fresh = STORE.get(sid) or {}
            new_slug = fresh.get("slug", "")
            if new_slug:
                STORE.update(
                    sid,
                    work={
                        "base": new_slug,
                        "title": plan.get("title") or fresh.get("idea", ""),
                        "skill": plan.get("skill", ""),
                        "characters": plan.get("characters", []),
                        "pages": [{"n": 1, "slug": new_slug, "beat": "", "state": "done"}],
                    },
                )
                # agent 可能"跳过并说明"后正常收尾——没有成品页就不算完成，
                # 标记失败让恢复按钮出现（补格是确定性的，不再烧一轮 agent）
                if not (OUT_DIR / "pages" / f"{new_slug}.png").is_file():
                    raise RuntimeError(
                        "生成未完成：部分格子失败，未合成成品页——可点「恢复」补齐")
            return answer

        if sb_exists:
            # 分镜已在：确定性补格 + 重合成，不惊动 agent（恢复 = 接着上次，不是重来）
            return _sse(_short_resume_worker(sid, slug, on_done=on_done))
        # 没有分镜产物：只能整轮重跑；若剧本已存过，注入上下文让 agent 沿用而非另起
        script_data = _load_script_data(slug) if slug else None
        if script_data:
            title = script_data.get("title", "")
            idea = (f"【失败重试】此前已保存剧本 slug={slug}（标题《{title}》）。"
                    "请沿用该标题重新 save_script（保持 slug 不变），先检查 "
                    "storyboards/ 与 panels/ 已有产物，只补缺失的步骤，"
                    "不要重做已完成的部分。\n\n" + idea)
        return _sse(_agent_worker(sid, idea, plan.get("skill", ""), on_done=on_done))
    return _sse(_planned_worker(sid, plan, idea))


@app.get("/api/run")
async def run(idea: str, skill: str = ""):
    """旧的一次性直通生成（前端已改走 plan → 确认 → generate；此入口保留兼容）。

    Args:
        idea: 用户的想法（query 参数）。
        skill: 画师 skill id；""=模型自选（auto）；"none"=不启用；其余为显式指定。
    Returns:
        text/event-stream：log / file / done / error 四种 JSON 事件（含 sid）。
    Raises:
        HTTPException: 400 想法为空或画师 id 非法；409 已有任务在跑。
    """
    idea = (idea or "").strip()
    if not idea:
        raise HTTPException(status_code=400, detail="idea is empty")
    if skill and skill != "none" and load_skill(skill) is None:
        available = ", ".join(s["id"] for s in list_skills())
        raise HTTPException(status_code=400, detail=f"unknown skill; available: {available}")
    global _busy
    if _busy:
        raise HTTPException(status_code=409, detail="another comic is already generating")
    _busy = True

    snap = STORE.create(idea, skill)
    return _sse(_agent_worker(snap["sid"], idea, skill))


def _short_resume_worker(sid: str, slug: str, on_done: Any = None):
    """单页确定性恢复 worker：plan_layout → 补缺格 → 重合成（不经过 agent/LLM）。"""

    async def worker(queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        echo = _echo_factory(sid, queue)
        try:
            echo(f"🔧 恢复单页 {slug}：只补缺失的格子，不重写剧本/分镜")
            STORE.set_state(sid, SessionState.PANELS)
            imagegen = OpenAIImageGen()
            ok, msg = await resume_pipeline(
                OUT_DIR, slug, imagegen, size_map_for(imagegen.model), echo
            )
            if ok:
                if on_done is not None:
                    msg = await on_done(msg) or msg
                STORE.set_state(sid, SessionState.DONE)
                STORE.append_event(sid, "done", {"answer": msg})
                queue.put_nowait({"type": "done", "sid": sid, "answer": msg})
            else:
                STORE.set_state(sid, SessionState.FAILED, msg)
                STORE.append_event(sid, "error", {"message": msg})
                queue.put_nowait({"type": "error", "sid": sid, "message": msg})
        except asyncio.CancelledError:
            STORE.set_state(sid, SessionState.CANCELLED, "client disconnected")
            raise
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            STORE.set_state(sid, SessionState.FAILED, msg)
            queue.put_nowait({"type": "error", "sid": sid, "message": msg})
        finally:
            global _busy  # noqa: PLW0603
            _busy = False
            queue.put_nowait(None)

    return worker


@app.get("/api/sessions/{sid}/resume")
async def resume_session(sid: str):
    """从产物断点续跑一个会话（补缺格 → 重合成；不经过 LLM）。

    Returns:
        text/event-stream：与 /api/run 相同的事件类型。
    Raises:
        HTTPException: 404 会话/分镜不存在；409 已有任务在跑。
    """
    snap = STORE.get(sid)
    if snap is None:
        raise HTTPException(status_code=404, detail="no such session")
    slug = snap.get("slug", "")
    if not slug or not (OUT_DIR / "storyboards" / f"{slug}.json").is_file():
        raise HTTPException(status_code=404, detail="no storyboard — run anew")
    global _busy
    if _busy:
        raise HTTPException(status_code=409, detail="another comic is already generating")
    _busy = True

    return _sse(_short_resume_worker(sid, slug))