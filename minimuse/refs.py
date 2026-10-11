"""角色参考图（垫图）：上传 → 清单 → 注入剧本 → 逐格垫图。

目录结构（都在工作区内）：
    refs/refs.json        清单 {角色名: {"file": "x.png", "desc": "一句话外貌"}}
    refs/<角色名>.png     参考图（统一重存为 PNG，丢弃 EXIF 等噪音）

两个作用点：
    1. save_script 前——refs_prompt() 注入"已提供这些角色，必须使用"；
    2. generate_panel 时——按该格 cast 匹配参考图，传给画师做垫图
       （人物一致性的第二道保险，第一道是文字外貌卡）。
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

MAX_REF_BYTES = 8 * 1024 * 1024  # 8MB，足够手机照片
MAX_REF_SIDE = 2048  # 垫图不需要 4K；超大图等比缩小，省上传与推理时间
_SAFE_NAME = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff ·-]{0,30}$")


def refs_dir(workspace: Path) -> Path:
    """参考图目录（workspace/refs）。"""
    return workspace / "refs"


def _manifest_path(workspace: Path) -> Path:
    """清单文件路径（workspace/refs/refs.json）。"""
    return refs_dir(workspace) / "refs.json"


def load_refs(workspace: Path) -> dict[str, dict[str, str]]:
    """读参考图清单。

    Args:
        workspace: 工作区。
    Returns:
        {角色名: {"file": 文件名, "desc": 描述}}；无清单返回空 dict。
    """
    try:
        data = json.loads(_manifest_path(workspace).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(workspace: Path, refs: dict[str, dict[str, str]]) -> None:
    """原子写清单（临时文件 + 替换）。"""
    d = refs_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    tmp = _manifest_path(workspace).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(refs, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(_manifest_path(workspace))


def _normalize_image(raw: bytes, out: Path) -> tuple[int, int]:
    """校验并重存参考图：转 RGB PNG，最长边压到 MAX_REF_SIDE。

    Args:
        raw: 上传的原始图片字节。
        out: 目标 PNG 路径。
    Returns:
        (宽, 高)。
    Raises:
        ValueError: 不是可解码的图片。
    """
    import io

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 — 解码失败统一变 400
        raise ValueError(f"无法识别的图片文件：{exc}") from exc
    img = img.convert("RGB")
    if max(img.size) > MAX_REF_SIDE:
        scale = MAX_REF_SIDE / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return img.size


def save_ref(workspace: Path, name: str, raw: bytes, desc: str = "") -> dict[str, Any]:
    """保存/覆盖一个角色参考图，并更新清单。

    Args:
        workspace: 工作区。
        name: 角色名（中文/英文/数字，≤31 字符；即清单键，也是 cast 匹配键）。
        raw: 图片字节（jpg/png/webp…PIL 能读即可）。
        desc: 一句话外貌描述（可空；空则等视觉模型补或用户后补）。
    Returns:
        {"name", "file", "desc", "width", "height"}。
    Raises:
        ValueError: 名字非法 / 图片过大 / 不是图片。
    """
    name = (name or "").strip()
    if not _SAFE_NAME.match(name):
        raise ValueError("角色名需要 1-31 个字符（中文/字母/数字/空格/·/-）")
    if len(raw) > MAX_REF_BYTES:
        raise ValueError(f"图片太大（上限 {MAX_REF_BYTES // 1024 // 1024}MB）")
    fname = re.sub(r"[^\w\u4e00-\u9fff-]", "_", name) + ".png"
    w, h = _normalize_image(raw, refs_dir(workspace) / fname)
    refs = load_refs(workspace)
    refs[name] = {"file": fname, "desc": (desc or "").strip()}
    _save_manifest(workspace, refs)
    return {"name": name, "file": fname, "desc": refs[name]["desc"], "width": w, "height": h}


def update_desc(workspace: Path, name: str, desc: str) -> None:
    """只改描述（视觉模型自动补描述后回填用）。"""
    refs = load_refs(workspace)
    if name in refs:
        refs[name]["desc"] = desc.strip()
        _save_manifest(workspace, refs)


def delete_ref(workspace: Path, name: str) -> bool:
    """删除一个参考图（清单 + 文件）。返回是否存在过。"""
    refs = load_refs(workspace)
    entry = refs.pop(name, None)
    if entry is None:
        return False
    _save_manifest(workspace, refs)
    f = refs_dir(workspace) / entry["file"]
    if f.is_file():
        f.unlink()
    return True


def ref_files_for(workspace: Path, cast: list[str]) -> list[Path]:
    """按这一格的 cast 匹配参考图文件（名字精确匹配；同名去重）。

    Args:
        workspace: 工作区。
        cast: 该格出场角色名列表。
    Returns:
        命中的参考图路径列表（最多 4 张——再多画师也吃不下）。
    """
    refs = load_refs(workspace)
    out: list[Path] = []
    for name in cast:
        entry = refs.get(name) or refs.get(name.strip())
        if not entry:
            continue
        f = refs_dir(workspace) / entry["file"]
        if f.is_file() and f not in out:
            out.append(f)
    return out[:4]


def refs_prompt(workspace: Path) -> str:
    """把参考图清单变成注入给 save_script 的指令块。

    Returns:
        指令块文本；无参考图时为空串（零成本）。
    """
    refs = load_refs(workspace)
    if not refs:
        return ""
    lines = ["【已上传角色参考图】以下角色必须在剧本中出现，characters 必须包含同名角色，"
             "look 卡要与描述一致（外貌以参考图为准）："]
    for name, e in refs.items():
        desc = e.get("desc") or "外貌以参考图为准"
        lines.append(f"- {name}：{desc}")
    return "\n".join(lines)


def decode_data_url(data_url: str) -> bytes:
    """解 data:image/...;base64,xxx 为字节（Web 上传用）。

    Raises:
        ValueError: 格式不是 base64 data URL。
    """
    m = re.match(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.*)$", data_url or "", re.S)
    if not m:
        raise ValueError("需要 data:image/...;base64,... 格式的图片")
    try:
        return base64.b64decode(m.group(1), validate=True)
    except binascii.Error as exc:
        raise ValueError("base64 解码失败") from exc
