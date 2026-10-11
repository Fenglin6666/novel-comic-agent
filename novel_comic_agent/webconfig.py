"""Web 端模型配置：前端可直接改 文本/生图 的密钥、端点、模型名。

配置落地 comic_out/web_config.json（.gitignore 已覆盖 comic_out/，不会进 git）。
服务启动时加载进环境变量——OpenAILLM / OpenAIImageGen 每次运行时新建、
各自从环境读配置，所以保存后下一次生成即生效，无需重启。
安全边界：GET 接口永不回传明文密钥（只给尾号掩码）；明文只落在本机文件里。
"""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

# 两组模型各自的环境变量名（与 CLI 的 run-*.ps1 完全同名，互相兼容）
ENV_MAP = {
    "text": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_BASE_URL",
             "model": "MINIMUSE_MODEL"},
    "image": {"api_key": "IMAGE_API_KEY", "base_url": "IMAGE_BASE_URL",
              "model": "IMAGE_MODEL"},
}
FIELDS = ("api_key", "base_url", "model")


def config_path(workspace: Path) -> Path:
    """配置文件路径（workspace 下，随工作区走）。"""
    return Path(workspace) / "web_config.json"


def load_config(workspace: Path) -> dict[str, dict[str, str]]:
    """读配置文件；坏 JSON/不存在返回空骨架（不抛，密钥文件坏了别拦启动）。

    每组额外带 key_for：该密钥是为哪个端点保存的——换服务商未换密钥时
    前端能精确提示"当前密钥属于旧服务商"。
    """
    try:
        data = json.loads(config_path(workspace).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {g: {} for g in ENV_MAP}
    out: dict[str, dict[str, str]] = {}
    for group in ENV_MAP:
        raw = data.get(group) if isinstance(data.get(group), dict) else {}
        keep = (*FIELDS, "key_for")
        out[group] = {k: str(raw[k]) for k in keep if k in raw and raw[k]}
    return out


def apply_config(workspace: Path, environ: MutableMapping[str, str] = os.environ) -> None:
    """启动时把文件里的配置写进环境变量（只写非空项，不覆盖脚本给的空缺）。"""
    for group, fields in load_config(workspace).items():
        for field, env_name in ENV_MAP[group].items():
            if fields.get(field):
                environ[env_name] = fields[field]
        # 字节 Seedream 的水印会毁掉漫画分格；走 Ark 生图时默认关（同 run-ark.ps1）
        if group == "image" and "ark" in fields.get("base_url", "").lower():
            environ.setdefault("IMAGE_WATERMARK", "false")


def save_config(
    workspace: Path,
    text: dict[str, str] | None,
    image: dict[str, str] | None,
    environ: MutableMapping[str, str] = os.environ,
) -> dict[str, dict[str, str]]:
    """合并保存前端提交的配置：非空字段才生效（空 = 保持现状）。

    Args:
        text/image: {"api_key"?, "base_url"?, "model"?}，值需为非空字符串。
        environ: 写环境变量（测试时可传替身）。
    Returns:
        保存后的完整配置（含明文，仅供内部/日志调试，不要直出给前端）。
    """
    merged = load_config(workspace)
    for group, patch in (("text", text), ("image", image)):
        if not patch:
            continue
        new_base = str(patch.get("base_url") or "").strip()
        for field in FIELDS:
            val = str(patch.get(field) or "").strip()
            if val:
                merged[group][field] = val
                environ[ENV_MAP[group][field]] = val
        # 密钥换了就记下它属于哪个端点——"换服务商没换密钥"是最常见的 401 陷阱
        if str(patch.get("api_key") or "").strip():
            merged[group]["key_for"] = new_base or merged[group].get("key_for", "")
    # 同 apply_config 的 Ark 水印处理：换到字节生图时自动关水印
    if "ark" in merged["image"].get("base_url", "").lower():
        environ.setdefault("IMAGE_WATERMARK", "false")
        merged["image"]["watermark"] = "false"
    p = config_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), "utf-8")
    return merged


def _mask(key: str) -> str:
    """密钥掩码：只露末 4 位；空返回空串。"""
    return f"••••{key[-4:]}" if len(key) >= 4 else "••••"


def public_view(workspace: Path, environ: MutableMapping[str, str] = os.environ) -> dict[str, Any]:
    """GET /api/config 的视图：端点/模型明文（本就非敏感），密钥只给尾号。"""
    saved = load_config(workspace)
    out: dict[str, Any] = {}
    for group, env_names in ENV_MAP.items():
        def env(f: str, *, _names=env_names, _g=group) -> str:
            return environ.get(_names[f], "") or saved[_g].get(f, "")

        key = env("api_key")
        out[group] = {
            "base_url": env("base_url"),
            "model": env("model"),
            "has_key": bool(key),
            "key_hint": _mask(key) if key else "",
            "key_for": saved[group].get("key_for", ""),
        }
    return out
