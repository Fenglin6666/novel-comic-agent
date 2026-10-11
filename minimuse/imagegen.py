"""图片生成抽象层：文本模型负责"想"，这一层负责"画"。

对应 nanoMuse 的多 provider 思路：BaseImageGen 之下可以换任何
OpenAI 兼容的图片端点（gpt-image、CogView、通义、硅基流动的 Kolors……）。
"""

from __future__ import annotations

import base64
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from PIL import ImageFont


class BaseImageGen(ABC):
    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        out_path: Path,
        size: str = "1024x1024",
        ref_images: list[Path] | None = None,
    ) -> Path:
        """把提示词画成一张图写到 out_path。失败抛异常，由工具层变成失败结果。"""


class OpenAIImageGen(BaseImageGen):
    """任何 OpenAI 兼容的图片端点。

    和文本 LLM 分开配置，因为它们常常不是同一家：
      IMAGE_API_KEY   密钥（缺省回落 OPENAI_API_KEY）
      IMAGE_BASE_URL  端点（缺省回落 OPENAI_BASE_URL）
      IMAGE_MODEL     模型名，如 gpt-image-2、cogview-3-flash
    """

    name = "openai-image"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        watermark: bool | None = None,
    ) -> None:
        """创建图片客户端；显式参数优先，其次 IMAGE_*，最后回落 OPENAI_*。

        Args:
            model: 图片模型名，缺省读 IMAGE_MODEL（再缺省 gpt-image-2）。
            base_url: 端点，缺省读 IMAGE_BASE_URL / OPENAI_BASE_URL。
            api_key: 密钥，缺省读 IMAGE_API_KEY / OPENAI_API_KEY。
            watermark: 水印开关（字节 Seedream 等支持）；None=不传该参数，
                缺省读环境变量 IMAGE_WATERMARK（如 "false" 表示关水印）。
        """
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=base_url
            or os.environ.get("IMAGE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or None,
            api_key=api_key
            or os.environ.get("IMAGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model or os.environ.get("IMAGE_MODEL", "gpt-image-2")
        wm = watermark
        if wm is None:
            env = os.environ.get("IMAGE_WATERMARK")
            if env is not None:
                wm = env.strip().lower() in ("1", "true", "yes", "on")
        self.watermark = wm

    async def generate(
        self,
        prompt: str,
        *,
        out_path: Path,
        size: str = "1024x1024",
        ref_images: list[Path] | None = None,
    ) -> Path:
        """按提示词生成一张图并写到 out_path。

        Args:
            prompt: 完整出图提示词（风格卡等已由工具层拼好）。
            out_path: 目标 PNG 路径（父目录会自动创建）。
            size: "宽x高" 像素档位。
            ref_images: 可选参考图（垫图）路径；None=纯文生图。
        Returns:
            写好的文件路径（即 out_path）。
        Raises:
            RuntimeError: 端点既没返回 b64_json 也没返回 url。
        """
        kwargs: dict[str, Any] = {"model": self.model, "prompt": prompt, "size": size, "n": 1}
        if self.watermark is not None:
            kwargs["extra_body"] = {"watermark": self.watermark}
        if ref_images:
            # 垫图：字节 Seedream 在 generate 上吃 image 参数（base64 数据 URL）；
            # OpenAI gpt-image 走 edit 端点（image 收文件元组）。按端点分派。
            is_ark = "ark" in str(getattr(self.client, "base_url", "")).lower()
            if is_ark or "seedream" in self.model.lower():
                kwargs["image"] = [_data_url(p) for p in ref_images]
                resp = await self.client.images.generate(**kwargs)
            else:
                files = [(p.name, p.read_bytes(), "image/png") for p in ref_images]
                edit_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "image": files,
                    "prompt": prompt,
                    "n": 1,
                }
                if self.watermark is not None:
                    edit_kwargs["extra_body"] = {"watermark": self.watermark}
                resp = await self.client.images.edit(**edit_kwargs)
        else:
            resp = await self.client.images.generate(**kwargs)
        item = resp.data[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        b64 = getattr(item, "b64_json", None)
        if b64:
            out_path.write_bytes(base64.b64decode(b64))
            return out_path
        url = getattr(item, "url", None)
        if not url:
            raise RuntimeError(f"{self.model} returned neither b64_json nor url")
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.get(url)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        return out_path


def _data_url(path: Path) -> str:
    """读图片文件转 data URL（Seedream 垫图参数格式）。"""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------- 文字排版公用件
# 合成页要画中文字（标题/旁白/气泡），字体查找与像素级折行放这里共用。

_FONT_PATHS = [
    r"C:\Windows\Fonts\msyh.ttc",  # Windows 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux CI
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载一个支持中文的字体（微软雅黑优先，跨平台回退）。

    Args:
        size: 字号（像素）。
    Returns:
        PIL 字体对象；找不到系统字体时退回 Pillow 默认字体。
    """
    for p in _FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # 旧 Pillow 的 load_default 不带字号
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: float) -> list[str]:
    """按像素宽度折行：ASCII 尽量按词断，CJK 逐字断。

    Args:
        text: 原文（可含换行，换行处强制分段）。
        font: 测宽用的字体（必须与绘制时同一字体）。
        max_width: 单行最大像素宽度。
    Returns:
        折行后的行列表。
    """
    lines: list[str] = []
    for para in text.split("\n"):
        units: list[str] = []
        for word in re.split(r"\s+", para.strip()):
            buf = ""
            for ch in word:
                if ord(ch) > 0x2E80:  # CJK 与全角字符自成一段
                    if buf:
                        units.append(buf)
                        buf = ""
                    units.append(ch)
                else:
                    buf += ch
            if buf:
                units.append(buf)
        if not units:
            lines.append("")
            continue
        line = ""
        for unit in units:
            trial = f"{line} {unit}" if line else unit
            if font.getlength(trial) <= max_width or not line:
                line = trial
            else:
                lines.append(line)
                line = unit
        lines.append(line)
    return lines


