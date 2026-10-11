"""可选的视觉描述：给参考图自动生成一句话外貌卡。

配置（均为可选；未配置时 describe 返回空串，由用户手填描述）：
    VISION_API_KEY   密钥（如智谱免费档）
    VISION_BASE_URL  端点，如 https://open.bigmodel.cn/api/paas/v4
    VISION_MODEL     模型名，如 glm-4v-flash
"""

from __future__ import annotations

import base64
import os
from pathlib import Path


def configured() -> bool:
    """是否配置了视觉模型。"""
    return bool(os.environ.get("VISION_API_KEY") or os.environ.get("OPENAI_API_KEY"))


async def describe_image(path: Path) -> str:
    """让视觉模型用一句话描述角色外貌（年龄/发型/服装/标志特征）。

    Args:
        path: 参考图路径。
    Returns:
        ≤80 字的中文描述；未配置视觉模型或失败返回空串（不阻塞上传）。
    """
    if not (os.environ.get("VISION_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return ""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=os.environ.get("VISION_BASE_URL") or None,
            api_key=os.environ.get("VISION_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        )
        b64 = base64.b64encode(path.read_bytes()).decode()
        resp = await client.chat.completions.create(
            model=os.environ.get("VISION_MODEL", "glm-4v-flash"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "一句话描述这个角色的外貌：年龄、体型、发型发色、服装、"
                            "标志性特征。60字以内，只输出描述本身。",
                        },
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()[:80]
    except Exception:  # noqa: BLE001 — 描述失败不该挡住上传
        return ""
