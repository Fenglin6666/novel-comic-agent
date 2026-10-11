"""字节方舟垫图调用形状的回归测试：image 参数必须走 extra_body 透传。

背景：OpenAI SDK 的 typed 方法（images.generate）不收 image 关键字——
直接传会在本地 TypeError（真实跑一轮才暴露的 bug）。
"""

from __future__ import annotations

import base64
from pathlib import Path


from novel_comic_agent.imagegen import OpenAIImageGen

ARK = "https://ark.cn-beijing.volces.com/api/v3"


class _Item:
    b64_json = base64.b64encode(b"png-bytes").decode()
    url = None


class _Resp:
    data = [_Item()]


def run(coro):
    return asyncio_run(coro)


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


def test_ark_ref_images_go_through_extra_body(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class FakeImages:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class FakeClient:
        def __init__(self, *, base_url=None, api_key=None):
            self.base_url = base_url
            self.images = FakeImages()

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)

    gen = OpenAIImageGen(model="doubao-seedream-5-0-flash-260915",
                         base_url=ARK, api_key="k", watermark=False)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"ref-bytes")
    out = tmp_path / "out.png"

    run(gen.generate("提示词", out_path=out, size="1024x1024", ref_images=[ref]))

    # 旧 bug：image 作为顶层 kwarg 传给 images.generate → 本地 TypeError
    assert "image" not in captured
    body = captured.get("extra_body") or {}
    assert len(body["image"]) == 1
    assert body["image"][0].startswith("data:image/png;base64,")
    assert body["watermark"] is False
    assert captured["model"] == "doubao-seedream-5-0-flash-260915"
    assert out.read_bytes() == b"png-bytes"


def test_non_ark_ref_images_use_edit_endpoint(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class FakeImages:
        async def generate(self, **kwargs):
            captured["generate"] = kwargs
            return _Resp()

        async def edit(self, **kwargs):
            captured["edit"] = kwargs
            return _Resp()

    class FakeClient:
        def __init__(self, *, base_url=None, api_key=None):
            self.base_url = base_url
            self.images = FakeImages()

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)

    gen = OpenAIImageGen(model="gpt-image-2",
                         base_url="https://api.openai.com/v1", api_key="k")
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"ref-bytes")
    out = tmp_path / "out.png"

    run(gen.generate("提示词", out_path=out, size="1024x1024", ref_images=[ref]))

    assert "generate" not in captured and "edit" in captured
    files = captured["edit"]["image"]
    assert files[0][0] == "ref.png" and files[0][1] == b"ref-bytes"


def test_plain_generate_no_extra_image(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class FakeImages:
        async def generate(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class FakeClient:
        def __init__(self, *, base_url=None, api_key=None):
            self.base_url = base_url
            self.images = FakeImages()

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)

    gen = OpenAIImageGen(model="gpt-image-2",
                         base_url="https://api.openai.com/v1", api_key="k")
    run(gen.generate("提示词", out_path=tmp_path / "out.png", size="1024x1024"))
    assert "extra_body" not in captured or "image" not in captured.get("extra_body", {})
