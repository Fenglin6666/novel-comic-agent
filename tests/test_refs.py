"""角色参考图（垫图）的离线测试。"""

from __future__ import annotations

import io

from PIL import Image

from minimuse.refs import (
    decode_data_url,
    delete_ref,
    load_refs,
    ref_files_for,
    refs_prompt,
    save_ref,
)


def _png_bytes(w=64, h=64) -> bytes:
    img = Image.new("RGB", (w, h), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_save_load_match_and_prompt(tmp_path):
    """保存 → 清单 → cast 匹配 → 剧本注入块的完整链路。"""
    e = save_ref(tmp_path, "小满", _png_bytes(), desc="红裙短发女孩")
    assert e["name"] == "小满" and e["desc"] == "红裙短发女孩"
    assert load_refs(tmp_path)["小满"]["file"].endswith(".png")

    hits = ref_files_for(tmp_path, ["小满", "路人甲"])  # 路人无参考图，跳过
    assert len(hits) == 1 and hits[0].is_file()

    block = refs_prompt(tmp_path)
    assert "小满" in block and "红裙短发女孩" in block
    assert refs_prompt(tmp_path / "empty-ws") == ""  # 无参考图=零注入


def test_desc_autofill_placeholder_and_delete(tmp_path):
    """无描述时清单保留空串；删除同时清掉清单与文件。"""
    save_ref(tmp_path, "阿七", _png_bytes())
    assert load_refs(tmp_path)["阿七"]["desc"] == ""
    assert "外貌以参考图为准" in refs_prompt(tmp_path)  # 注入层的兜底话术

    assert delete_ref(tmp_path, "阿七") is True
    assert delete_ref(tmp_path, "阿七") is False
    assert load_refs(tmp_path) == {}
    assert not list((tmp_path / "refs").glob("*.png"))


def test_decode_data_url_and_bad_input(tmp_path):
    import base64
    import pytest

    b64 = base64.b64encode(_png_bytes(10, 10)).decode()
    assert decode_data_url(f"data:image/png;base64,{b64}")[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        decode_data_url("not-a-data-url")
    with pytest.raises(ValueError, match="角色名"):
        save_ref(tmp_path, "", b64)  # 名字非法
