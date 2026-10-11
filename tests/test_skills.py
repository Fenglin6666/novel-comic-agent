"""画师 skill（索引层 + 正文注入 + 模型自选）的离线测试。"""

from __future__ import annotations

import pytest

from minimuse import skills


def _write(path, text):
    path.write_text(text.strip() + "\n", "utf-8")


def test_upload_without_frontmatter(tmp_path, monkeypatch):
    """用户上传的裸 md（无 frontmatter）也能用：标题/简介自动回落。"""
    monkeypatch.setattr(skills, "_DEFAULT_DIR", tmp_path)
    assert skills.list_skills() == []

    saved = skills.save_skill("my-style.md", "# 我的风格\n\n核心词：粗墨线、留白")
    assert saved == {"id": "my-style", "title": "我的风格"}

    items = skills.list_skills()  # 索引层字段
    assert [i["id"] for i in items] == ["my-style"]
    assert items[0]["description"] == "核心词：粗墨线、留白"
    assert "视觉基因" not in str(items)  # 索引层绝不带正文

    prompt = skills.apply_skill("BASE", "my-style")
    assert prompt.startswith("BASE")  # 拼在基础提示词后
    assert "我的风格" in prompt and "粗墨线" in prompt  # 正文注入
    assert "画师风格 Skill" in prompt  # 指令头注入
    assert "视觉基因" in prompt  # 注入头要求模型按此提炼

    assert skills.apply_skill("BASE", None) == "BASE"  # 不选=不改
    assert skills.apply_skill("BASE", "") == "BASE"


def test_frontmatter_index_and_injection(tmp_path, monkeypatch):
    """带 frontmatter 的画师卡：索引层只出 title/description，正文选中才注入。"""
    monkeypatch.setattr(skills, "_DEFAULT_DIR", tmp_path)
    _write(
        tmp_path / "fake-master.md",
        """
        ---
        title: 假想大师
        description: 测试画师，适合哲理短篇。
        ---

        # 假想大师（虚构）

        ## 视觉基因
        - 线条：粗墨线
        - 构图：大量留白

        ## 避开
        糖果色
        """,
    )

    items = skills.list_skills()
    assert items == [
        {
            "id": "fake-master",
            "title": "假想大师",
            "description": "测试画师，适合哲理短篇。",
        }
    ]

    index = skills.skill_index_text()  # 选择器看到的索引
    assert "fake-master" in index and "测试画师" in index
    assert "粗墨线" not in index  # 正文细节不进索引

    block = skills.load_skill("fake-master")
    assert "视觉基因" in block and "假想大师（虚构）" in block  # 正文层
    assert "save_script 的 style 字段必须提炼这张卡" in block  # 注入头约束


def test_skill_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "_DEFAULT_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown style skill"):
        skills.apply_skill("BASE", "nope")
    with pytest.raises(ValueError, match="\\.md"):
        skills.save_skill("bad.txt", "x")
    with pytest.raises(ValueError, match="为空"):
        skills.save_skill("empty.md", "   ")


def test_selection_messages_and_parse(tmp_path, monkeypatch):
    """选择器：索引进上下文；回复解析容忍围栏与废话，拒绝非法 id。"""
    monkeypatch.setattr(skills, "_DEFAULT_DIR", tmp_path)
    _write(
        tmp_path / "fake-master.md",
        """
        ---
        title: 假想大师
        description: 测试画师，适合哲理短篇。
        ---

        # 假想大师

        ## 视觉基因
        - 线条：粗墨线
        """,
    )

    msgs = skills.selection_messages("一个关于时间的故事")
    assert "画师选择器" in msgs[0].content
    assert "一个关于时间的故事" in msgs[-1].content
    assert "fake-master" in msgs[-1].content  # 索引在 user 消息里

    valid = [s["id"] for s in skills.list_skills()]
    fenced = '```json\n{"skill": "fake-master", "reason": "气质契合"}\n```'
    assert skills.parse_selection(fenced, valid) == {"id": "fake-master", "reason": "气质契合"}
    chatty = '我认为是 {"skill": "fake-master", "reason": ""} 就这样'
    assert skills.parse_selection(chatty, valid)["reason"]
    assert skills.parse_selection("完全没有 JSON", valid) is None
    assert skills.parse_selection('{"skill": "not-exist", "reason": "x"}', valid) is None
