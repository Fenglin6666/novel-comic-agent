"""Web 端模型配置（webconfig.py）的离线测试：环境变量替身 + 临时工作区。"""

from __future__ import annotations

from pathlib import Path

from minimuse.webconfig import apply_config, config_path, load_config, public_view, save_config


def _env() -> dict[str, str]:
    return {}


def test_save_applies_env_and_persists(tmp_path: Path):
    env = _env()
    save_config(tmp_path, {"api_key": "sk-test-1234", "model": "deepseek-chat"},
                {"api_key": "ark-key-5678", "base_url": "https://ark.cn-beijing.volces.com/api/v3"},
                environ=env)
    # 非空字段写入对应环境变量
    assert env["OPENAI_API_KEY"] == "sk-test-1234"
    assert env["MINIMUSE_MODEL"] == "deepseek-chat"
    assert env["IMAGE_API_KEY"] == "ark-key-5678"
    assert env["IMAGE_BASE_URL"] == "https://ark.cn-beijing.volces.com/api/v3"
    # 走字节方舟 → 自动关水印（同 run-ark.ps1 的约定）
    assert env["IMAGE_WATERMARK"] == "false"
    # 落盘且能读回
    assert load_config(tmp_path)["text"]["api_key"] == "sk-test-1234"
    assert config_path(tmp_path).is_file()


def test_save_merges_and_empty_keeps_old(tmp_path: Path):
    env = _env()
    save_config(tmp_path, {"api_key": "sk-old", "model": "deepseek-chat"},
                {"api_key": "ark-old"}, environ=env)
    env2 = _env()
    save_config(tmp_path, {"api_key": "sk-new"}, None, environ=env2)
    cfg = load_config(tmp_path)
    assert cfg["text"]["api_key"] == "sk-new"          # 新值覆盖
    assert cfg["text"]["model"] == "deepseek-chat"     # 未提供的字段保留
    assert cfg["image"]["api_key"] == "ark-old"        # image 未提交 → 原样
    assert env2["OPENAI_API_KEY"] == "sk-new"
    assert "IMAGE_API_KEY" not in env2                 # 空 patch 不动环境


def test_apply_config_restores_into_fresh_env(tmp_path: Path):
    save_config(tmp_path, {"api_key": "sk-x", "base_url": "https://api.deepseek.com/v1",
                           "model": "deepseek-chat"},
                {"api_key": "ark-x", "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                 "model": "doubao-seedream-5-0-flash-260915"}, environ=_env())
    env = _env()
    apply_config(tmp_path, environ=env)
    assert env["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"
    assert env["IMAGE_MODEL"] == "doubao-seedream-5-0-flash-260915"
    assert env["IMAGE_WATERMARK"] == "false"


def test_public_view_never_returns_plaintext_key(tmp_path: Path):
    save_config(tmp_path, {"api_key": "sk-secret-9999-abcd"}, None, environ=_env())
    view = public_view(tmp_path, environ=_env())
    assert view["text"]["has_key"] is True
    assert view["text"]["key_hint"].endswith("abcd")
    assert "••••" in view["text"]["key_hint"]
    plain = str(view)
    assert "sk-secret-9999" not in plain  # 明文绝不出现


def test_bad_or_missing_file_returns_skeleton(tmp_path: Path):
    assert load_config(tmp_path) == {"text": {}, "image": {}}
    config_path(tmp_path).write_text("{broken", "utf-8")
    assert load_config(tmp_path) == {"text": {}, "image": {}}
    view = public_view(tmp_path, environ={"OPENAI_API_KEY": "sk-env-8888-ffff"})
    assert view["text"]["key_hint"].endswith("ffff")  # 环境变量里的密钥也只给掩码
