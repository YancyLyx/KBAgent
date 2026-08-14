# -*- coding: utf-8 -*-
"""阶段 0：服务端签发匿名身份 token 的测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.api.anon_auth import (
    issue_anonymous_user_id,
    issue_token,
    verify_token,
)


def test_anonymous_user_id_format():
    """服务端签发的匿名 uid 有固定前缀且每次不同"""
    uid1 = issue_anonymous_user_id()
    uid2 = issue_anonymous_user_id()
    assert uid1.startswith("user_")
    assert uid2.startswith("user_")
    assert uid1 != uid2


def test_token_roundtrip():
    """签发的 token 能被验证回原 user_id"""
    uid = issue_anonymous_user_id()
    token = issue_token(uid)
    assert verify_token(token) == uid


def test_token_tampered_rejected():
    """篡改 uid 或签名都验不过（客户端不能伪造任意 user_id）"""
    uid = issue_anonymous_user_id()
    token = issue_token(uid)
    real_uid, sig = token.rsplit(".", 1)

    forged_uid = "user_attacker"
    assert verify_token(f"{forged_uid}.{sig}") is None
    assert verify_token(f"{real_uid}.deadbeef") is None
    assert verify_token("not-a-token") is None
    assert verify_token("") is None


def test_token_secret_isolation(monkeypatch):
    """换 secret 后旧 token 失效（生产必须配置 ANON_TOKEN_SECRET）"""
    uid = issue_anonymous_user_id()
    token = issue_token(uid)
    monkeypatch.setenv("ANON_TOKEN_SECRET", "another-secret")
    assert verify_token(token) is None


def test_resolve_user_id_branches():
    """chat_api 身份解析各分支"""
    from app.api.chat_api import _resolve_user_id
    from app.api.anon_auth import issue_token

    # 无 token：/chat 签发新身份，其他端点返回匿名
    uid, new_token = _resolve_user_id(None, issue_if_missing=True)
    assert uid and new_token
    uid2, new_token2 = _resolve_user_id(None, issue_if_missing=False)
    assert uid2 is None and new_token2 is None

    # 带合法 token：复用身份，不签发新 token
    token = issue_token(uid)
    resolved, issued = _resolve_user_id(f"Bearer {token}", issue_if_missing=True)
    assert resolved == uid and issued is None

    # 带伪造 token：按 issue_if_missing 决定
    resolved2, issued2 = _resolve_user_id("Bearer user_x.ffff", issue_if_missing=True)
    assert resolved2.startswith("user_") and issued2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
