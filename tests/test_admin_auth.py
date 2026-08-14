# -*- coding: utf-8 -*-
"""管理端鉴权测试：密码校验 + admin token 签发/验签/过期"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.api import admin_auth


def test_verify_password_plain():
    """明文密码：正确通过、错误拒绝、恒定时间比较"""
    assert admin_auth.verify_password("admin123") is True
    assert admin_auth.verify_password("wrong") is False


def test_verify_password_pbkdf2(monkeypatch):
    """pbkdf2 哈希：正确密码通过、错误拒绝"""
    h = admin_auth.hash_password("s3cret!")
    assert h.startswith("pbkdf2$")
    monkeypatch.setenv("ADMIN_PASSWORD", h)
    assert admin_auth.verify_password("s3cret!") is True
    assert admin_auth.verify_password("wrong") is False


def test_verify_password_disabled(monkeypatch):
    """未配置密码时不放行"""
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(admin_auth, "_DEV_PASSWORD", "")
    assert admin_auth.verify_password("admin123") is False


def test_admin_token_roundtrip():
    """签发的 admin token 能验签回用户名"""
    token = admin_auth.issue_admin_token("admin", ttl_seconds=3600)
    assert admin_auth.verify_admin_token(token) == "admin"


def test_admin_token_rejects_tamper():
    """篡改签名/前缀/用户名都验不过"""
    token = admin_auth.issue_admin_token("admin", ttl_seconds=3600)
    _, exp, sig = token.split(".")
    assert admin_auth.verify_admin_token(f"user.{exp}.{sig}") is None
    assert admin_auth.verify_admin_token(f"admin.{exp}.deadbeef") is None
    assert admin_auth.verify_admin_token("not-a-token") is None
    assert admin_auth.verify_admin_token("") is None


def test_admin_token_expired():
    """过期 token 拒绝"""
    token = admin_auth.issue_admin_token("admin", ttl_seconds=-1)
    assert admin_auth.verify_admin_token(token) is None


def test_admin_token_secret_isolation(monkeypatch):
    """换 secret 后旧 token 失效（生产必须配置 ADMIN_TOKEN_SECRET）"""
    token = admin_auth.issue_admin_token("admin", ttl_seconds=3600)
    monkeypatch.setenv("ADMIN_TOKEN_SECRET", "another-secret")
    assert admin_auth.verify_admin_token(token) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
