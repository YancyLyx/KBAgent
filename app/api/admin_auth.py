# -*- coding: utf-8 -*-
"""管理端鉴权（单管理员 + 带过期 HMAC token）

- 凭据来自环境变量 ADMIN_USERNAME / ADMIN_PASSWORD
  （支持明文或 pbkdf2 哈希两种格式，生产建议用哈希，见 hash_password）
- 登录成功签发 HMAC 签名 token：`admin.{exp_ts}.{hmac(username:exp_ts)}`
- 管理接口通过 `Authorization: Bearer <token>` 验签，过期自动失效
- 生产必须设置 ADMIN_TOKEN_SECRET，否则使用开发密钥（仅限本地演示）
"""

import hashlib
import hmac
import os
import time
from typing import Optional


_DEV_SECRET = "kbagent-admin-dev-secret-change-me"
_DEV_PASSWORD = "admin123"  # 仅当 .env 未配置 ADMIN_PASSWORD 时的本地开发默认


def _secret() -> str:
    return os.getenv("ADMIN_TOKEN_SECRET", _DEV_SECRET)


def get_admin_username() -> str:
    return os.getenv("ADMIN_USERNAME", "admin")


def _admin_password() -> Optional[str]:
    return os.getenv("ADMIN_PASSWORD") or _DEV_PASSWORD


def verify_password(password: str) -> bool:
    """校验密码：支持 pbkdf2 哈希（`pbkdf2$iter$salt_hex$hash_hex`）或明文。

    两种都做恒定时间比较，防止时序侧信道。
    """
    stored = _admin_password()
    if not stored:
        return False
    if stored.startswith("pbkdf2$"):
        try:
            _, iterations, salt_hex, hash_hex = stored.split("$", 3)
            derived = hashlib.pbkdf2_hmac(
                "sha256",
                (password or "").encode(),
                bytes.fromhex(salt_hex),
                int(iterations),
            ).hex()
            return hmac.compare_digest(derived, hash_hex)
        except (ValueError, TypeError):
            return False
    return hmac.compare_digest(password or "", stored)


def hash_password(password: str, iterations: int = 100_000) -> str:
    """生成 pbkdf2 哈希（用于初始化 ADMIN_PASSWORD 配置）"""
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, iterations
    )
    return f"pbkdf2${iterations}${salt.hex()}${derived.hex()}"


def issue_admin_token(username: str, ttl_seconds: int = 24 * 3600) -> str:
    """签发带过期时间的 admin token"""
    exp = int(time.time()) + ttl_seconds
    payload = f"{username}:{exp}"
    sig = hmac.new(_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"admin.{exp}.{sig}"


def verify_admin_token(token: str) -> Optional[str]:
    """验签并检查过期，返回 username；无效/过期返回 None"""
    if not token:
        return None
    try:
        prefix, exp_str, sig = token.split(".", 2)
    except ValueError:
        return None
    if prefix != "admin":
        return None
    try:
        exp = int(exp_str)
    except ValueError:
        return None
    if time.time() > exp:
        return None
    payload = f"{get_admin_username()}:{exp}"
    expected = hmac.new(
        _secret().encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    if hmac.compare_digest(sig, expected):
        return get_admin_username()
    return None
