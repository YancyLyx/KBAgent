# -*- coding: utf-8 -*-
"""匿名身份 token（阶段 0：user_id 由服务端签发，客户端不能自由伪造）

完整登录鉴权（阶段 1）之前的过渡方案：
- 服务端签发 HMAC 签名 token：`{user_id}.{hmac_sha256_hex}`
- 客户端在后续请求的 `Authorization: Bearer <token>` 中携带
- 服务端验签得到 user_id；不知道签名密钥的客户端无法伪造任意 user_id
- 生产环境必须设置环境变量 `ANON_TOKEN_SECRET`，否则使用开发密钥（仅限本地演示）
"""

import hashlib
import hmac
import os
import uuid
from typing import Optional


_DEV_SECRET = "kbagent-dev-anon-secret-change-me"


def _secret() -> str:
    return os.getenv("ANON_TOKEN_SECRET", _DEV_SECRET)


def issue_anonymous_user_id() -> str:
    """签发一个新的匿名 user_id（服务端生成，客户端不可自选）"""
    return f"user_{uuid.uuid4().hex[:12]}"


def issue_token(user_id: str) -> str:
    """为 user_id 签发签名 token"""
    sig = hmac.new(
        _secret().encode(), user_id.encode(), hashlib.sha256
    ).hexdigest()
    return f"{user_id}.{sig}"


def verify_token(token: str) -> Optional[str]:
    """验签 token，返回 user_id；无效返回 None"""
    if not token:
        return None
    try:
        user_id, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    expected = hmac.new(
        _secret().encode(), user_id.encode(), hashlib.sha256
    ).hexdigest()
    if hmac.compare_digest(sig, expected):
        return user_id
    return None
