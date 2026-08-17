# -*- coding: utf-8 -*-
"""用户偏好存储（JSONL 文件，只追加 + 状态标记）

设计：
- 每用户一个文件：data/profiles/{user_id}.prefs.jsonl（user_id 由服务端签发，文件名安全）
- 只追加不合并：保留完整时间线，注入时交由 LLM 综合判断矛盾并澄清
- 每条带 id/status：删除 = 追加 deleted 标记；修改 = 追加新条目并 supersedes 旧 id
- 读取时计算「当前生效」（过滤 deleted / 被 supersedes 取代的条目）
"""

import json
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional


PROFILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "profiles",
)

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(user_id: str) -> threading.Lock:
    with _locks_guard:
        if user_id not in _locks:
            _locks[user_id] = threading.Lock()
        return _locks[user_id]


def _path(user_id: str) -> str:
    return os.path.join(PROFILES_DIR, f"{user_id}.prefs.jsonl")


def append_pref(
    user_id: str,
    text: str,
    source: str = "manual",
    supersedes: Optional[str] = None,
    delete_of: Optional[str] = None,
) -> Dict:
    """追加一条偏好（source: extraction / manual / clarification / manual_edit）"""
    entry = {
        "id": uuid.uuid4().hex[:12],
        "text": text,
        "updated_at": datetime.now().isoformat(),
        "source": source,
        "status": "active",
    }
    if supersedes:
        entry["supersedes"] = supersedes
    if delete_of:
        entry["status"] = "deleted"
        entry["delete_of"] = delete_of
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with _lock_for(user_id):
        with open(_path(user_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def get_timeline(user_id: str) -> List[Dict]:
    """全部历史条目（按写入顺序，即时间顺序）"""
    path = _path(user_id)
    if not os.path.exists(path):
        return []
    entries: List[Dict] = []
    with _lock_for(user_id):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def get_active(user_id: str, limit: Optional[int] = None) -> List[Dict]:
    """当前生效偏好（过滤 deleted / 被 supersedes 取代的），按时间排序"""
    timeline = get_timeline(user_id)
    superseded = {
        e["supersedes"] for e in timeline if e.get("supersedes")
    }
    deleted = {
        e["delete_of"] for e in timeline if e.get("status") == "deleted"
    }
    active = [
        e for e in timeline
        if e.get("status") == "active"
        and e["id"] not in superseded
        and e["id"] not in deleted
    ]
    if limit:
        active = active[-limit:]
    return active


def delete_pref(user_id: str, pref_id: str) -> bool:
    """软删除：追加一条 deleted 标记（保留历史）"""
    if not any(e["id"] == pref_id for e in get_timeline(user_id)):
        return False
    append_pref(user_id, "", source="manual", delete_of=pref_id)
    return True


def update_pref(user_id: str, pref_id: str, new_text: str) -> bool:
    """修改：追加新条目并 supersedes 旧条目（旧条目保留可追溯）"""
    text = (new_text or "").strip()
    if not text:
        return False
    if not any(e["id"] == pref_id for e in get_timeline(user_id)):
        return False
    append_pref(user_id, text, source="manual_edit", supersedes=pref_id)
    return True
