# -*- coding: utf-8 -*-
"""演进式摘要 JSONL 存储（时间线语义，与用户偏好统一）

摘要的本质是"按用户的时间线"：每版追加一行、保留完整历史、可人工纠错，
与偏好（pref_store）同一存储哲学——时间线类数据用文件，语义召回类
（历史对话）才用向量库（Chroma）。

文件：data/profiles/{user_id}.summary.jsonl，只追加；超 keep_recent 时
重写保留最近 N 行（低频，可接受）。读最新 = 取文件最后一行。
"""

import json
import os
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional


SUMMARIES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "profiles",
)

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(user_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(user_id)
        if lock is None:
            lock = threading.Lock()
            _locks[user_id] = lock
        return lock


def _path(user_id: str) -> str:
    return os.path.join(SUMMARIES_DIR, f"{user_id}.summary.jsonl")


def add_summary(
    user_id: str,
    summary: str,
    key_points: str = "",
    keep_recent: int = 20,
) -> bool:
    """追加一版摘要（时间线）；超 keep_recent 时重写保留最近 N 行"""
    if not user_id or not summary:
        return False
    entry = {
        "id": uuid.uuid4().hex[:12],
        "summary": summary,
        "key_points": key_points,
        "created_at": datetime.now().isoformat(),
    }
    with _lock_for(user_id):
        os.makedirs(SUMMARIES_DIR, exist_ok=True)
        try:
            with open(_path(user_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _trim_if_needed(user_id, keep_recent)
            return True
        except OSError as e:
            print(f"保存摘要失败: {e}")
            return False


def _trim_if_needed(user_id: str, keep_recent: int) -> None:
    """文件行数超过 keep_recent 时重写保留最近 N 行（持锁调用）"""
    path = _path(user_id)
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        if len(lines) <= keep_recent:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-keep_recent:])
    except OSError as e:
        print(f"摘要清理失败: {e}")


def get_summaries(user_id: str, limit: int = 3) -> List[Dict]:
    """读取最近 N 版摘要（最新在前）。

    文件很小（每用户 ≤ keep_recent 行），全量读取可接受；
    若要严格高效，可倒序读最后 N 行，但当前量级没必要。
    """
    path = _path(user_id)
    if not os.path.exists(path):
        return []
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries[-limit:][::-1]


def get_running_summary(user_id: str) -> str:
    """取最新一版摘要"""
    summaries = get_summaries(user_id, limit=1)
    return summaries[0]["summary"] if summaries else ""


def clear(user_id: str) -> bool:
    """清空该用户的摘要时间线（管理端人工纠错）"""
    with _lock_for(user_id):
        try:
            if os.path.exists(_path(user_id)):
                os.remove(_path(user_id))
            return True
        except OSError as e:
            print(f"清空摘要失败: {e}")
            return False


def migrate_from_sqlite(user_id: str, rows: List[Dict], keep_recent: int = 20) -> int:
    """一次性迁移：把 SQLite 的历史摘要按时间顺序写入 JSONL（仅文件不存在时）"""
    path = _path(user_id)
    if os.path.exists(path) or not rows:
        return 0
    with _lock_for(user_id):
        os.makedirs(SUMMARIES_DIR, exist_ok=True)
        written = 0
        try:
            with open(path, "w", encoding="utf-8") as f:
                # rows 按 created_at 升序（最旧在前），保持时间线顺序
                for r in rows:
                    f.write(
                        json.dumps(
                            {
                                "id": uuid.uuid4().hex[:12],
                                "summary": r.get("summary", ""),
                                "key_points": r.get("key_points", ""),
                                "created_at": r.get("created_at", ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
            _trim_if_needed(user_id, keep_recent)
            return written
        except OSError as e:
            print(f"摘要迁移失败: {e}")
            return 0
