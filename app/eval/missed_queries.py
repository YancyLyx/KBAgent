# -*- coding: utf-8 -*-
"""检索失败 query 日志（系统层闭环）

记录"检索不到/低置信"的 query，供高频未命中聚合分析——把失败 query 变成
知识库迭代的养料（第二篇文章：失败 query 是金矿，定期分析 → 补充知识库）。

存储：JSONL 追加写（与 request_metrics 一致，天然支持多进程追加）。
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional


MISSED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "missed_queries.jsonl",
)

# 文件行数超过此值时，重写保留最近 MAX_ENTRIES 条，防止无限膨胀
MAX_ENTRIES = 5000
# 文件超过此大小（字节）触发重写
TRIM_SIZE_BYTES = 2 * 1024 * 1024

_missed_lock = threading.Lock()


def record_missed(
    query: str,
    tag: str = "",
    reason: str = "empty",
    user_id: Optional[str] = None,
    rerank_scores: Optional[List[float]] = None,
) -> None:
    """记录一条检索失败 query

    reason: empty（空结果）/ low_score（低于相关性阈值）等；
    rerank_scores: 本次检索的重排分数分布（可选，用于判断"差一点还是差很远"）。
    """
    if not query or not query.strip():
        return
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:200],
        "tag": tag[:50],
        "reason": reason,
        "user_id": (user_id or "")[:50],
        "rerank_scores": rerank_scores[:10] if rerank_scores else None,
    }
    with _missed_lock:
        os.makedirs(os.path.dirname(MISSED_PATH), exist_ok=True)
        try:
            with open(MISSED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _trim_if_needed()
        except OSError:
            pass


def _trim_if_needed() -> None:
    """文件过大时重写保留最近 MAX_ENTRIES 条（持锁调用）"""
    try:
        if os.path.exists(MISSED_PATH) and os.path.getsize(MISSED_PATH) > TRIM_SIZE_BYTES:
            # load_missed 最新在前，重写前反转回"追加顺序=时间序（最新在末尾）"
            lines = list(reversed(load_missed(limit=MAX_ENTRIES)))
            with open(MISSED_PATH, "w", encoding="utf-8") as f:
                for e in lines:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_missed(limit: int = 1000) -> List[Dict]:
    """读取最近 N 条检索失败记录（最新在前）"""
    if not os.path.exists(MISSED_PATH):
        return []
    lines = []
    try:
        with open(MISSED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (OSError, json.JSONDecodeError):
        return []
    return lines[-limit:][::-1]


def get_missed_summary(top_n: int = 20) -> List[Dict]:
    """聚合高频未命中：按 (query, tag) 计数降序，供管理端展示知识库盲区"""
    counts: Dict[tuple, int] = {}
    last_seen: Dict[tuple, str] = {}
    reasons: Dict[tuple, set] = {}
    for e in load_missed(limit=5000):
        key = (e.get("query", ""), e.get("tag", ""))
        if not key[0]:
            continue
        counts[key] = counts.get(key, 0) + 1
        ts = e.get("timestamp", "")
        if ts > last_seen.get(key, ""):
            last_seen[key] = ts
        r = e.get("reason", "empty")
        reasons.setdefault(key, set()).add(r)
    top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    return [
        {
            "query": q,
            "tag": t,
            "count": c,
            "last_seen": last_seen.get((q, t), ""),
            "reasons": sorted(reasons.get((q, t), set())),
        }
        for (q, t), c in top
    ]
