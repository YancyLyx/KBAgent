# -*- coding: utf-8 -*-
"""实时评测存储模块

每次对话结束后自动评分，结果持久化到 JSON 文件。
供 /api/admin/eval/realtime 读取。
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional

SCORES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "realtime_scores.json",
)

_record_lock = threading.Lock()


def record(query: str, answer: str, relevance: int, completeness: int,
           usefulness: int, explanation: str, memory_context: str = "",
           faithfulness: Optional[int] = None) -> None:
    """记录一条实时评测分数

    memory_context：本轮注入的记忆片段摘要（回忆/画像/演进式摘要），
    用于 badcase 回溯时定位"回答是否被某条记忆污染"。旧记录无此字段，
    读取端用 .get() 兼容。
    faithfulness：忠实度 1-5（提供参考文档时才评测，无则 None）。
    """
    # 读-改-写加锁：避免并发请求交错写坏 JSON
    with _record_lock:
        scores = load()
        scores.append({
            "query": query,
            "answer": answer[:200],
            "relevance": relevance,
            "completeness": completeness,
            "usefulness": usefulness,
            "explanation": explanation,
            "memory_context": memory_context[:300],
            "faithfulness": faithfulness,
            "timestamp": datetime.now().isoformat(),
        })
        scores = scores[-200:]  # 只保留最近 200 条
        os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
        with open(SCORES_PATH, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)


def load() -> List[Dict]:
    """读取所有实时评测记录"""
    if not os.path.exists(SCORES_PATH):
        return []
    try:
        with open(SCORES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []
