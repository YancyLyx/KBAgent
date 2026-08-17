# -*- coding: utf-8 -*-
"""评测告警模块

当单条对话评分低于阈值时，自动记录告警，供管理面板展示。
"""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional

ALERTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval_alerts.json",
)

# 评分低于此值视为异常
THRESHOLDS = {"relevance": 3, "completeness": 3, "usefulness": 3}

_alert_lock = threading.Lock()


def check_and_alert(
    query: str,
    answer: str,
    relevance: int,
    completeness: int,
    usefulness: int,
    explanation: str = "",
    memory_context: str = "",
    faithfulness: Optional[int] = None,
) -> bool:
    """检查评分是否低于阈值，若低于则记录告警。返回是否触发了告警。"""
    low_faithfulness = (
        faithfulness is not None and faithfulness < THRESHOLDS["usefulness"]
    )
    if (
        usefulness >= THRESHOLDS["usefulness"]
        and relevance >= THRESHOLDS["relevance"]
        and not low_faithfulness
    ):
        return False

    # 读-改-写加锁：避免并发请求交错写坏 JSON
    with _alert_lock:
        alerts = load_alerts()
        alerts.append({
            "query": query[:200],
            "answer": answer[:200],
            "scores": {"relevance": relevance, "completeness": completeness, "usefulness": usefulness},
            "thresholds": THRESHOLDS,
            "explanation": explanation,
            "memory_context": memory_context[:300],
            "faithfulness": faithfulness,
            "timestamp": datetime.now().isoformat(),
        })
        alerts = alerts[-200:]  # 保留最近 200 条
        os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
        with open(ALERTS_PATH, "w", encoding="utf-8") as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
    return True


def load_alerts() -> List[Dict]:
    """读取所有告警记录"""
    if not os.path.exists(ALERTS_PATH):
        return []
    try:
        with open(ALERTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def get_alert_count() -> int:
    """获取告警总数"""
    return len(load_alerts())


def get_summary() -> Dict:
    """获取评测摘要（含告警统计）"""
    alerts = load_alerts()
    recent = alerts[-20:]
    low_usefulness = sum(1 for a in recent if a["scores"]["usefulness"] < 3)
    avg_usefulness = sum(a["scores"]["usefulness"] for a in recent) / max(len(recent), 1)
    return {
        "total_alerts": len(alerts),
        "recent_alerts": len(recent),
        "low_usefulness_count": low_usefulness,
        "avg_usefulness_recent": round(avg_usefulness, 2),
    }
