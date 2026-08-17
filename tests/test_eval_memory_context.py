# -*- coding: utf-8 -*-
"""评测埋点 memory_context 测试：badcase 回溯时能查到本轮注入了哪些记忆"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.eval import realtime
from app.eval import alerts


@pytest.fixture
def tmp_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(realtime, "SCORES_PATH", str(tmp_path / "scores.json"))
    return tmp_path


@pytest.fixture
def tmp_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_PATH", str(tmp_path / "alerts.json"))
    return tmp_path


def test_record_keeps_memory_context(tmp_scores):
    realtime.record(
        "差旅报销标准",
        "每晚 400 元",
        relevance=5,
        completeness=4,
        usefulness=5,
        explanation="直接命中",
        memory_context="## 用户偏好历史\n- 2026-08-01: 常出差",
    )
    records = realtime.load()
    assert records[-1]["memory_context"] == "## 用户偏好历史\n- 2026-08-01: 常出差"
    # 截断上限 300
    realtime.record(
        "q2", "a2", 4, 4, 4, "", memory_context="x" * 500
    )
    assert len(realtime.load()[-1]["memory_context"]) == 300


def test_record_without_memory_context_compatible(tmp_scores):
    """旧调用不传 memory_context：字段为空字符串，不破坏记录"""
    realtime.record("q", "a", 4, 4, 4, "")
    record = realtime.load()[-1]
    assert record["memory_context"] == ""


def test_alert_keeps_memory_context(tmp_alerts):
    alerted = alerts.check_and_alert(
        "差旅报销标准",
        "可以报销",
        relevance=5,
        completeness=1,
        usefulness=1,
        explanation="信息不足",
        memory_context="## 对话摘要\n用户提到错误标准",
    )
    assert alerted is True
    alert = alerts.load_alerts()[-1]
    assert alert["memory_context"] == "## 对话摘要\n用户提到错误标准"
    assert alert["query"] == "差旅报销标准"


def test_alert_old_records_load_ok(tmp_alerts):
    """旧告警无 memory_context 字段，load 兼容"""
    with open(alerts.ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump([{"query": "q", "answer": "a"}], f, ensure_ascii=False)
    data = alerts.load_alerts()
    assert data[0]["query"] == "q"
