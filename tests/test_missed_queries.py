# -*- coding: utf-8 -*-
"""检索失败 query 日志测试：记录、读取、高频聚合"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.eval import missed_queries


@pytest.fixture
def tmp_missed(tmp_path, monkeypatch):
    monkeypatch.setattr(missed_queries, "MISSED_PATH", str(tmp_path / "missed.jsonl"))
    return tmp_path


def test_record_and_load(tmp_missed):
    missed_queries.record_missed("远程办公补贴怎么申请", "远程办公", reason="empty")
    missed_queries.record_missed("远程办公补贴怎么申请", "远程办公", reason="empty")
    missed_queries.record_missed("行权期多久", "股权激励", reason="low_score")
    records = missed_queries.load_missed()
    assert len(records) == 3
    # 最新在前
    assert records[0]["query"] == "行权期多久"
    assert records[2]["query"] == "远程办公补贴怎么申请"
    assert records[0]["reason"] == "low_score"


def test_record_skips_empty_query(tmp_missed):
    missed_queries.record_missed("", "财务")
    missed_queries.record_missed("  ", "财务")
    assert missed_queries.load_missed() == []


def test_record_with_scores_and_user(tmp_missed):
    missed_queries.record_missed(
        "q", "财务", reason="low_score", user_id="u1", rerank_scores=[0.3, 0.2]
    )
    r = missed_queries.load_missed()[0]
    assert r["rerank_scores"] == [0.3, 0.2]
    assert r["user_id"] == "u1"


def test_get_missed_summary_groups_by_query_tag(tmp_missed):
    missed_queries.record_missed("A", "财务")
    missed_queries.record_missed("A", "财务")
    missed_queries.record_missed("A", "考勤")
    missed_queries.record_missed("B", "财务")
    summary = missed_queries.get_missed_summary(top_n=2)
    assert summary[0]["query"] == "A"
    assert summary[0]["tag"] == "财务"
    assert summary[0]["count"] == 2
    assert summary[1]["count"] == 1
    assert "empty" in summary[0]["reasons"]


def test_missed_summary_empty_when_no_file(tmp_missed):
    assert missed_queries.get_missed_summary() == []
    assert missed_queries.load_missed() == []


def test_trim_keeps_recent_when_oversized(tmp_missed, monkeypatch):
    """文件超过 TRIM_SIZE_BYTES 时重写保留最近 MAX_ENTRIES 条"""
    # 模拟一个超大的历史文件
    with open(missed_queries.MISSED_PATH, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(
                json.dumps(
                    {"query": f"q{i}", "tag": "t", "timestamp": str(i),
                     "padding": "x" * 50}
                )
                + "\n"
            )
    monkeypatch.setattr(missed_queries, "MAX_ENTRIES", 50)
    monkeypatch.setattr(missed_queries, "TRIM_SIZE_BYTES", 1000)
    missed_queries.record_missed("new", "t")
    # 重写后保留最近 MAX_ENTRIES（50）条（含新写的一条）
    records = missed_queries.load_missed(limit=10000)
    assert len(records) == 50
    assert records[0]["query"] == "new"
