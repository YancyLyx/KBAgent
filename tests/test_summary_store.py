# -*- coding: utf-8 -*-
"""演进式摘要 JSONL 存储测试：追加/倒序读/清理/清空/迁移"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.memory import summary_store


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(summary_store, "SUMMARIES_DIR", str(tmp_path))
    return tmp_path


def test_add_and_get_running(tmp_dir):
    summary_store.add_summary("u1", "第一版：用户做 API 开发")
    summary_store.add_summary("u1", "第二版：改用 JWT 认证")
    assert summary_store.get_running_summary("u1") == "第二版：改用 JWT 认证"


def test_get_summaries_latest_first(tmp_dir):
    summary_store.add_summary("u1", "A")
    summary_store.add_summary("u1", "B")
    summary_store.add_summary("u1", "C")
    summaries = summary_store.get_summaries("u1", limit=2)
    assert [s["summary"] for s in summaries] == ["C", "B"]
    assert "created_at" in summaries[0]


def test_keep_recent_trim(tmp_dir):
    for i in range(5):
        summary_store.add_summary("u1", f"第{i}版", keep_recent=3)
    summaries = summary_store.get_summaries("u1", limit=10)
    assert len(summaries) == 3
    assert summaries[0]["summary"] == "第4版"


def test_clear(tmp_dir):
    summary_store.add_summary("u1", "A")
    assert summary_store.get_running_summary("u1") == "A"
    assert summary_store.clear("u1") is True
    assert summary_store.get_running_summary("u1") == ""


def test_migrate_from_sqlite(tmp_dir):
    rows = [
        {"summary": "旧版1", "key_points": "", "created_at": "2026-01-01T00:00:00"},
        {"summary": "旧版2", "key_points": "", "created_at": "2026-02-01T00:00:00"},
    ]
    n = summary_store.migrate_from_sqlite("u_mig", rows)
    assert n == 2
    assert summary_store.get_running_summary("u_mig") == "旧版2"
    # 已存在则不重复迁移
    assert summary_store.migrate_from_sqlite("u_mig", rows) == 0


def test_empty_input_safe(tmp_dir):
    assert summary_store.add_summary("", "x") is False
    assert summary_store.get_running_summary("nobody") == ""
    assert summary_store.get_summaries("nobody") == []
