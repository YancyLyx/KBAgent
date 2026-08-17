# -*- coding: utf-8 -*-
"""用户偏好 JSONL 存储测试：追加 / 时间线 / 软删 / 修改(supersedes)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.memory import pref_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(pref_store, "PROFILES_DIR", str(tmp_path / "profiles"))
    yield


def test_append_and_timeline(store):
    pref_store.append_pref("u1", "不喜欢吃青菜", source="extraction")
    pref_store.append_pref("u1", "喜欢吃菠菜", source="extraction")
    timeline = pref_store.get_timeline("u1")
    assert len(timeline) == 2
    assert timeline[0]["text"] == "不喜欢吃青菜"
    assert timeline[1]["text"] == "喜欢吃菠菜"
    assert timeline[0]["status"] == "active"
    assert timeline[0]["id"]


def test_delete_is_soft(store):
    e1 = pref_store.append_pref("u1", "不喜欢吃青菜", source="extraction")
    pref_store.append_pref("u1", "喜欢吃菠菜", source="extraction")
    assert pref_store.delete_pref("u1", e1["id"]) is True
    # 时间线保留全部历史
    assert len(pref_store.get_timeline("u1")) == 3
    # 当前生效只留下一条 active 且未被删除的
    active = pref_store.get_active("u1")
    assert len(active) == 1
    assert active[0]["text"] == "喜欢吃菠菜"
    # 删除不存在的条目返回 False
    assert pref_store.delete_pref("u1", "nonexist") is False


def test_update_supersedes(store):
    old = pref_store.append_pref("u1", "示例用 Python", source="extraction")
    assert pref_store.update_pref("u1", old["id"], "示例用 Go") is True
    timeline = pref_store.get_timeline("u1")
    assert len(timeline) == 2
    assert timeline[1]["supersedes"] == old["id"]
    # 当前生效：旧条目被取代，只留新条目
    active = pref_store.get_active("u1")
    assert len(active) == 1
    assert active[0]["text"] == "示例用 Go"
    # 空文本不允许
    assert pref_store.update_pref("u1", old["id"], "   ") is False


def test_active_limit(store):
    for i in range(5):
        pref_store.append_pref("u1", f"偏好{i}", source="manual")
    active = pref_store.get_active("u1", limit=3)
    assert len(active) == 3
    assert active[-1]["text"] == "偏好4"  # 取最近 3 条


def test_user_isolation(store):
    pref_store.append_pref("u1", "u1 的偏好", source="manual")
    pref_store.append_pref("u2", "u2 的偏好", source="manual")
    assert len(pref_store.get_active("u1")) == 1
    assert len(pref_store.get_active("u2")) == 1
    assert pref_store.get_active("u1")[0]["text"] == "u1 的偏好"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
