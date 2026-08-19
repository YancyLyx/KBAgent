# -*- coding: utf-8 -*-
"""LongMemory 偏好合并与审计测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.memory.long_memory import LongMemory


@pytest.fixture()
def memory(tmp_path, monkeypatch):
    """LongMemory 实例：审计日志写入 tmp，测试后清理测试用户数据"""
    import app.memory.long_memory as lm
    monkeypatch.setattr(lm, "PROFILES_AUDIT_DIR", str(tmp_path / "profiles"))
    mem = LongMemory()
    yield mem
    for uid in ["u_pref", "u_audit", "u_sum", "u_del", "u_clear"]:
        mem.clear_user_memory(uid)


def test_profile_merge_same_text_timestamp_overwrite(memory):
    """同文本偏好：新时间戳覆盖旧时间戳，旧时间戳不覆盖新（防乱序）"""
    from datetime import datetime, timedelta
    future_ts = (datetime.now() + timedelta(days=1)).isoformat()
    past_ts = (datetime.now() - timedelta(days=1)).isoformat()

    memory.update_user_profile(
        "u_pref",
        preferences={"extracted_facts": ["喜欢 Python 示例"]},
        source="summary_extraction",
    )
    # 同文本、时间戳更新（晚于第一条）→ 覆盖旧条目（模拟用户改口）
    memory.update_user_profile(
        "u_pref",
        preferences={"extracted_facts": [
            {"text": "喜欢 Python 示例", "updated_at": future_ts}
        ]},
        source="summary_extraction",
    )
    # 同文本、更旧的时间戳 → 不覆盖（乐观锁：旧请求不能覆盖新请求）
    memory.update_user_profile(
        "u_pref",
        preferences={"extracted_facts": [
            {"text": "喜欢 Python 示例", "updated_at": past_ts}
        ]},
        source="summary_extraction",
    )
    # 新文本 → 合并并存
    memory.update_user_profile(
        "u_pref",
        preferences={"extracted_facts": ["工作场景用公司模板"]},
        source="summary_extraction",
    )

    facts = memory.get_user_memory("u_pref")["profile"]["preferences"]["extracted_facts"]
    assert len(facts) == 2
    by_text = {f["text"]: f for f in facts}
    assert by_text["喜欢 Python 示例"]["updated_at"] == future_ts
    assert by_text["工作场景用公司模板"]["updated_at"]  # 新条目带当前时间戳


def test_profile_audit_md_written(memory, tmp_path):
    """每次画像更新 append 人可读审计日志（变更前/变更后/来源）"""
    memory.update_user_profile(
        "u_audit",
        preferences={"extracted_facts": ["偏好 A"]},
        source="summary_extraction",
    )
    memory.update_user_profile(
        "u_audit",
        preferences={"extracted_facts": ["偏好 B"]},
        source="summary_extraction",
    )

    audit = (tmp_path / "profiles" / "u_audit.md")
    assert audit.exists()
    content = audit.read_text(encoding="utf-8")
    assert content.count("## ") == 2          # 两次更新两条记录
    assert "source=summary_extraction" in content
    assert "变更前" in content and "变更后" in content


def test_format_facts_for_prompt():
    """画像注入格式化：兼容 str/dict、按时间倒序、截断条数与长度"""
    from app.agent.agent_manager import AgentManager
    facts = [
        "旧版纯字符串",
        {"text": "最新偏好，这一条应该排最前面" * 5, "updated_at": "2026-08-12T20:00:00"},
        {"text": "中间偏好", "updated_at": "2026-08-11T10:00:00"},
        {"text": "", "updated_at": "2026-08-10T10:00:00"},  # 空文本跳过
    ]
    lines = AgentManager._format_facts_for_prompt(facts, max_facts=2, max_chars=10)
    assert len(lines) == 2
    assert "最新偏好" in lines[0]            # 时间最新排第一
    assert "（2026-08-12）" in lines[0]      # 带日期标注
    assert "旧版纯字符串" not in "".join(lines)  # 超过 max_facts 的旧条目被裁掉
    text_part = lines[0][2:].split("（")[0]   # 去掉 "- " 前缀和时间戳后缀
    assert len(text_part) <= 10               # 文本被截断到 max_chars


def test_summary_keep_recent(memory):
    """演进式摘要只保留最近 N 行，防止表无限增长"""
    memory.summary_keep_recent = 3
    for i in range(5):
        memory.add_summary("u_sum", f"第{i}版摘要")

    summaries = memory.get_summaries("u_sum", limit=10)
    assert len(summaries) == 3
    assert summaries[0]["summary"] == "第4版摘要"  # 最新在前，保留最近 3 行
    memory.clear_user_memory("u_sum")


def test_delete_profile_fact(memory):
    """管理端纠错：删除指定偏好条目并记录审计"""
    memory.update_user_profile(
        "u_del",
        preferences={"extracted_facts": ["错误的偏好A", "正确的偏好B"]},
        source="summary_extraction",
    )
    ok = memory.delete_profile_fact("u_del", "错误的偏好A")
    assert ok is True

    facts = memory.get_user_memory("u_del")["profile"]["preferences"]["extracted_facts"]
    texts = [f["text"] for f in facts]
    assert "错误的偏好A" not in texts
    assert "正确的偏好B" in texts

    # 不存在的条目返回 False
    assert memory.delete_profile_fact("u_del", "不存在的条目") is False
    memory.clear_user_memory("u_del")


def test_clear_user_memory_removes_summary(memory):
    """清空用户记忆同时删除演进式摘要"""
    memory.add_summary("u_clear", "测试摘要")
    memory.update_user_profile(
        "u_clear",
        preferences={"extracted_facts": ["偏好"]},
        source="summary_extraction",
    )
    assert memory.get_running_summary("u_clear")
    assert memory.get_user_memory("u_clear")["profile"]

    assert memory.clear_user_memory("u_clear") is True
    assert memory.get_running_summary("u_clear") == ""
    assert memory.get_user_memory("u_clear") == {"user_id": "u_clear", "topics": [],
                                                 "profile": {}, "total_interactions": 0,
                                                 "frequent_topics": []}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
