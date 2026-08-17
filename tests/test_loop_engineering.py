# -*- coding: utf-8 -*-
"""Loop Engineering 测试：上下文预算动态计算、无进展检测、配置加载"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager


def _budget_cfg(**overrides):
    cfg = {
        "model_window_tokens": 64000,
        "tool_result_max_chars": 3000,
        "tool_result_min_chars": 500,
        "output_reserve_tokens": 2000,
        "safety_margin_tokens": 4000,
        "chars_per_token": 1.0,
    }
    cfg.update(overrides)
    return cfg


def test_budget_normal_case():
    """正常情况：预算在上限内、随上下文占用收紧"""
    # 64K 窗口 − 当前 5K − 输出预留 2K − 安全 4K = 53K token
    # 剩 5 轮 → 53K / 5 × 0.5 = 5.3K token → clamp 到 3000
    b = AgentManager._compute_tool_result_budget(
        current_chars=5000, remaining_rounds=5, budget_config=_budget_cfg()
    )
    assert b == 3000

    # 上下文占用 50K → 可用 8K → 剩 1 轮 → 4K → clamp 到 3000
    b = AgentManager._compute_tool_result_budget(
        current_chars=50000, remaining_rounds=1, budget_config=_budget_cfg()
    )
    assert b == 3000


def test_budget_clamps_to_min_when_window_tight():
    """窗口快满/上下文很高时回退到下限，而不是算出负数"""
    b = AgentManager._compute_tool_result_budget(
        current_chars=90000, remaining_rounds=5, budget_config=_budget_cfg()
    )
    assert b == 500

    # 窗口极小（模拟 8K 小模型）也要有下限保护
    b = AgentManager._compute_tool_result_budget(
        current_chars=7000,
        remaining_rounds=3,
        budget_config=_budget_cfg(model_window_tokens=8000),
    )
    assert b == 500


def test_budget_remaining_rounds_at_least_one():
    assert AgentManager._compute_tool_result_budget(
        100, 0, _budget_cfg()
    ) >= 500


def test_budget_empty_config_uses_defaults():
    b = AgentManager._compute_tool_result_budget(0, 5, None)
    assert 500 <= b <= 3000


def test_signature_for_search_result_uses_sections():
    """检索工具签名 = 来源章节集合 + 段数：两个查询命中同一批文档视为无进展"""
    r1 = "检索到以下相关文档（来源：财务）：\n\n[1] 差旅报销标准\n  （来源章节：第四章）\n\n[2] 住宿标准\n  （来源章节：第五章）\n"
    r2 = "检索到以下相关文档（来源：财务）：\n\n[1] 差旅报销标准（另一段）\n  （来源章节：第四章）\n\n[2] 住宿标准补充\n  （来源章节：第五章）\n"
    assert AgentManager._tool_result_signature(
        "search_knowledge_base", r1
    ) == AgentManager._tool_result_signature("search_knowledge_base", r2)
    assert AgentManager._tool_result_signature(
        "search_knowledge_base", r1
    ) != AgentManager._tool_result_signature(
        "search_knowledge_base", r1.replace("第四章", "第一章")
    )


def test_signature_for_other_tools_uses_text_prefix():
    s1 = AgentManager._tool_result_signature("read_file", "文件 A 内容：\nhello world")
    s2 = AgentManager._tool_result_signature("read_file", "文件 A 内容：\nhello world")
    s3 = AgentManager._tool_result_signature("read_file", "文件 B 内容：\nhello world")
    assert s1 == s2
    assert s1 != s3
    assert AgentManager._tool_result_signature("read_file", "") == ""


def test_agent_loads_loop_config():
    """AgentManager 初始化时从 loop_config.yaml 加载预算/轮数/Self-RAG 开关"""
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    assert agent.max_loop_rounds == 5
    assert agent.tool_result_max_chars == 3000
    assert agent.tool_result_min_chars == 500
    assert agent.no_progress_threshold >= 1
    assert isinstance(agent.self_rag_enabled, bool)
    assert agent.llm_tokens["react_loop"] == 1000
    assert agent.llm_tokens["summary"] == 400
    assert agent.llm_tokens["fallback"] == 500
    assert agent.llm_tokens["self_rag_judge"] == 150
    assert agent.llm_tokens["recall_rewrite"] == 100
    assert "薪酬绩效" in agent.strict_tags
    assert isinstance(agent.lenient_tags, list)
