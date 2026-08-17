# -*- coding: utf-8 -*-
"""检索兜底策略测试：诚实原则 prompt、strict/lenient 场景、失败 query 记录"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager
from app.eval import missed_queries


def _agent(tmp_path, monkeypatch):
    monkeypatch.setattr(missed_queries, "MISSED_PATH", str(tmp_path / "missed.jsonl"))
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    agent.strict_tags = ["薪酬绩效", "财务"]
    agent.lenient_tags = ["综合"]
    agent.tool_router = SimpleNamespace(
        skill_pipeline=SimpleNamespace(
            retrieve=lambda *a, **k: [{
                "content": "低相关候选内容",
                "parent_section": "附录",
            }]
        )
    )
    return agent


def test_system_prompt_contains_honesty_rule():
    """诚实原则必须写进主循环 system prompt（不能靠模型自觉）"""
    prompt = AgentManager.SYSTEM_PROMPT_TPL.format(
        skill_intro="## 知识库概览",
        fallback_rule="- 默认按严格场景处理",
    )
    assert "不要编造答案" in prompt
    assert "引导换关键词或补充细节" in prompt
    assert "以下为通用知识" in prompt


def test_build_fallback_rule_lists_tags():
    agent = AgentManager(
        user_id="u", enable_memory=False, enable_rag=False, enable_tools=False
    )
    agent.strict_tags = ["薪酬绩效", "财务"]
    agent.lenient_tags = ["综合"]
    rule = agent._build_fallback_rule()
    assert "薪酬绩效" in rule
    assert "严格场景" in rule
    assert "综合" in rule
    assert "宽松场景" in rule


@pytest.mark.asyncio
async def test_strict_tag_guidance(tmp_path, monkeypatch):
    """strict 场景：明确拒绝编造 + 记录失败 query + 附低相关候选"""
    agent = _agent(tmp_path, monkeypatch)
    result = await agent._handle_missed_retrieval(
        "离职后股权怎么处理", "薪酬绩效", "知识库「薪酬绩效」中未找到相关内容。"
    )
    assert "严格场景" in result
    assert "不要用模型内部知识编造" in result
    assert "宽松场景" not in result
    assert "相关度较低，仅供参考" in result
    assert "低相关候选内容" in result
    # 失败 query 已记录（系统层闭环）
    records = missed_queries.load_missed()
    assert records[0]["query"] == "离职后股权怎么处理"
    assert records[0]["tag"] == "薪酬绩效"


@pytest.mark.asyncio
async def test_lenient_tag_guidance(tmp_path, monkeypatch):
    """lenient 场景：允许通用知识兜底但必须标注来源"""
    agent = _agent(tmp_path, monkeypatch)
    result = await agent._handle_missed_retrieval(
        "综合类常识问题", "综合", "知识库「综合」中未找到相关内容。"
    )
    assert "宽松场景" in result
    assert "以下为通用知识，并非企业知识库内容" in result
    assert "严格场景" not in result


@pytest.mark.asyncio
async def test_missed_without_pipeline_still_records(tmp_path, monkeypatch):
    """无 pipeline（检索组件不可用）也要记录失败 query、给出场景话术"""
    agent = _agent(tmp_path, monkeypatch)
    agent.tool_router = SimpleNamespace(skill_pipeline=None)
    agent.rag_pipeline = None
    result = await agent._handle_missed_retrieval(
        "q", "财务", "知识库「财务」中未找到相关内容。"
    )
    assert "严格场景" in result
    assert missed_queries.load_missed()[0]["query"] == "q"


@pytest.mark.asyncio
async def test_chat_empty_result_triggers_fallback(tmp_path, monkeypatch):
    """集成：检索空结果时 tool_result 附带严格场景话术，LLM 能看到"""
    monkeypatch.setattr(missed_queries, "MISSED_PATH", str(tmp_path / "missed.jsonl"))
    monkeypatch.setattr(
        "app.eval.metrics.METRICS_PATH", str(tmp_path / "metrics.jsonl")
    )
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=True,
    )
    agent.strict_tags = ["财务"]
    agent.lenient_tags = []
    calls = {"n": 0}

    def _resp(tool_calls=None, content=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=content, tool_calls=tool_calls
            ))]
        )

    class FakeCompletions:
        async def create(self, **kwargs):
            calls["n"] += 1
            messages = kwargs.get("messages", [])
            if calls["n"] == 1:
                return _resp(tool_calls=[SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="search_knowledge_base",
                        arguments='{"query": "报销流程", "tag": "财务"}',
                    ),
                )])
            joined = "\n".join(str(m.get("content", "")) for m in messages)
            assert "未找到相关内容（严格场景）" in joined
            assert "不要用模型内部知识编造" in joined
            return _resp(content="未找到相关制度，请换关键词或咨询人力。")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent.rag_pipeline = SimpleNamespace(llm_client=FakeClient(), llm_model="m")
    agent.tool_router.skill_manager = SimpleNamespace(
        get_tool_schemas=lambda: [{
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "检索",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}, "tag": {"type": "string"}},
                    "required": ["query", "tag"]},
            },
        }]
    )
    agent.tool_router.skill_pipeline = SimpleNamespace(
        retrieve=lambda *a, **k: []  # 空结果
    )
    agent.skill_manager = SimpleNamespace(get_intro=lambda: "")
    agent.self_rag_enabled = False

    result = await agent.chat("报销流程是什么")
    assert "未找到相关制度" in result["answer"]
    assert missed_queries.load_missed()[0]["query"] == "报销流程是什么"
