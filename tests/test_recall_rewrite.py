# -*- coding: utf-8 -*-
"""LLM 意图分析测试（闲聊判断 + 指代补全，一次调用）

背景：语义判断交给 LLM 而非硬编码关键词——向量检索对"无实体词的纯指代"
（如"那个方案后来怎么样了"）会失效，需要 LLM 补全；是否闲聊也由 LLM 判断
（注入路由用）。
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager


def _agent_with_fake_llm(content="JWT 认证方案部署进展"):
    """构造一个带 stub LLM 客户端的 Agent（不加载模型）"""
    agent = AgentManager(
        user_id="test_user",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )

    # 注意：类体不访问外层函数局部变量（`class X: content = content` 会 NameError），
    # 所以用 type() 动态构造，字典字面量在函数作用域内正常解析
    FakeMsg = type("FakeMsg", (), {"content": content})
    FakeChoice = type("FakeChoice", (), {"message": FakeMsg()})
    FakeResp = type("FakeResp", (), {"choices": [FakeChoice()]})

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent.rag_pipeline = type(
        "P", (), {"llm_client": FakeClient(), "llm_model": "fake-model"}
    )()
    return agent


def test_intent_analysis_reference_rewrite():
    """意图分析：LLM 把指代补全成明确实体，且判定非闲聊"""
    agent = _agent_with_fake_llm(
        content='{"is_small_talk": false, "rewritten_query": "JWT 认证方案部署进展"}'
    )
    result = asyncio.run(agent._analyze_query_intent("那个方案后来怎么样了"))
    assert result["is_small_talk"] is False
    assert result["rewritten_query"] == "JWT 认证方案部署进展"


def test_intent_analysis_small_talk():
    """意图分析：问候判定为闲聊，改写保持原样"""
    agent = _agent_with_fake_llm(
        content='{"is_small_talk": true, "rewritten_query": "你好"}'
    )
    result = asyncio.run(agent._analyze_query_intent("你好"))
    assert result["is_small_talk"] is True
    assert result["rewritten_query"] == "你好"


def test_intent_analysis_no_reference_keeps_query():
    """意图分析：无指代时 rewritten_query 原样输出"""
    agent = _agent_with_fake_llm(
        content='{"is_small_talk": false, "rewritten_query": "报销流程是什么"}'
    )
    result = asyncio.run(agent._analyze_query_intent("报销流程是什么"))
    assert result["rewritten_query"] == "报销流程是什么"


def test_intent_analysis_string_boolean_parsed_strictly():
    """LLM 返回字符串布尔（"false"）不能被 Python 的 truthiness 误判"""
    agent = _agent_with_fake_llm(
        content='{"is_small_talk": "false", "rewritten_query": "报销流程是什么"}'
    )
    result = asyncio.run(agent._analyze_query_intent("报销流程是什么"))
    assert result["is_small_talk"] is False

    agent2 = _agent_with_fake_llm(
        content='{"is_small_talk": "true", "rewritten_query": "你好"}'
    )
    result2 = asyncio.run(agent2._analyze_query_intent("你好"))
    assert result2["is_small_talk"] is True


def test_intent_fallback_on_error():
    """意图分析失败：回退为"非闲聊 + 原查询"（宁可按知识问题处理，不丢上下文）"""
    agent = _agent_with_fake_llm()

    class FailCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("llm unavailable")

    class FakeChat:
        completions = FailCompletions()

    class FakeClient:
        chat = FakeChat()

    agent.rag_pipeline = type(
        "P", (), {"llm_client": FakeClient(), "llm_model": "fake-model"}
    )()
    result = asyncio.run(agent._analyze_query_intent("那个方案后来怎么样了"))
    assert result == {"is_small_talk": False, "rewritten_query": "那个方案后来怎么样了"}


def test_intent_skipped_without_rag():
    """没有 RAG/LLM 时不做意图分析，直接回退"""
    agent = AgentManager(
        user_id="test_user",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    assert agent.rag_pipeline is None
    result = asyncio.run(agent._analyze_query_intent("你好"))
    assert result == {"is_small_talk": False, "rewritten_query": "你好"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
