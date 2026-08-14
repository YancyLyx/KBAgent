# -*- coding: utf-8 -*-
"""查询改写（指代补全）测试

背景：向量检索对"无实体词的纯指代"（如"那个方案后来怎么样了"）会失效——
三条历史记忆的相似度几乎相同，分不出哪条是"那个方案"。
方案：检索前用 LLM 把指代补全成明确实体；只有命中指代词才触发（低频低成本），
失败回退原查询。
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


def test_may_contain_reference():
    """指代检测：含指代词触发，正常查询不触发"""
    assert AgentManager._may_contain_reference("上次说的那个认证方案")
    assert AgentManager._may_contain_reference("那个方案后来怎么样了")
    assert AgentManager._may_contain_reference("我之前提到的问题")
    assert not AgentManager._may_contain_reference("报销流程是什么")
    assert not AgentManager._may_contain_reference("员工持股计划的行权期")


def test_rewrite_query_for_recall():
    """改写：LLM 把指代补全成明确实体"""
    agent = _agent_with_fake_llm(content="JWT 认证方案部署进展")
    result = asyncio.run(agent._rewrite_query_for_recall("那个方案后来怎么样了"))
    assert result == "JWT 认证方案部署进展"


def test_rewrite_fallback_on_error():
    """改写失败：回退原查询，不让回忆链路挂掉"""
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
    result = asyncio.run(agent._rewrite_query_for_recall("那个方案后来怎么样了"))
    assert result == "那个方案后来怎么样了"


def test_rewrite_skipped_without_rag():
    """没有 RAG/LLM 时不改写，直接返回原查询"""
    agent = AgentManager(
        user_id="test_user",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    assert agent.rag_pipeline is None
    result = asyncio.run(agent._rewrite_query_for_recall("那个方案后来怎么样了"))
    assert result == "那个方案后来怎么样了"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
