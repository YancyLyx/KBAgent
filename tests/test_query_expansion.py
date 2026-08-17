# -*- coding: utf-8 -*-
"""查询增强测试：多查询分解 / HyDE / agent 集成"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.rag.query_expansion import decompose_query, generate_hypothetical_doc
from app.agent.agent_manager import AgentManager


def _client_responding(content):
    FakeMsg = type("FakeMsg", (), {"content": content})
    FakeChoice = type("FakeChoice", (), {"message": FakeMsg()})
    FakeResp = type("FakeResp", (), {"choices": [FakeChoice()]})

    class Completions:
        async def create(self, **kwargs):
            return FakeResp()

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def _client_failing():
    class Completions:
        async def create(self, **kwargs):
            raise RuntimeError("llm unavailable")

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


@pytest.mark.asyncio
async def test_decompose_query_json():
    client = _client_responding('["差旅住宿标准是多少", "餐补标准是多少"]')
    subs = await decompose_query(client, "m", "差旅和餐补分别怎么规定？")
    assert subs == ["差旅住宿标准是多少", "餐补标准是多少"]


@pytest.mark.asyncio
async def test_decompose_query_fallback():
    assert await decompose_query(_client_failing(), "m", "问题") == ["问题"]
    assert await decompose_query(_client_responding("不是JSON"), "m", "问题") == ["问题"]


@pytest.mark.asyncio
async def test_hyde():
    client = _client_responding("员工持股计划的行权期为 3 年，自授予日起计算。")
    doc = await generate_hypothetical_doc(client, "m", "行权期是多久")
    assert "行权期" in doc
    assert await generate_hypothetical_doc(_client_failing(), "m", "q") is None


def test_should_expand_query():
    assert AgentManager._should_expand_query("差旅和餐补分别怎么规定")
    assert AgentManager._should_expand_query("这是一段非常长的问题，超过二十五个字，需要分解成子查询来处理")
    assert not AgentManager._should_expand_query("行权期是多久")


def _agent_with_retrieve(retrieve_fn):
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    agent.rag_pipeline = SimpleNamespace(
        llm_client=_client_responding('["子问题A", "子问题B"]'),
        llm_model="m",
        retrieve=retrieve_fn,
    )
    return agent


@pytest.mark.asyncio
async def test_search_with_expansion_merges():
    def fake_retrieve(query, top_k=3, use_rerank=True, tag=None):
        if query == "子问题A" or query == "差旅和餐补分别怎么规定":
            return [{"content": f"结果-{query}", "section": "第四章"}]
        return []

    agent = _agent_with_retrieve(fake_retrieve)
    result = await agent._search_with_expansion(
        {"query": "差旅和餐补分别怎么规定", "tag": "差旅报销"}
    )
    assert result is not None
    assert "结果-差旅和餐补分别怎么规定" in result
    assert "多路查询合并结果" in result


@pytest.mark.asyncio
async def test_search_with_expansion_skips_simple():
    def fake_retrieve(query, top_k=3, use_rerank=True, tag=None):
        return [{"content": "x", "section": ""}]

    agent = _agent_with_retrieve(fake_retrieve)
    # 短查询不触发增强 → 返回 None（走原路径）
    assert await agent._search_with_expansion({"query": "行权期", "tag": "股权激励"}) is None


@pytest.mark.asyncio
async def test_search_with_expansion_no_results():
    agent = _agent_with_retrieve(lambda *a, **k: [])
    assert await agent._search_with_expansion(
        {"query": "差旅和餐补分别怎么规定", "tag": "差旅报销"}
    ) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
