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


class _FakeReranker:
    """模拟 CrossEncoder 精排：保持传入顺序、补 rerank_score（合并仲裁用）"""

    def rerank(self, query, docs, top_k=3, threshold=None, autocut=False, drop_ratio=0.3):
        out = list(docs)
        for i, d in enumerate(out):
            d.setdefault("rerank_score", 1.0 - i * 0.1)
        return out[:top_k]


def _agent_with_retrieve(retrieve_fn, llm_content='["子问题A", "子问题B"]'):
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    agent.rag_pipeline = SimpleNamespace(
        llm_client=_client_responding(llm_content),
        llm_model="m",
        retrieve=retrieve_fn,
        reranker=_FakeReranker(),
    )
    return agent


@pytest.mark.asyncio
async def test_search_with_expansion_merges():
    def fake_retrieve(query, top_k=3, use_rerank=True, tag=None):
        if query == "子问题A" or query == "差旅和餐补分别怎么规定":
            return [{"content": f"结果-{query}", "section": "第四章"}]
        return []

    agent = _agent_with_retrieve(fake_retrieve)
    agent.query_expansion_enabled = True  # 显式开启：本用例测合并逻辑，默认配置是关
    result = await agent._search_with_expansion(
        {"query": "差旅和餐补分别怎么规定", "tag": "差旅报销"}
    )
    assert result is not None
    assert "结果-差旅和餐补分别怎么规定" in result
    assert "多路查询合并结果" in result


@pytest.mark.asyncio
async def test_search_with_expansion_llm_judges_simple_no_split():
    """LLM 判断无需拆分（返回 [原 query]）→ 不增强，走原单次检索路径"""
    def fake_retrieve(query, top_k=3, use_rerank=True, tag=None):
        return [{"content": "x", "section": ""}]

    agent = _agent_with_retrieve(
        fake_retrieve, llm_content='["行权期是多久"]'
    )
    assert await agent._search_with_expansion(
        {"query": "行权期是多久", "tag": "股权激励"}
    ) is None


@pytest.mark.asyncio
async def test_search_with_expansion_disabled_by_config():
    """配置关闭 query_expansion 时直接走原路径"""
    agent = _agent_with_retrieve(lambda *a, **k: [])
    agent.query_expansion_enabled = False
    assert await agent._search_with_expansion(
        {"query": "差旅和餐补分别怎么规定", "tag": "差旅报销"}
    ) is None


@pytest.mark.asyncio
async def test_search_with_expansion_no_results():
    agent = _agent_with_retrieve(lambda *a, **k: [])
    assert await agent._search_with_expansion(
        {"query": "差旅和餐补分别怎么规定", "tag": "差旅报销"}
    ) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
