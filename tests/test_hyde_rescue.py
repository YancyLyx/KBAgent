# -*- coding: utf-8 -*-
"""HyDE 检索失败抢救单元测试

触发条件：常规检索空结果（missed 链入口）且 hyde_rescue 启用。
- 成功：生成假设文档补检命中 → 带"（HyDE 补检命中）"返回，不进 missed 链
- 失败/关闭：走原 missed 链（记录失败 query + strict/lenient 话术）
"""
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.agent_manager import AgentManager


def _client_responding(content):
    FakeMsg = type("FakeMsg", (), {"content": content})
    FakeChoice = type("FakeChoice", (), {"message": FakeMsg()})
    FakeResp = type("FakeResp", (), {"choices": [FakeChoice()]})

    class Completions:
        async def create(self, **kwargs):
            return FakeResp()

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def _agent_with_pipeline(retrieve_fn):
    agent = AgentManager(
        user_id="u", enable_memory=False, enable_rag=False, enable_tools=False
    )
    agent.rag_pipeline = SimpleNamespace(
        llm_client=_client_responding(""), llm_model="m"
    )
    agent.tool_router = SimpleNamespace(
        skill_pipeline=SimpleNamespace(retrieve=retrieve_fn)
    )
    agent.hyde_rescue_enabled = True
    return agent


def _ctx(content, section="第四章"):
    return {"content": content, "section": section}


@pytest.mark.asyncio
async def test_hyde_rescue_hit_returns_marked(monkeypatch):
    """空结果 → HyDE 补检命中 → 带标记返回，不进 missed 链"""
    calls = []

    def retrieve(query, top_k, use_rerank, tag):
        calls.append((query, use_rerank))
        if query == "差旅报销标准是多少的假设文档":
            return [_ctx("差旅住宿每晚上限 500 元。")]
        return []

    agent = _agent_with_pipeline(retrieve)

    async def fake_hypo(client, model, q):
        return "差旅报销标准是多少的假设文档"

    monkeypatch.setattr(
        "app.rag.query_expansion.generate_hypothetical_doc", fake_hypo
    )
    result = await agent._handle_missed_retrieval(
        "差旅报销标准是多少", "差旅报销", "知识库「差旅报销」中未找到相关内容。"
    )
    assert result is not None
    assert "HyDE 补检命中" in result
    assert "差旅住宿每晚上限 500 元" in result
    # 补检用假设文档 + 精排路径（use_rerank=True），只检索一次补检
    assert ("差旅报销标准是多少的假设文档", True) in calls


@pytest.mark.asyncio
async def test_hyde_rescue_miss_falls_to_missed_chain(monkeypatch):
    """HyDE 补检仍空 → 走原 missed 链（strict 话术），不返回空"""
    def retrieve(query, top_k, use_rerank, tag):
        return []  # 主检、补检都空

    agent = _agent_with_pipeline(retrieve)

    async def fake_hypo(client, model, q):
        return "不存在的假设文档"

    monkeypatch.setattr(
        "app.rag.query_expansion.generate_hypothetical_doc", fake_hypo
    )
    result = await agent._handle_missed_retrieval(
        "不存在的问题", "薪酬绩效", "知识库「薪酬绩效」中未找到相关内容。"
    )
    assert result is not None
    assert "HyDE 补检命中" not in result
    assert ("未找到" in result) or ("引导" in result) or ("相关知识库" in result)


@pytest.mark.asyncio
async def test_hyde_rescue_disabled_skips(monkeypatch):
    """关闭 hyde_rescue 时直接走 missed 链，不生成假设文档"""
    def retrieve(query, top_k, use_rerank, tag):
        return []

    agent = _agent_with_pipeline(retrieve)
    agent.hyde_rescue_enabled = False

    async def fake_hypo(client, model, q):
        raise AssertionError("disabled 时不应调用 HyDE")

    monkeypatch.setattr(
        "app.rag.query_expansion.generate_hypothetical_doc", fake_hypo
    )
    result = await agent._handle_missed_retrieval(
        "不存在的问题", "薪酬绩效", "知识库「薪酬绩效」中未找到相关内容。"
    )
    assert result is not None
    assert "HyDE 补检命中" not in result


@pytest.mark.asyncio
async def test_all_below_threshold_feeds_hyde_rescue(monkeypatch):
    """全低于阈值 0.5 → 精排过滤成空 → 工具返回"未找到" → HyDE 抢救触发。

    链路（各环均有测试）：
      rerank 全 < 0.5 → []（test_rerank_threshold.py::test_threshold_empty_result）
      [] → tool_router 返回 "该知识库中未找到相关内容"（含"未找到相关内容"）
      react 循环检测 → _handle_missed_retrieval → 第 0 步 HyDE 补检
    本用例用工具层的真实空结果文案喂入，确认 HyDE 抢救确实触发。
    """
    calls = []

    def retrieve(query, top_k, use_rerank, tag):
        calls.append(query)
        if query == "低相关查询的假设文档":
            return [_ctx("差旅住宿每晚上限 500 元。")]
        return []  # 首次（原 query）为空 = 全低于阈值被滤掉

    agent = _agent_with_pipeline(retrieve)

    async def fake_hypo(client, model, q):
        return "低相关查询的假设文档"

    monkeypatch.setattr(
        "app.rag.query_expansion.generate_hypothetical_doc", fake_hypo
    )
    # 工具层在空结果时的真实文案
    result = await agent._handle_missed_retrieval(
        "低相关查询", "差旅报销", "该知识库中未找到相关内容"
    )
    assert result is not None
    assert "HyDE 补检命中" in result
    assert "差旅住宿每晚上限 500 元" in result
    assert "低相关查询的假设文档" in calls  # HyDE 假设文档确实被用于补检
