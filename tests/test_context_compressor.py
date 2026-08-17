# -*- coding: utf-8 -*-
"""上下文压缩测试：超长工具结果 LLM 压缩、失败回退截断"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.rag.context_compressor import compress_text
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
async def test_compress_text_success():
    long_text = "关键条款：差旅住宿每晚上限 500 元。" + ("重复填充" * 200)
    result = await compress_text(
        _client_responding("差旅住宿每晚上限 500 元。"),
        "fake-model",
        "出差住宿标准是多少",
        long_text,
        max_chars=100,
    )
    assert result == "差旅住宿每晚上限 500 元。"


@pytest.mark.asyncio
async def test_compress_text_fallback_on_error():
    result = await compress_text(
        _client_failing(), "fake-model", "q", "x" * 5000, max_chars=100
    )
    assert result is None


@pytest.mark.asyncio
async def test_compress_text_truncates_oversized_output():
    result = await compress_text(
        _client_responding("很" * 300),
        "fake-model",
        "q",
        "x" * 5000,
        max_chars=100,
    )
    assert result is not None
    assert len(result) <= 100 + len("...[压缩结果超长已截断]")
    assert "已截断" in result


def _agent_with_llm(client):
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    agent.rag_pipeline = SimpleNamespace(
        llm_client=client,
        llm_model="fake-model",
    )
    return agent


@pytest.mark.asyncio
async def test_agent_compress_tool_result():
    agent = _agent_with_llm(_client_responding("压缩后的关键内容"))
    result = await agent._compress_tool_result("问题", "x" * 5000)
    assert result == "压缩后的关键内容"


@pytest.mark.asyncio
async def test_agent_compress_tool_result_no_llm():
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    assert agent.rag_pipeline is None
    assert await agent._compress_tool_result("问题", "x" * 5000) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
