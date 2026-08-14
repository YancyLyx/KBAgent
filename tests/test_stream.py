# -*- coding: utf-8 -*-
"""SSE 流式：Agent 层流式生成与 tool_calls 聚合测试"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def _tc(index, tc_id="", name="", args=""):
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


async def _stream_of(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_collect_content_tokens():
    """流式文本：逐 token 回调 + 聚合为完整回答"""
    tokens = []

    async def cb(tok):
        tokens.append(tok)

    stream = _stream_of([
        _chunk(content="你好"),
        _chunk(content="，我是"),
        _chunk(content="助手"),
    ])
    msg = await AgentManager._collect_stream_message(stream, cb)
    assert msg.content == "你好，我是助手"
    assert msg.tool_calls is None
    assert tokens == ["你好", "，我是", "助手"]


@pytest.mark.asyncio
async def test_collect_tool_calls_aggregated():
    """流式工具调用：按 index 聚合 id/name/arguments 增量"""
    stream = _stream_of([
        _chunk(tool_calls=[_tc(0, tc_id="call_1", name="search_", args='{"query"')]),
        _chunk(tool_calls=[_tc(0, name="knowledge_base", args=': "报销"}')]),
        _chunk(tool_calls=[_tc(1, tc_id="call_2", name="read_file", args='{"path": "a.md"}')]),
    ])
    msg = await AgentManager._collect_stream_message(
        stream, lambda tok: asyncio.sleep(0)
    )
    assert msg.content is None
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].function.name == "search_knowledge_base"
    assert msg.tool_calls[0].function.arguments == '{"query": "报销"}'
    assert msg.tool_calls[1].function.name == "read_file"


@pytest.mark.asyncio
async def test_collect_empty_delta_ignored():
    """空 delta（如 role 分片）不影响聚合"""
    stream = _stream_of([
        SimpleNamespace(choices=[SimpleNamespace(delta=None)]),
        _chunk(content="OK"),
    ])
    msg = await AgentManager._collect_stream_message(
        stream, lambda tok: asyncio.sleep(0)
    )
    assert msg.content == "OK"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
