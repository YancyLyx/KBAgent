# -*- coding: utf-8 -*-
"""流式 + 工具调用的完整链路测试：第一轮流式返回 tool_calls，第二轮流式返回 content"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tc(index, tc_id="", name="", args=""):
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _stream_of(chunks):
    for c in chunks:
        yield c


class _ToolCallStream:
    """第一轮：流式返回两个 tool_calls 的增量"""

    def __init__(self):
        self.chunks = [
            _chunk(_delta()),
            _chunk(_delta(tool_calls=[_tc(0, tc_id="call_1", name="search_knowledge_base", args='{"query"')])),
            _chunk(_delta(tool_calls=[_tc(0, args=': "行权期", "tag": "股权"}')])),
            _chunk(_delta(content="")),
        ]
        self.i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= len(self.chunks):
            raise StopAsyncIteration
        c = self.chunks[self.i]
        self.i += 1
        return c


class _AnswerStream:
    """第二轮：流式返回 content"""

    def __init__(self):
        self.chunks = [
            _chunk(_delta()),
            _chunk(_delta(content="行权期")),
            _chunk(_delta(content="为 3 年")),
        ]
        self.i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= len(self.chunks):
            raise StopAsyncIteration
        c = self.chunks[self.i]
        self.i += 1
        return c


def _make_agent():
    """构造带 fake LLM/工具的 Agent：第一轮工具调用，第二轮回答"""
    agent = AgentManager(
        user_id="stream_tool_u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=True,
    )

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _ToolCallStream()
            return _AnswerStream()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent.rag_pipeline = SimpleNamespace(
        llm_client=FakeClient(),
        llm_model="fake-model",
        retrieve=lambda *a, **k: [],
    )

    class FakeSkill:
        def get_intro(self):
            return "## 知识库概览"

        def get_tool_schemas(self):
            return [{
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "检索",
                    "parameters": {"type": "object", "properties": {
                        "query": {"type": "string"}, "tag": {"type": "string"}},
                        "required": ["query", "tag"]},
                },
            }]

        def get_category_reference(self, tag):
            return f"## {tag}"

    agent.skill_manager = FakeSkill()
    agent.tool_router.skill_manager = FakeSkill()
    agent.tool_router.skill_pipeline = SimpleNamespace(
        retrieve=lambda *a, **k: [{"content": "行权期为 3 年", "section": "第四章"}]
    )
    # 本测试聚焦流式工具链路，关闭 LLM 意图分析/查询增强避免额外调用干扰计数
    agent.intent_analysis_enabled = False
    agent.query_expansion_enabled = False
    return agent


@pytest.mark.asyncio
async def test_stream_chat_with_tool_then_answer():
    """工具调用轮无 token，回答轮逐 token 推送，最终 answer 完整"""
    agent = _make_agent()
    tokens = []

    async def cb(tok):
        tokens.append(tok)

    result = await agent.chat(
        "员工持股计划的行权期是多久",
        stream_callback=cb,
    )
    assert result["answer"] == "行权期为 3 年"
    assert "".join(tokens) == "行权期为 3 年"
    assert result.get("tool_used") == "search_knowledge_base"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
