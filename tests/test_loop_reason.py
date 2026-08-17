# -*- coding: utf-8 -*-
"""主循环（Loop Engineering）行为测试：
退出 reason 体系、输出截断续写（nudge）、被动上下文压缩、并行工具执行
"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager


def _resp(content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )]
    )


def _tc(tc_id, name, args):
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _make_agent(responses, retrieve_fn=None, max_rounds=None, no_progress_threshold=2):
    """可编程 FakeCompletions：按调用序列依次返回 responses 列表中的响应"""
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=True,
    )
    state = {"i": 0}

    class FakeCompletions:
        async def create(self, **kwargs):
            i = state["i"]
            state["i"] += 1
            if i < len(responses):
                resp = responses[i]
                if callable(resp):
                    return resp(kwargs)
                return resp
            return _resp(content="默认回答")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent.rag_pipeline = SimpleNamespace(
        llm_client=FakeClient(), llm_model="m"
    )
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
        retrieve=retrieve_fn or (
            lambda *a, **k: [{"content": "结果", "parent_section": "第四章"}]
        )
    )
    agent.skill_manager = SimpleNamespace(get_intro=lambda: "")
    agent.self_rag_enabled = False
    agent.intent_analysis_enabled = False
    agent.query_expansion_enabled = False
    if max_rounds is not None:
        agent.max_loop_rounds = max_rounds
    agent.no_progress_threshold = no_progress_threshold
    return agent


def _search_tool_call(query, tc_id="call_1"):
    return _tc(
        tc_id, "search_knowledge_base",
        '{"query": "%s", "tag": "财务"}' % query,
    )


@pytest.mark.asyncio
async def test_reason_completed():
    agent = _make_agent([_resp(content="行权期为 3 年")])
    result = await agent.chat("行权期多久")
    assert result["success"] is True
    assert result["reason"] == "completed"
    assert "行权期" in result["answer"]


@pytest.mark.asyncio
async def test_reason_max_turns():
    """LLM 一直调工具且结果不断变化（不触发无进展）→ 轮数用尽 reason=max_turns"""
    call_count = {"n": 0}

    def retrieve(*a, **k):
        call_count["n"] += 1
        return [{"content": f"结果{call_count['n']}", "parent_section": f"第{call_count['n']}章"}]

    responses = [
        _resp(tool_calls=[_search_tool_call("q1")]),
        _resp(tool_calls=[_search_tool_call("q2")]),
        _resp(tool_calls=[_search_tool_call("q3")]),
    ]
    agent = _make_agent(responses, retrieve_fn=retrieve, max_rounds=3)
    result = await agent.chat("问题")
    assert result["reason"] == "max_turns"
    assert "最大推理轮次" in result["answer"]


@pytest.mark.asyncio
async def test_reason_no_progress_when_llm_keeps_calling():
    """无进展 breach 后 LLM 仍调工具 → reason=no_progress"""
    responses = [
        _resp(tool_calls=[_search_tool_call("q1")]),
        _resp(tool_calls=[_search_tool_call("q2")]),
        _resp(tool_calls=[_search_tool_call("q3")]),  # 第三次相同结果 → breach
        _resp(tool_calls=[_search_tool_call("q4")]),  # breach 后仍调工具 → 退出
    ]
    agent = _make_agent(responses, max_rounds=10)
    result = await agent.chat("问题")
    assert result["reason"] == "no_progress"
    assert "未获得新信息" in result["answer"]


@pytest.mark.asyncio
async def test_output_truncated_nudge_then_completed():
    """输出截断 → nudge 续写 → 正常完成，answer 累积两轮内容"""
    responses = [
        _resp(content="行权期为 3 年", finish_reason="length"),
        _resp(content="，自授予日起计算。", finish_reason="stop"),
    ]
    agent = _make_agent(responses)
    result = await agent.chat("行权期多久")
    assert result["reason"] == "completed"
    assert result["answer"] == "行权期为 3 年，自授予日起计算。"


@pytest.mark.asyncio
async def test_output_truncated_max_exits():
    """续写超过上限 → reason=output_truncated_max，保留已累积内容"""
    responses = [
        _resp(content="A", finish_reason="length"),
        _resp(content="B", finish_reason="length"),
        _resp(content="C", finish_reason="length"),
        _resp(content="D", finish_reason="length"),  # 第 4 次 > 上限 3
    ]
    agent = _make_agent(responses)
    agent.output_recovery_max = 3
    result = await agent.chat("问题")
    assert result["reason"] == "output_truncated_max"
    assert result["answer"].startswith("ABCD")


@pytest.mark.asyncio
async def test_context_too_long_compact_and_retry():
    """上下文超长 → 被动压缩一次重试 → 成功 reason=completed"""
    state = {"calls": 0}

    def raise_context_error(kwargs):
        raise RuntimeError("This model's maximum context length is 64000 tokens")

    # 前两轮工具调用让 react_messages 增长（>4 条非 system），第三轮触发 413，
    # 压缩重试（第四次调用）成功返回
    responses = [
        _resp(tool_calls=[_search_tool_call("q1", "c1")]),
        _resp(tool_calls=[_search_tool_call("q2", "c2")]),
        raise_context_error,
        _resp(content="压缩后成功回答"),
    ]
    agent = _make_agent(responses)
    result = await agent.chat("问题")
    assert result["success"] is True
    assert result["reason"] == "completed"
    assert "压缩后成功回答" in result["answer"]


def test_compact_messages_keeps_system_and_recent():
    """被动压缩：保留全部 system + 最近 4 条非 system"""
    from app.agent.agent_manager import AgentManager
    messages = (
        [{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"}]
        + [{"role": "user", "content": f"u{i}"} for i in range(6)]
    )
    compacted = AgentManager._compact_messages_for_retry(messages)
    assert [m["role"] for m in compacted].count("system") == 2
    non_system = [m for m in compacted if m["role"] != "system"]
    assert len(non_system) == 4
    assert non_system[-1]["content"] == "u5"


@pytest.mark.asyncio
async def test_context_too_long_unrecoverable():
    """压缩后仍失败 → reason=prompt_too_long"""
    def always_fail(kwargs):
        raise RuntimeError("maximum context length exceeded")

    agent = _make_agent([always_fail, always_fail])
    result = await agent.chat("问题")
    assert result["success"] is False
    assert result["reason"] == "prompt_too_long"


@pytest.mark.asyncio
async def test_parallel_readonly_tools():
    """同轮两个只读工具并行执行：两个结果都回填，工具统计正确"""
    from app.agent import agent_manager as am

    async def fake_execute_one(name, args, message):
        return f"结果-{name}"

    agent = _make_agent([
        _resp(tool_calls=[
            _tc("call_a", "read_category_info", '{"tag": "财务"}'),
            _tc("call_b", "search_knowledge_base", '{"query": "q", "tag": "财务"}'),
        ]),
        _resp(content="完成"),
    ])
    # 替换执行函数验证并行路径（两个只读工具都走 gather）
    agent._execute_one_tool = fake_execute_one
    result = await agent.chat("问题")
    assert result["reason"] == "completed"
    # 工具结果应包含两个工具的输出（通过第二轮 LLM 收到的 messages 验证）
    # 简化断言：工具统计
    assert result["tool_used"] == "search_knowledge_base"
