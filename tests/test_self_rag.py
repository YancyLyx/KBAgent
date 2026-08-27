# -*- coding: utf-8 -*-
"""Self-RAG 检索充分性判断测试"""

import sys
import asyncio
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agent.agent_manager import AgentManager


def _client_responding(content):
    FakeMsg = type("FakeMsg", (), {"content": content})
    FakeChoice = type("FakeChoice", (), {"message": FakeMsg()})
    FakeResp = type("FakeResp", (), {"choices": [FakeChoice()]})

    class Completions:
        async def create(self, **kwargs):
            return FakeResp()

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


class _FakeReranker:
    def rerank(self, query, docs, top_k=3, threshold=None, autocut=False, drop_ratio=0.3):
        out = list(docs)
        for i, d in enumerate(out):
            d.setdefault("rerank_score", 1.0 - i * 0.1)
        return out[:top_k]


def _agent_with_retrieve(retrieve_fn):
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    agent.rag_pipeline = SimpleNamespace(
        llm_client=_client_responding(""), llm_model="m", reranker=_FakeReranker()
    )
    agent.tool_router = SimpleNamespace(
        skill_pipeline=SimpleNamespace(retrieve=retrieve_fn)
    )
    return agent


def _ctx(content, section):
    return {"content": content, "parent_section": section}


@pytest.mark.asyncio
async def test_sufficient_returns_first_result_no_second_retrieval():
    """sufficient=true：只检索一次，返回第一次结果，不触发补检"""
    calls = []

    def retrieve(query, top_k, use_rerank, tag):
        calls.append(query)
        return [_ctx("差旅住宿每晚上限 500 元。", "第四章")]

    agent = _agent_with_retrieve(retrieve)
    agent.rag_pipeline.llm_client = _client_responding(
        '{"sufficient": true, "rewritten_query": ""}'
    )
    result = await agent._search_with_self_rag({"query": "住宿能报多少", "tag": "财务"})
    assert result is not None
    assert "差旅住宿每晚上限 500 元" in result
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_insufficient_rewrites_and_merges():
    """sufficient=false：按改写查询补检，内容去重后合并"""
    calls = []

    def retrieve(query, top_k, use_rerank, tag):
        calls.append(query)
        if query == "住宿能报多少":
            return [_ctx("差旅住宿每晚上限 500 元。", "第四章")]
        return [
            _ctx("差旅住宿每晚上限 500 元。", "第四章"),
            _ctx("一线城市住宿标准见附表。", "附表"),
        ]

    agent = _agent_with_retrieve(retrieve)
    agent.rag_pipeline.llm_client = _client_responding(
        '{"sufficient": false, "rewritten_query": "一线城市住宿标准"}'
    )
    result = await agent._search_with_self_rag({"query": "住宿能报多少", "tag": "财务"})
    assert result is not None
    assert "一线城市住宿标准见附表" in result
    assert len(calls) == 2
    # 去重：重复内容只出现一次
    assert result.count("差旅住宿每晚上限 500 元") == 1


@pytest.mark.asyncio
async def test_insufficient_no_new_info_stops_after_one_retry():
    """补检结果与首次完全重复（无新信息）→ 合并结果不变、只补检一次"""
    calls = []
    same = [_ctx("差旅住宿每晚上限 500 元。", "第四章")]

    def retrieve(query, top_k, use_rerank, tag):
        calls.append(query)
        return list(same)

    agent = _agent_with_retrieve(retrieve)
    agent.self_rag_max_rounds = 3  # 即使配置允许多次，无新增也必须停
    agent.rag_pipeline.llm_client = _client_responding(
        '{"sufficient": false, "rewritten_query": "住宿标准"}'
    )
    result = await agent._search_with_self_rag({"query": "住宿能报多少", "tag": "财务"})
    assert result is not None
    assert result.count("差旅住宿每晚上限 500 元") == 1
    assert len(calls) == 2  # 首次 + 1 次补检，不再重复


@pytest.mark.asyncio
async def test_failure_falls_back_to_none():
    """LLM 判断失败/检索失败 → 返回 None，走原工具路径"""
    agent = _agent_with_retrieve(lambda *a, **k: [_ctx("x", "s")])
    agent.rag_pipeline.llm_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=type(
                "C",
                (),
                {"create": lambda self, **kw: (_ for _ in ()).throw(RuntimeError("fail"))},
            )()
        )
    )
    result = await agent._search_with_self_rag({"query": "q", "tag": "财务"})
    assert result is None

    # 无 pipeline（skill_pipeline 为 None）也回退
    agent2 = _agent_with_retrieve(lambda *a, **k: [])
    agent2.tool_router = SimpleNamespace(skill_pipeline=None)
    assert await agent2._search_with_self_rag({"query": "q", "tag": "财务"}) is None


def test_parse_json_loose_variants():
    p = AgentManager._parse_json_loose
    assert p('{"sufficient": true}') == {"sufficient": True}
    assert p('```json\n{"sufficient": false}\n```') == {"sufficient": False}
    assert p('判断结果：{"sufficient": true, "rewritten_query": "x"} 结束') == {
        "sufficient": True,
        "rewritten_query": "x",
    }
    assert p("不是 JSON") == {}
    assert p("") == {}


def test_format_search_result_shape():
    r = AgentManager._format_search_result(
        "q", "财务", [_ctx("内容A", "第一章"), _ctx("内容B", "第二章")]
    )
    assert "[1] 内容A" in r
    assert "（来源章节：第二章）" in r
    assert "来源：财务" in r


@pytest.mark.asyncio
async def test_chat_no_progress_detection_integration(tmp_path, monkeypatch):
    """集成：LLM 换参数连续检索但结果相同 → 第二次重复时回填系统提示"""
    # 隔离监控埋点：chat() 末尾会 record_request 写 data/request_metrics.jsonl，
    # 集成测试不能污染仓库真实数据
    monkeypatch.setattr(
        "app.eval.metrics.METRICS_PATH", str(tmp_path / "metrics.jsonl")
    )
    agent = AgentManager(
        user_id="u",
        enable_memory=False,
        enable_rag=False,
        enable_tools=True,
    )
    agent.no_progress_threshold = 2
    queries = ["差旅报销", "差旅 报销", "差旅报销标准"]
    calls = {"n": 0}

    def _resp(tool_calls=None, content=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=content, tool_calls=tool_calls
            ))]
        )

    def _tc_args(q):
        return [SimpleNamespace(
            id=f"call_{calls['n']}",
            function=SimpleNamespace(
                name="search_knowledge_base",
                arguments=f'{{"query": "{q}", "tag": "财务"}}',
            ),
        )]

    class FakeCompletions:
        async def create(self, **kwargs):
            calls["n"] += 1
            messages = kwargs.get("messages", [])
            if calls["n"] <= 3:
                # 前三轮都返回工具调用（参数不同但检索结果相同）
                return _resp(tool_calls=_tc_args(queries[calls["n"] - 1]))
            # 第四轮：LLM 看到无进展提示后直接回答
            joined = "\n".join(
                str(m.get("content", "")) for m in messages
            )
            assert "[系统提示] 本轮检索/工具结果与上一轮基本相同" in joined
            return _resp(content="差旅报销标准为每晚 400 元。")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent.rag_pipeline = SimpleNamespace(
        llm_client=FakeClient(), llm_model="fake-model"
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
        retrieve=lambda *a, **k: [{
            "content": "差旅报销标准为每晚 400 元",
            "parent_section": "第四章",
        }]
    )
    agent.skill_manager = SimpleNamespace(get_intro=lambda: "")
    # Self-RAG 关闭：走工具原路径（避免额外 LLM 调用干扰计数）
    agent.self_rag_enabled = False
    # 意图分析/查询增强关闭：本测试聚焦无进展检测，避免额外 LLM 调用干扰计数
    agent.intent_analysis_enabled = False
    agent.query_expansion_enabled = False

    result = await agent.chat("差旅报销标准是多少")
    assert result["success"] is True
    assert "400 元" in result["answer"]
    assert calls["n"] == 4
