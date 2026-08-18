# -*- coding: utf-8 -*-
"""第三篇文章三层评估体系测试：
检索 Precision@k、生成忠实度、Pairwise 比较、Judge 校准函数、工具成功率
"""

import sys
import json
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.eval.schemas import EvalScore, EvalReport
from app.eval.evaluator import Evaluator
from app.eval import realtime, alerts, metrics


def _client_responding(content):
    FakeMsg = type("FakeMsg", (), {"content": content})
    FakeChoice = type("FakeChoice", (), {"message": FakeMsg()})
    FakeResp = type("FakeResp", (), {"choices": [FakeChoice()]})

    class Completions:
        def __init__(self):
            self.prompts = []

        async def create(self, **kwargs):
            self.prompts.append(kwargs["messages"][0]["content"])
            return FakeResp()

    completions = Completions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _evaluator_with_client(client):
    ev = Evaluator.__new__(Evaluator)
    ev.model = "fake-model"
    ev.client = client
    return ev


@pytest.mark.asyncio
async def test_evaluate_with_contexts_adds_faithfulness():
    client, completions = _client_responding(
        '{"relevance": 5, "completeness": 4, "usefulness": 5, '
        '"faithfulness": 2, "explanation": "编造了内容"}'
    )
    ev = _evaluator_with_client(client)
    score = await ev.evaluate("q", "a", contexts="参考文档：行权期 3 年")
    assert score is not None
    assert score.faithfulness == 2
    prompt = completions.prompts[0]
    assert "忠实度（Faithfulness）" in prompt
    assert "参考文档" in prompt


@pytest.mark.asyncio
async def test_evaluate_without_contexts_no_faithfulness():
    client, completions = _client_responding(
        '{"relevance": 5, "completeness": 4, "usefulness": 5, "explanation": "ok"}'
    )
    ev = _evaluator_with_client(client)
    score = await ev.evaluate("q", "a")
    assert score.faithfulness is None
    assert "忠实度" not in completions.prompts[0]


def test_parse_faithfulness():
    raw = ('{"relevance": 4, "completeness": 3, "usefulness": 5, '
           '"faithfulness": 1, "explanation": "幻觉"}')
    score = Evaluator._parse(raw, "q", "a")
    assert score.faithfulness == 1
    # 忠实度字段格式异常：整条评分不因此作废
    bad = ('{"relevance": 4, "completeness": 3, "usefulness": 5, '
           '"faithfulness": "two", "explanation": "异常"}')
    score2 = Evaluator._parse(bad, "q", "a")
    assert score2 is not None
    assert score2.faithfulness is None
    assert score2.relevance == 4


def test_parse_truncated_json_fallback():
    """评测输出被 max_tokens 截断（缺右花括号）→ 逐字段提取退化，不整条作废"""
    raw = ('{"relevance": 5, "completeness": 4, "usefulness": 3, '
           '"faithfulness": 5, "explanation": "回答直接针对问题，'
           '但信息可以更完整')
    score = Evaluator._parse(raw, "q", "a")
    assert score is not None
    assert score.relevance == 5
    assert score.completeness == 4
    assert score.usefulness == 3
    assert score.faithfulness == 5
    assert "回答直接针对问题" in score.explanation


def test_parse_json_with_surrounding_text():
    """JSON 前后混了说明文字 → 花括号子串解析"""
    raw = '评分结果如下：{"relevance": 4, "completeness": 3, "usefulness": 4, "explanation": "ok"} 完毕'
    score = Evaluator._parse(raw, "q", "a")
    assert score is not None
    assert score.relevance == 4


@pytest.mark.asyncio
async def test_compare_pairwise():
    client, completions = _client_responding(
        '{"winner": "a", "reason": "A 更完整且忠实"}'
    )
    ev = _evaluator_with_client(client)
    result = await ev.compare("q", "answer A", "answer B", contexts="文档")
    assert result == {"winner": "a", "reason": "A 更完整且忠实"}
    assert "对比评测员" in completions.prompts[0]


def test_parse_compare_fallback_text():
    assert Evaluator._parse_compare("回答 B 更好，因为...") == {
        "winner": "b", "reason": "回答 B 更好，因为..."
    }
    assert Evaluator._parse_compare("乱码") is None
    assert Evaluator._parse_compare('{"winner": "x", "reason": "r"}') == {
        "winner": "tie", "reason": "r"
    }


def test_runner_report_faithfulness_avg():
    from app.eval.runner import EvalRunner
    scores = [
        EvalScore(query="q1", answer="a1", relevance=4, completeness=4,
                  usefulness=4, faithfulness=5, explanation="", timestamp=""),
        EvalScore(query="q2", answer="a2", relevance=4, completeness=4,
                  usefulness=4, faithfulness=3, explanation="", timestamp=""),
        EvalScore(query="q3", answer="a3", relevance=4, completeness=4,
                  usefulness=4, faithfulness=None, explanation="", timestamp=""),
    ]
    report = EvalRunner._build_report(scores)
    assert report.avg_faithfulness == 4.0  # (5+3)/2，None 不参与
    assert report.total_samples == 3

    report_none = EvalRunner._build_report(
        [EvalScore(query="q", answer="a", relevance=4, completeness=4,
                   usefulness=4, faithfulness=None, explanation="", timestamp="")]
    )
    assert report_none.avg_faithfulness is None


def test_record_keeps_faithfulness(tmp_path, monkeypatch):
    monkeypatch.setattr(realtime, "SCORES_PATH", str(tmp_path / "s.json"))
    realtime.record("q", "a", 4, 4, 5, "", faithfulness=2)
    record = realtime.load()[-1]
    assert record["faithfulness"] == 2
    realtime.record("q2", "a2", 4, 4, 5, "")  # 兼容缺省
    assert realtime.load()[-1]["faithfulness"] is None


def test_alert_low_faithfulness_triggers(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "ALERTS_PATH", str(tmp_path / "a.json"))
    # 其他维度都高，只有忠实度低 → 仍要告警（幻觉是最致命失败模式）
    assert alerts.check_and_alert(
        "q", "a", relevance=5, completeness=5, usefulness=5,
        explanation="", faithfulness=1,
    ) is True
    assert alerts.check_and_alert(
        "q", "a", relevance=5, completeness=5, usefulness=5,
        explanation="", faithfulness=5,
    ) is False


def test_metrics_tool_success_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRICS_PATH", str(tmp_path / "m.jsonl"))
    metrics.record_request("q1", 10, tool_calls=2, tool_failures=0)
    metrics.record_request("q2", 10, tool_calls=3, tool_failures=1)
    metrics.record_request("q3", 10)  # 无工具调用
    s = metrics.summary()
    assert s["tool_calls_total"] == 5
    assert s["tool_success_rate"] == 1 - 1 / 5


def test_precision_at_k_from_eval_script():
    from scripts.eval_testcorpus_v2 import precision_at_k
    results = [
        [
            {"content": "答案在第一条", "parent_content": ""},
            {"content": "无关内容", "parent_content": ""},
            {"content": "答案在第三条", "parent_content": ""},
        ]
    ]
    gts = ["答案在第一条"]
    # P@1 = 1/1；P@3 = 1/3（top-3 中只有 1 条相关）
    assert precision_at_k(results, gts, 1) == 1.0
    assert abs(precision_at_k(results, gts, 3) - 1 / 3) < 1e-9


def test_calibration_cohen_kappa():
    from scripts.eval_judge_calibration import cohen_kappa, _agreement_report
    # 完全一致 → kappa = 1
    assert cohen_kappa([5, 4, 3], [5, 4, 3]) == 1.0
    # 完全不一致 → kappa = -1
    assert cohen_kappa([1, 1, 2, 2], [2, 2, 1, 1]) == -1.0
    # 空输入安全
    assert cohen_kappa([], []) == 0.0

    manual = [
        {"query": "q", "answer": "a", "relevance": 5, "completeness": 4,
         "usefulness": 5, "faithfulness": 5},
        {"query": "q2", "answer": "a2", "relevance": 4, "completeness": 4,
         "usefulness": 4, "faithfulness": 4},
    ]
    judge = [
        {"query": "q", "answer": "a", "relevance": 5, "completeness": 4,
         "usefulness": 5, "faithfulness": 5},
        {"query": "q2", "answer": "a2", "relevance": 4, "completeness": 4,
         "usefulness": 4, "faithfulness": 4},
    ]
    rep = _agreement_report(manual, judge)
    assert rep["relevance"]["kappa"] == 1.0
    assert rep["faithfulness"]["samples"] == 2
