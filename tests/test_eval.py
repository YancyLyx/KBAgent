# -*- coding: utf-8 -*-
"""测试 Phase 3：LLM-as-Judge 评测"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.eval.schemas import EvalScore, EvalReport
from app.eval.runner import EvalRunner
from app.eval.evaluator import Evaluator
import json
import asyncio


def test_eval_schemas():
    """评测数据模型"""
    score = EvalScore(
        query="测试问题", answer="测试回答",
        relevance=4, completeness=3, usefulness=4,
        explanation="评分说明", timestamp="2024-01-01",
    )
    assert score.relevance == 4
    assert score.usefulness == 4

    report = EvalReport(
        total_samples=10,
        avg_relevance=4.0,
        avg_completeness=3.5,
        avg_usefulness=4.2,
        avg_total=3.9,
        samples=[score],
        run_id="eval_test",
        created_at="2024-01-01",
    )
    assert report.avg_total == 3.9
    print("✓ test_eval_schemas")


def test_eval_parse():
    """Evaluator JSON 解析（不调 LLM）"""
    raw = '''{"relevance": 4, "completeness": 3, "usefulness": 5, "explanation": "回答很好"}'''
    score = Evaluator._parse(raw, "问题", "回答")
    assert score is not None
    assert score.relevance == 4
    assert score.completeness == 3
    assert score.usefulness == 5

    # Markdown code block 包裹
    raw2 = '```json\n{"relevance": 5, "completeness": 4, "usefulness": 5, "explanation": "完美"}\n```'
    score2 = Evaluator._parse(raw2, "问题", "回答")
    assert score2 is not None
    assert score2.relevance == 5

    # 无效 JSON
    raw3 = "不是 JSON"
    score3 = Evaluator._parse(raw3, "问题", "回答")
    assert score3 is None

    print("✓ test_eval_parse")


def test_eval_runner():
    """EvalRunner 批量运行（mock evaluator）"""
    runner = EvalRunner()

    # Mock evaluator
    class MockEval:
        async def evaluate(self, query, answer):
            from datetime import datetime
            return EvalScore(
                query=query, answer=answer,
                relevance=4, completeness=3, usefulness=5,
                explanation="OK", timestamp=datetime.now().isoformat(),
            )

    runner.evaluator = MockEval()

    conversations = [
        {"message": "问题1", "answer": "回答1"},
        {"message": "问题2", "answer": "回答2"},
        {"message": "问题3", "answer": ""},  # 无回答，跳过
    ]

    report = asyncio.run(runner.run(conversations))
    assert report.total_samples == 2
    assert report.avg_relevance == 4.0
    assert report.avg_usefulness == 5.0

    # 验证持久化
    history = runner.get_history()
    assert len(history) >= 1
    print("✓ test_eval_runner")


if __name__ == "__main__":
    print("=== Phase 3 Tests ===")
    test_eval_schemas()
    test_eval_parse()
    test_eval_runner()
    print("\n✅ Phase 3 全部通过")
