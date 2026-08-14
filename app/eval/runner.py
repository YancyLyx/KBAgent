# -*- coding: utf-8 -*-
"""评测运行器

批量运行 LLM-as-Judge 评测，生成聚合报告，支持回归对比。
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter

from .schemas import EvalScore, EvalReport
from .evaluator import Evaluator


EVAL_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval_results.json",
)


class EvalRunner:
    """评测运行器"""

    def __init__(self, evaluator: Optional[Evaluator] = None):
        self.evaluator = evaluator or Evaluator()
        self._ensure_data_dir()

    @staticmethod
    def _ensure_data_dir():
        os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)

    # ------------------------------------------------------------------
    # 运行评测
    # ------------------------------------------------------------------

    async def run(self, conversations: List[Dict]) -> EvalReport:
        """对一批对话运行评测"""
        scores: List[EvalScore] = []

        for conv in conversations:
            query = conv.get("message") or conv.get("query", "")
            answer = conv.get("answer", "")
            if not answer:
                continue

            score = await self.evaluator.evaluate(query, answer)
            if score:
                score.timestamp = conv.get("timestamp", datetime.now().isoformat())
                scores.append(score)

        report = self._build_report(scores)
        self._save(report)
        return report

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report(scores: List[EvalScore]) -> EvalReport:
        total = len(scores)
        if total == 0:
            return EvalReport(
                total_samples=0, avg_relevance=0.0, avg_completeness=0.0,
                avg_usefulness=0.0, avg_total=0.0, samples=[],
                run_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                created_at=datetime.now().isoformat(),
            )

        avg_rel = sum(s.relevance for s in scores) / total
        avg_com = sum(s.completeness for s in scores) / total
        avg_use = sum(s.usefulness for s in scores) / total

        return EvalReport(
            total_samples=total,
            avg_relevance=round(avg_rel, 2),
            avg_completeness=round(avg_com, 2),
            avg_usefulness=round(avg_use, 2),
            avg_total=round((avg_rel + avg_com + avg_use) / 3, 2),
            samples=scores,
            run_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save(self, report: EvalReport):
        history = self._load_history()
        history.append(report.model_dump() if hasattr(report, 'model_dump') else report.dict())
        # 只保留最近 10 次
        history = history[-10:]
        with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _load_history(self) -> list:
        if not os.path.exists(EVAL_RESULTS_PATH):
            return []
        try:
            with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_latest_report(self) -> Optional[EvalReport]:
        history = self._load_history()
        if not history:
            return None
        latest = history[-1]
        return EvalReport(**latest)

    def get_history(self) -> list:
        return self._load_history()
