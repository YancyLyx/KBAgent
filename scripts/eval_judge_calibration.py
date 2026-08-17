#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-as-Judge 一致性校准脚本

第三篇文章的加分项：LLM Judge 不是黑箱，必须和人工标注做一致性校准。
本脚本做两类校准：

1. 人工标注一致性（Human-Judge Agreement）：
   对同一批样本，人工按 Rubric 打分、LLM-as-Judge 打分，
   逐维度算 Cohen's Kappa（分类一致性）与 Pearson 相关系数（分数相关性）。
   业界经验阈值：Kappa ≥ 0.6-0.7 才算校准通过；不通过则换模型或调 Rubric。

2. Judge 稳定性（Test-Retest）：
   同一批样本用同一 Judge 跑两遍，算两次打分的一致比例与 Kappa，
   衡量 Judge 自身的可复现性（LLM 生成有随机性，稳定性是校准的前提）。

用法：
  python scripts/eval_judge_calibration.py \
      --labels data/eval_judge/manual_labels.json \
      [--contexts data/eval_judge/contexts.json] \
      [--retest]

manual_labels.json 格式（人工按 Rubric 打分，relevance/completeness/usefulness
为 1-5，faithfulness 可选——提供参考文档时建议标注）：
[
  {"query": "行权期是多久", "answer": "3 年", "relevance": 5, "completeness": 4,
   "usefulness": 5, "faithfulness": 5}
]

contexts.json 格式（可选，与 labels 一一对应，供忠实度评测）：
["参考文档文本 1", "参考文档文本 2", ...]

说明：校准数字依赖人工标注样本（建议 ≥30 条），未标注时本脚本给出提示，
不会假装产出校准结论。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.eval.evaluator import Evaluator


def cohen_kappa(a: List[int], b: List[int]) -> float:
    """Cohen's Kappa：两个评分者对同一批样本的分类一致性（-1 ~ 1）"""
    n = len(a)
    if n == 0:
        return 0.0
    classes = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = {c: a.count(c) / n for c in classes}
    pb = {c: b.count(c) / n for c in classes}
    expected = sum(pa[c] * pb[c] for c in classes)
    if expected >= 1.0:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def pearson(a: List[int], b: List[int]) -> Optional[float]:
    if len(a) < 2:
        return None
    try:
        return round(float(np.corrcoef(a, b)[0, 1]), 4)
    except Exception:
        return None


def _dim_values(labels: List[dict], key: str) -> List[int]:
    return [int(item[key]) for item in labels if item.get(key) is not None]


def _align(manual: List[int], judge: List[int]) -> tuple:
    """按 manual 非空索引对齐（judge 可能缺样本）"""
    pairs = [
        (m, j) for m, j in zip(manual, judge)
        if m is not None and j is not None
    ]
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _agreement_report(manual: List[dict], judge: List[dict]) -> Dict:
    report = {}
    for dim in ("relevance", "completeness", "usefulness", "faithfulness"):
        m_vals = [item.get(dim) for item in manual]
        j_vals = [item.get(dim) for item in judge]
        m_aligned, j_aligned = _align(m_vals, j_vals)
        if len(m_aligned) < 2:
            report[dim] = {"samples": len(m_aligned), "kappa": None, "pearson": None}
            continue
        kappa = cohen_kappa(m_aligned, j_aligned)
        corr = pearson(m_aligned, j_aligned)
        exact = sum(1 for x, y in zip(m_aligned, j_aligned) if x == y) / len(m_aligned)
        report[dim] = {
            "samples": len(m_aligned),
            "kappa": kappa,
            "pearson": corr,
            "exact_match_rate": round(exact, 4),
        }
    return report


def _stability_report(scores_a: List[dict], scores_b: List[dict]) -> Dict:
    """两次 Judge 打分的一致性（稳定性）"""
    report = {}
    for dim in ("relevance", "completeness", "usefulness", "faithfulness"):
        a = [s.get(dim) for s in scores_a]
        b = [s.get(dim) for s in scores_b]
        m_aligned, j_aligned = _align(a, b)
        if len(m_aligned) < 2:
            report[dim] = {"samples": len(m_aligned), "kappa": None}
            continue
        within_1 = sum(
            1 for x, y in zip(m_aligned, j_aligned) if abs(x - y) <= 1
        ) / len(m_aligned)
        report[dim] = {
            "samples": len(m_aligned),
            "kappa": cohen_kappa(m_aligned, j_aligned),
            "within_1_rate": round(within_1, 4),
        }
    return report


async def _judge_all(labels: List[dict], contexts: Optional[List[str]]) -> List[dict]:
    evaluator = Evaluator()
    scores = []
    for i, item in enumerate(labels):
        ctx = contexts[i] if contexts and i < len(contexts) else None
        score = await evaluator.evaluate(
            item.get("query", ""), item.get("answer", ""), contexts=ctx
        )
        if score is None:
            continue
        scores.append({
            "query": score.query,
            "answer": score.answer,
            "relevance": score.relevance,
            "completeness": score.completeness,
            "usefulness": score.usefulness,
            "faithfulness": score.faithfulness,
        })
    return scores


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge 一致性校准")
    parser.add_argument("--labels", required=True, help="人工标注 JSON 路径")
    parser.add_argument("--contexts", default="", help="参考文档 JSON 路径（可选）")
    parser.add_argument("--retest", action="store_true", help="额外跑第二遍做稳定性校准")
    args = parser.parse_args()

    labels_path = project_root / args.labels
    if not labels_path.exists():
        print(
            f"[校准] 未找到人工标注文件 {labels_path}\n"
            "请先按脚本头注释的格式准备 manual_labels.json（建议 ≥30 条），"
            "再运行校准——校准数字不能假装，没有标注就不产出结论。"
        )
        return
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    contexts = None
    if args.contexts:
        with open(project_root / args.contexts, encoding="utf-8") as f:
            contexts = json.load(f)

    print(f"[校准] 样本数：{len(labels)}，开始 LLM-as-Judge 打分（第 1 遍）...")
    judge_1 = await _judge_all(labels, contexts)
    print(f"[校准] 第 1 遍有效评分：{len(judge_1)} 条")

    report = {
        "sample_count": len(labels),
        "judge_samples": len(judge_1),
        "human_judge_agreement": _agreement_report(labels, judge_1),
    }

    if args.retest and judge_1:
        print("[校准] 运行第 2 遍（稳定性/Test-Retest）...")
        judge_2 = await _judge_all(labels, contexts)
        report["judge_stability"] = _stability_report(judge_1, judge_2)

    out_dir = project_root / "data" / "eval_judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[校准] 报告已保存：{out_path}")


if __name__ == "__main__":
    asyncio.run(main())
