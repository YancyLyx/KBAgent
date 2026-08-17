# -*- coding: utf-8 -*-
"""Autocut 分数悬崖动态截断测试"""

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from app.rag.reranker import Reranker


def _reranker_with_scores(scores):
    r = Reranker.__new__(Reranker)
    r.model = SimpleNamespace(predict=lambda pairs: np.array(scores, dtype=float))
    return r


def _docs(contents):
    return [{"content": c} for c in contents]


def test_autocut_cuts_at_cliff():
    """分数 [0.9, 0.88, 0.5]：0.88→0.5 落差 43% > 30%，截到 2 条"""
    r = _reranker_with_scores([0.9, 0.88, 0.5])
    result = r.rerank("q", _docs(["A", "B", "C"]), top_k=3, autocut=True)
    assert [d["content"] for d in result] == ["A", "B"]


def test_autocut_keeps_all_without_cliff():
    """分数均匀下降，无 >30% 断层 → 全部保留"""
    r = _reranker_with_scores([0.9, 0.85, 0.8])
    result = r.rerank("q", _docs(["A", "B", "C"]), top_k=3, autocut=True)
    assert len(result) == 3


def test_autocut_keeps_at_least_one():
    """悬崖在第一处 [0.9, 0.1]：也要保留 1 条，不能返回空"""
    r = _reranker_with_scores([0.9, 0.1])
    result = r.rerank("q", _docs(["A", "B"]), top_k=3, autocut=True)
    assert [d["content"] for d in result] == ["A"]


def test_autocut_respects_custom_drop_ratio():
    """落差 20% 但 drop_ratio=0.5 → 不截断；drop_ratio=0.1 → 截断"""
    r = _reranker_with_scores([0.9, 0.72])
    kept = r.rerank("q", _docs(["A", "B"]), top_k=3, autocut=True, drop_ratio=0.5)
    assert len(kept) == 2
    cut = r.rerank("q", _docs(["A", "B"]), top_k=3, autocut=True, drop_ratio=0.1)
    assert len(cut) == 1


def test_autocut_disabled_by_default():
    r = _reranker_with_scores([0.9, 0.2, 0.1])
    result = r.rerank("q", _docs(["A", "B", "C"]), top_k=3)
    assert len(result) == 3


def test_autocut_combined_with_threshold():
    """threshold 先过滤低分，autocut 再找悬崖：两者叠加"""
    r = _reranker_with_scores([0.9, 0.88, 0.5, 0.1])
    result = r.rerank(
        "q", _docs(["A", "B", "C", "D"]), top_k=3, threshold=0.45, autocut=True
    )
    assert [d["content"] for d in result] == ["A", "B"]


def test_pipeline_passes_autocut_flag():
    """RAGPipeline.retrieve 把 autocut 传给 rerank（显式 True/False 覆盖配置）"""
    from app.rag.rag_pipeline import RAGPipeline

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.reranker = _reranker_with_scores([0.9, 0.5, 0.1])
    pipe.autocut_enabled = False
    pipe.autocut_drop_ratio = 0.3
    pipe.retriever = SimpleNamespace(
        parent_child_search=lambda *a, **k: [
            {"content": "A", "id": "1"},
            {"content": "B", "id": "2"},
            {"content": "C", "id": "3"},
        ]
    )
    result = pipe.retrieve("q", top_k=3, use_rerank=True, autocut=True)
    assert [d["content"] for d in result] == ["A", "B"]

    # 显式 False 覆盖配置开启
    pipe.autocut_enabled = True
    result = pipe.retrieve("q", top_k=3, use_rerank=True, autocut=False)
    assert len(result) == 3
