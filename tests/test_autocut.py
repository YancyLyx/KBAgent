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


def test_autocut_uses_cliff_inside_topk_not_global():
    """top_k 内断层不能被更靠后的更大落差掩盖：
    [0.9, 0.88, 0.5, 0.1] top_k=3 → 前 3 名内 43% 落差生效，截到 2 条
    （旧实现找全局最大落差 0.5→0.1=80% 会错误返回 3 条）"""
    r = _reranker_with_scores([0.9, 0.88, 0.5, 0.1])
    result = r.rerank("q", _docs(["A", "B", "C", "D"]), top_k=3, autocut=True)
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


def test_pipeline_defaults_all_rerank_enhancements_enabled():
    """重排层增强默认全开（读配置）：不传任何增强参数，
    阈值过滤 + Autocut + MMR 全部生效"""
    from app.rag.rag_pipeline import RAGPipeline

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.reranker = _reranker_with_scores([0.9, 0.88, 0.5, 0.1])
    pipe.autocut_enabled = True
    pipe.autocut_drop_ratio = 0.3
    pipe.min_rerank_score_default = 0.5   # config score_threshold
    pipe.diversity_rerank_enabled = True   # config diversity_rerank
    pipe.vector_store = SimpleNamespace(
        # 每个文档一个可区分的向量，MMR 不会误判重复
        encode_text=lambda texts: np.array(
            [[float(i), 0.0] for i in range(len(texts))], dtype=float
        )
    )
    pipe.retriever = SimpleNamespace(
        parent_child_search=lambda *a, **k: [
            {"content": "A", "id": "1"},
            {"content": "B", "id": "2"},
            {"content": "C", "id": "3"},
            {"content": "D", "id": "4"},
        ]
    )
    # 不传增强参数（模拟线上工具路径）：
    # threshold=0.5 过滤 0.1 → [0.9, 0.88, 0.5]
    # autocut top-3 内 0.88→0.5 落差 43% → 截到 [0.9, 0.88]
    # MMR 在 2 条上（向量可区分，顺序不变）
    result = pipe.retrieve("q", top_k=3, use_rerank=True)
    assert [d["content"] for d in result] == ["A", "B"]
