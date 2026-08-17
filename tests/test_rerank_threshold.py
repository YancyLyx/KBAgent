# -*- coding: utf-8 -*-
"""相关性阈值过滤测试：重排后剔除低分文档"""

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from app.rag.reranker import Reranker


def _reranker_with_scores(scores):
    """用 __new__ 绕过模型加载，注入 fake model.predict"""
    r = Reranker.__new__(Reranker)
    r.model = SimpleNamespace(predict=lambda pairs: np.array(scores, dtype=float))
    return r


def _docs(contents):
    return [{"content": c} for c in contents]


def test_threshold_filters_low_scores():
    r = _reranker_with_scores([0.9, 0.3, 0.6])
    docs = _docs(["高相关", "低相关", "中相关"])
    result = r.rerank("q", docs, top_k=3, threshold=0.5)
    assert [d["content"] for d in result] == ["高相关", "中相关"]


def test_threshold_none_keeps_all():
    r = _reranker_with_scores([0.9, 0.3, 0.6])
    docs = _docs(["高相关", "低相关", "中相关"])
    result = r.rerank("q", docs, top_k=3)
    assert len(result) == 3


def test_threshold_with_top_k():
    r = _reranker_with_scores([0.9, 0.8, 0.1])
    docs = _docs(["A", "B", "低分"])
    result = r.rerank("q", docs, top_k=1, threshold=0.5)
    assert [d["content"] for d in result] == ["A"]


def test_threshold_empty_result():
    r = _reranker_with_scores([0.1, 0.2])
    docs = _docs(["A", "B"])
    result = r.rerank("q", docs, top_k=3, threshold=0.9)
    assert result == []


def test_pipeline_passes_threshold():
    """RAGPipeline.retrieve 把 min_rerank_score 传给 rerank"""
    from app.rag.rag_pipeline import RAGPipeline

    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.reranker = _reranker_with_scores([0.9, 0.2])
    pipe.autocut_enabled = False
    pipe.autocut_drop_ratio = 0.3
    pipe.retriever = SimpleNamespace(
        parent_child_search=lambda *a, **k: [
            {"content": "A", "id": "1"},
            {"content": "B", "id": "2"},
        ]
    )
    result = pipe.retrieve("q", top_k=2, use_rerank=True, min_rerank_score=0.5)
    assert [d["content"] for d in result] == ["A"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
