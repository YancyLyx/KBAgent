# -*- coding: utf-8 -*-
"""MMR 去冗余测试：纯函数 + RAGPipeline 集成"""

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from app.rag.mmr import mmr_rerank


def test_mmr_prefers_diverse():
    """A/B 语义相同、C 不同：MMR top-2 应选 A + C 而不是 A + B"""
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.9, 0.1, 0.0])     # 与 A 高度相似（余弦≈0.995）
    emb_c = np.array([0.0, 0.0, 1.0])     # 与 A/B 不相似
    # 相关性：A 最高，B/C 接近；λ=0.5 时多样性惩罚能压过 B 的相关性优势
    scores = [0.9, 0.6, 0.55]

    order = mmr_rerank([emb_a, emb_b, emb_c], scores, lambda_=0.5, top_k=2)
    assert order[0] == 0          # A 相关性最高，先选
    assert order[1] == 2          # C 虽然相关性低，但多样性高 → 胜过冗余的 B


def test_mmr_high_lambda_prefers_relevance():
    """λ 接近 1：完全偏向相关性，B 会先于 C"""
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.99, 0.01, 0.0])
    emb_c = np.array([0.0, 0.0, 1.0])
    scores = [0.9, 0.8, 0.5]
    order = mmr_rerank([emb_a, emb_b, emb_c], scores, lambda_=0.99, top_k=2)
    assert order[:2] == [0, 1]


def test_mmr_empty_and_single():
    assert mmr_rerank([], []) == []
    order = mmr_rerank([np.array([1.0, 0.0])], [0.7], top_k=1)
    assert order == [0]


def test_mmr_normalizes_scores():
    """分数范围不同也能正常归一化比较"""
    emb_a = np.array([1.0, 0.0])
    emb_b = np.array([0.0, 1.0])
    # 分数差很大：归一化后仍选相关性最高的 A
    order = mmr_rerank([emb_a, emb_b], [5.0, 0.1], lambda_=0.7, top_k=1)
    assert order == [0]


def test_pipeline_apply_mmr_integration():
    """RAGPipeline._apply_mmr：用 __new__ 绕过模型加载，注入 fake encode"""
    from app.rag.rag_pipeline import RAGPipeline

    pipe = RAGPipeline.__new__(RAGPipeline)
    fake_embs = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pipe.vector_store = SimpleNamespace(
        encode_text=lambda texts: fake_embs
    )
    docs = [
        {"content": "A 的内容", "rerank_score": 0.9},
        {"content": "B 的内容（与 A 重复）", "rerank_score": 0.6},
        {"content": "C 的内容（不同主题）", "rerank_score": 0.55},
    ]
    result = pipe._apply_mmr("q", docs, top_k=2, lambda_=0.5)
    assert result[0]["content"] == "A 的内容"
    assert result[1]["content"] == "C 的内容（不同主题）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
