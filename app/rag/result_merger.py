# -*- coding: utf-8 -*-
"""多路检索结果合并（产品路径与评测共用）。

背景：多查询分解 / Self-RAG 补检会产生多路检索结果（每路一组父块）。
旧实现是「content 去重 + 先到先得」（按最早命中的顺序保留），跨文档信息容易
丢失——诊断证据：2docs 上多查询分解后 MRR 0.493→0.361、Recall@10 0.792→0.667。

这里统一用 RRF（Reciprocal Rank Fusion）按排名融合：
- 每路候选放大后参与（fetch 更多，给每篇文档浮上来的机会）；
- 排名跨路可比（rerank 分数跨 query 不可直接比）；
- 两路都命中的父块排名自然上升（1/(k+rank) 累加）。

使用方：agent_manager._search_with_expansion / _search_with_self_rag，以及
评测脚本（scripts/milvus/diag_merge_strategy.py）。
"""

from typing import Any, Dict, List

DEFAULT_RRF_K = 60


def _key_of(doc: Dict[str, Any]) -> str:
    """融合键：父块 id 优先，其次 content 前缀（保证跨路同一父块能累加）。"""
    k = str(doc.get("id") or doc.get("parent_id") or "")
    if k:
        return k
    return str(doc.get("content", ""))[:80]


def rrf_merge(
    results_per_query: List[List[Dict[str, Any]]],
    top_k: int = 3,
    k: int = DEFAULT_RRF_K,
) -> List[Dict[str, Any]]:
    """按 RRF 融合多路检索结果，返回排序后的 top_k 父块。

    Args:
        results_per_query: 每路 retrieve 的结果列表（已含 rerank_score 等字段）
        top_k: 最终返回条数
        k: RRF 平滑常数（与项目现有 RRF k=60 一致）

    Returns:
        融合后的结果列表（保留第一路的原始字段，附加 merge_score）
    """
    acc: Dict[str, Dict[str, Any]] = {}
    for results in results_per_query:
        if not results:
            continue
        for rank, doc in enumerate(results):
            key = _key_of(doc)
            if not key:
                continue
            if key not in acc:
                acc[key] = dict(doc)
                acc[key]["merge_score"] = 0.0
            acc[key]["merge_score"] += 1.0 / (k + rank + 1)

    merged = sorted(acc.values(), key=lambda x: x["merge_score"], reverse=True)
    return merged[:top_k]
