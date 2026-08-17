# -*- coding: utf-8 -*-
"""MMR（Maximum Marginal Relevance）去冗余

重排后 top-k 可能语义重复（多条讲同一件事）。MMR 在选择时平衡
「与查询的相关性」和「与已选文档的差异」：

    score(i) = λ · sim(q, i) - (1 - λ) · max(sim(i, j))   j ∈ 已选

λ 越大越偏向相关性，越小越偏向多样性（默认 0.7）。
query_scores 输入后会归一化到 [0,1]，与文档间余弦（0-1）可比。
"""

from typing import List, Optional

import numpy as np


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_rerank(
    doc_embs: List,
    query_scores: List[float],
    lambda_: float = 0.7,
    top_k: Optional[int] = None,
) -> List[int]:
    """按 MMR 返回选择顺序（索引列表）。

    Args:
        doc_embs: 文档向量（与 query_scores 一一对应）
        query_scores: 各文档与查询的相关性分数（重排分数或相似度）
        lambda_: 相关性权重，越大越相关、越小越多样
        top_k: 返回数量；None 表示全部

    Returns:
        按选择顺序排列的文档索引列表
    """
    n = len(doc_embs)
    if n == 0:
        return []

    scores = np.asarray(query_scores, dtype=float)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    selected: List[int] = []
    remaining = list(range(n))
    while remaining and (top_k is None or len(selected) < top_k):
        best_idx, best_val = -1, float("-inf")
        for i in remaining:
            redundancy = max(
                (_cosine(doc_embs[i], doc_embs[j]) for j in selected),
                default=0.0,
            )
            val = lambda_ * float(scores[i]) - (1 - lambda_) * redundancy
            if val > best_val:
                best_val, best_idx = val, i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected
