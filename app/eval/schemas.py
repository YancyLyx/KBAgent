# -*- coding: utf-8 -*-
"""评测数据模型"""

from pydantic import BaseModel
from typing import List, Optional


class EvalScore(BaseModel):
    """单条对话的评分"""
    query: str
    answer: str
    relevance: int       # 相关性 1-5
    completeness: int    # 完整性 1-5
    usefulness: int      # 有用性 1-5
    explanation: str     # 评分说明
    timestamp: str


class EvalReport(BaseModel):
    """聚合评测报告"""
    total_samples: int
    avg_relevance: float
    avg_completeness: float
    avg_usefulness: float
    avg_total: float
    samples: List[EvalScore]
    run_id: str
    created_at: str


class EvalSummary(BaseModel):
    """评测摘要（给前端展示）"""
    avg_scores: dict
    sample_count: int
    last_run: Optional[str] = None
    trend: Optional[str] = None  # "improved" / "declined" / "stable"
