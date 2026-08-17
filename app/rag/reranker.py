# -*- coding: utf-8 -*-
"""重排序模块

提供基于 Cross-Encoder 的重排序功能。
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import yaml


class Reranker:
    """重排序器"""

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """
        初始化重排序器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.rerank_config = self.config.get("rerank", {})
        self.model = self._init_model()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def _init_model(self) -> CrossEncoder:
        """初始化重排序模型"""
        model_name = self.rerank_config.get("model_name", "BAAI/bge-reranker-base")
        device = self.rerank_config.get("device", "cpu")
        
        print(f"正在加载 Rerank 模型: {model_name}...")
        model = CrossEncoder(model_name, device=device)
        print("Rerank 模型加载完成")
        return model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3,
        threshold: Optional[float] = None,
        autocut: bool = False,
        drop_ratio: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待排序文档列表
            top_k: 返回前 k 个结果
            threshold: 相关性阈值，低于阈值的文档被剔除（None 不启用）
            autocut: 是否启用"分数悬崖"动态截断——重排后相邻文档分数相对落差
                超过 drop_ratio 时，在落差处截断（替代固定 top_k 的噪声注入）
            drop_ratio: 相对落差阈值（如 0.3 = 后一名分数比前一名低 30% 以上
                视为断层，断层之后的内容大概率是噪声/弱相关）

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 准备输入数据
        pairs = []
        for doc in documents:
            content = doc.get("content", "")
            pairs.append([query, content])
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 将分数添加到文档中
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        
        # 按重排序分数排序
        sorted_docs = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # 相关性阈值过滤：剔除明显不相关的文档，避免噪声进 LLM
        if threshold is not None:
            sorted_docs = [
                d for d in sorted_docs
                if d["rerank_score"] >= threshold
            ]

        # 分数悬崖动态截断：找最大相对落差处截断（至少保留 1 条）
        if autocut and len(sorted_docs) > 1:
            scores = [float(d.get("rerank_score", 0.0)) for d in sorted_docs]
            max_drop = 0.0
            cut_idx = None
            for i in range(1, len(scores)):
                prev = scores[i - 1]
                cur = scores[i]
                if prev <= 0:
                    break
                rel_drop = (prev - cur) / prev
                if rel_drop > max_drop:
                    max_drop = rel_drop
                    cut_idx = i
            if cut_idx is not None and max_drop >= drop_ratio:
                sorted_docs = sorted_docs[:cut_idx]

        return sorted_docs[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        带阈值的重排序

        Args:
            query: 查询文本
            documents: 待排序文档列表
            top_k: 返回前 k 个结果
            threshold: 相关性分数阈值

        Returns:
            重排序后的文档列表（只返回超过阈值的）
        """
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # 过滤低于阈值的结果
        filtered = [doc for doc in reranked if doc["rerank_score"] >= threshold]
        
        return filtered[:top_k]
