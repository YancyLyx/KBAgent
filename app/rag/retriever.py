# -*- coding: utf-8 -*-
"""检索器模块

提供向量检索和关键词检索功能，支持混合检索。
"""

from typing import List, Dict, Any, Optional
from .vector_store import VectorStore


class Retriever:
    """检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        config_path: str = "config/rag_config.yaml"
    ):
        """
        初始化检索器

        Args:
            vector_store: 向量存储实例
            config_path: 配置文件路径
        """
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.vector_store = vector_store
        self.retrieval_config = self.config.get("retrieval_strategy", {})
        self.parent_child_config = self.config.get("parent_child", {})
        self.rrf_k = self.retrieval_config.get("rrf_k", 60)

    def vector_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = self.retrieval_config.get("vector_top_k", 10)
        
        results = self.vector_store.search(query, top_k=top_k, filters=filters)
        
        # 添加检索类型标记
        for result in results:
            result["search_type"] = "vector"
        
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索（向量检索 + BM25 关键词检索）

        同时执行向量检索和关键词检索，按权重合并分数后排序输出。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 元数据过滤条件（如 {"tag": "文献"}）

        Returns:
            合并排序后的检索结果
        """
        if top_k is None:
            top_k = self.retrieval_config.get("rerank_top_k", 3)

        # Milvus 后端：单次调用 Milvus 内置 RRFRanker，跳过手写 RRF（严禁二次 RRF）
        if getattr(self.vector_store, "backend", "chroma") == "milvus":
            return self.vector_store.hybrid_search(query, top_k=top_k, filters=filters)

        # Chroma 后端：手写 RRF（k=60），逻辑保持不变
        # 向量检索
        vector_results = self.vector_search(query, top_k=top_k * 2, filters=filters)
        # BM25 关键词检索
        keyword_results = self.vector_store.bm25_search(query, top_k=top_k * 2, filters=filters)

        # Reciprocal Rank Fusion: 按排名而不是分数合并
        k = self.rrf_k
        combined: Dict[str, Dict] = {}
        for rank, r in enumerate(vector_results):
            doc_id = r.get("id", "")
            combined[doc_id] = {
                "id": doc_id,
                "content": r.get("content", ""),
                "metadata": r.get("metadata", {}),
                "score": 1.0 / (k + rank + 1),
                "search_type": "hybrid",
            }
        for rank, r in enumerate(keyword_results):
            doc_id = r.get("id", "")
            if doc_id in combined:
                combined[doc_id]["score"] += 1.0 / (k + rank + 1)
            else:
                combined[doc_id] = {
                    "id": doc_id,
                    "content": r.get("content", ""),
                    # 保留 BM25 命中的 metadata：父子分块去重需要 parent_id
                    "metadata": r.get("metadata", {}),
                    "score": 1.0 / (k + rank + 1),
                    "search_type": "hybrid",
                }

        results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def parent_child_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        父子检索：检索子块 → 按 parent_id 去重 → 返回父块

        子块做精确检索（混合检索），去重后返回父块给 LLM。

        Args:
            query: 查询文本
            top_k: 返回父块数量
            filters: 过滤条件（如 {"tag": "文献"}），传给 vector_search

        Returns:
            去重后的父块列表
        """
        pc_enabled = self.parent_child_config.get("enabled", True)
        if not pc_enabled:
            return self.hybrid_search(query, top_k=top_k)

        if top_k is None:
            top_k = self.retrieval_config.get("rerank_top_k", 3)

        # 混合检索子块（带过滤）
        child_results = self.hybrid_search(query, top_k=top_k * 3, filters=filters)

        # 按 parent_id 去重，保留最高分
        parent_map: Dict[str, Dict] = {}
        for r in child_results:
            meta = r.get("metadata", {})
            pid = meta.get("parent_id", "")
            if not pid:
                # 没有 parent_id 的旧文档，直接保留
                pid = r.get("id", r.get("content", ""))  # fallback
                if pid not in parent_map:
                    parent_map[pid] = r
                continue

            score = r.get("score", 0)
            if pid not in parent_map or score > parent_map[pid]["score"]:
                parent_map[pid] = {
                    "id": pid,
                    "content": meta.get("parent_content", r["content"]),
                    "score": score,
                    "section": meta.get("parent_section", ""),
                    "page_range": meta.get("page_range", ""),
                    "content_types": meta.get("content_types", ""),
                    "source": meta.get("source", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "search_type": "parent_child",
                }

        # 按分数排序
        sorted_parents = sorted(
            parent_map.values(), key=lambda x: x["score"], reverse=True
        )
        return sorted_parents[:top_k]

    def retrieve(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        统一检索接口

        Args:
            query: 查询文本
            search_type: 检索类型 (vector/keyword/hybrid)
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        if search_type == "vector":
            return self.vector_search(query, top_k, filters)
        else:  # hybrid
            return self.hybrid_search(query, top_k, filters)
