# -*- coding: utf-8 -*-
"""向量存储模块

提供基于 ChromaDB 的向量存储功能。
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import yaml
import numpy as np
import re
from rank_bm25 import BM25Okapi

class VectorStore:
    """向量存储器"""

    backend = "chroma"

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        config_path: str = "config/rag_config.yaml",
    ):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.collection_name = collection_name
        
        # 初始化 embedding 模型
        self.embedding_model = self._init_embedding_model()
        
        # 初始化 ChromaDB 客户端
        self.client = self._init_chroma_client()
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection()

        # BM25 索引（实例级，避免多实例共享类属性脏状态）
        self._bm25 = None
        self._bm25_texts = []
        self._bm25_ids = []
        self._bm25_metas = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def _init_embedding_model(self) -> SentenceTransformer:
        """初始化 embedding 模型"""
        embedding_config = self.config.get("embedding", {})
        model_name = embedding_config.get("model_name", "BAAI/bge-small-zh")
        device = embedding_config.get("device", "cpu")
        
        print(f"正在加载 Embedding 模型: {model_name}...")
        model = SentenceTransformer(model_name, device=device)
        print("Embedding 模型加载完成")
        return model

    def _init_chroma_client(self) -> chromadb.Client:
        """初始化 ChromaDB 客户端"""
        # 从环境变量或配置获取路径
        db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        
        # 确保目录存在
        os.makedirs(db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        return client

    def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f"使用已存在的集合: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"创建新集合: {self.collection_name}")
        return collection

    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """
        将文本编码为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        embedding_config = self.config.get("embedding", {})
        normalize = embedding_config.get("normalize_embeddings", True)
        
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        添加文档块到向量库

        Args:
            chunks: 文档块列表
            batch_size: 批量大小
        """
        total = len(chunks)
        print(f"正在添加 {total} 个文档块到向量库...")
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            
            # 准备数据
            texts = [chunk["content"] for chunk in batch]
            embeddings = self.encode_text(texts)
            
            ids = [chunk.get("chunk_id", f"chunk_{i + j}") for j, chunk in enumerate(batch)]
            metadatas = [
                {
                    "source": chunk.get("source", "unknown"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "title": chunk.get("title", ""),
                    "parent_id": chunk.get("parent_id", ""),
                    "parent_content": chunk.get("parent_content", ""),
                    "parent_section": chunk.get("parent_section", ""),
                    "page_range": chunk.get("page_range", ""),
                    "content_types": ",".join(chunk.get("content_types", [])),
                    "tag": chunk.get("tag", ""),                }
                for chunk in batch
            ]
            
            # 添加到集合
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"已添加 {min(i + batch_size, total)}/{total} 个文档块")
        
        # 更新 BM25 索引
        self._rebuild_bm25()
        print(f"成功添加 {total} 个文档块")

    # ------------------------------------------------------------------
    # BM25 关键词检索
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> "list[str]":
        """中文分词：CJK 连续串按重叠二元组分词，英文/数字按单词。

        不能直接用 \\w+：中文没有空格分词，整句会变成一个 token，
        「差旅报销」永远匹配不上「差旅报销标准为每晚 400 元」。
        选择 bigram 而非 jieba：零依赖、快、对领域新词不敏感；
        代价是部分噪音，由混合检索 + CrossEncoder 精排兜底。
        """
        tokens = []
        for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
            if re.match(r"[\u4e00-\u9fff]+", part):
                if len(part) >= 2:
                    tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
                else:
                    tokens.append(part)
            else:
                tokens.append(part)
        return tokens

    def _rebuild_bm25(self):
        """从 ChromaDB 全量重建 BM25 索引"""
        all_docs = self.collection.get(include=["documents", "metadatas"])
        texts = all_docs.get("documents") or []
        ids = all_docs.get("ids") or []
        if not texts:
            self._bm25 = BM25Okapi([])
            self._bm25_texts = []
            self._bm25_ids = []
            self._bm25_metas = []
            return
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_texts = texts
        self._bm25_ids = ids
        self._bm25_metas = all_docs.get("metadatas") or []

    @staticmethod
    def _match_meta(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """判断 metadata 是否满足过滤条件（等值匹配，兼容 Chroma where 语义）"""
        meta = meta or {}
        for k, v in filters.items():
            if meta.get(k) != v:
                return False
        return True

    def bm25_search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> list:
        """BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据过滤（如 {"tag": "文献"}），与向量检索保持一致，
                避免按知识库分类检索时混入其他分类的文档。
        """
        if self._bm25 is None:
            self._rebuild_bm25()
        if not self._bm25_texts:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 过滤 + 按分数排序取 top_k
        candidates = range(len(self._bm25_texts))
        if filters:
            candidates = [
                i for i in candidates
                if self._match_meta(self._bm25_metas[i], filters)
            ]
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            # 注意：小语料下高频词的 idf 可能为负（BM25 原始公式），
            # 此时真实命中的分数也是负的，不能按 >0 过滤；
            # score == 0 表示查询词在文档中完全没有出现，才应排除。
            # RRF 合并只看排名不看绝对值，负分不影响混合检索。
            if scores[idx] != 0:
                results.append({
                    "id": self._bm25_ids[idx],
                    "content": self._bm25_texts[idx],
                    # 补齐 metadata：父子分块去重依赖 parent_id/parent_content，
                    # 缺 metadata 会导致 BM25 命中的子块丢失父块上下文
                    "metadata": self._bm25_metas[idx] or {},
                    "score": float(scores[idx]),
                    "search_type": "keyword",
                })
        return results

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """按元数据条件删除向量（如删除某文档的所有分块），并重建 BM25 索引。

        ChromaDB 支持 where 删除，因此文档删除/重索引时必须同步清理向量，
        否则删除后的文档仍会被检索到。

        Returns:
            删除的向量条数
        """
        existing = self.collection.get(where=where, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
            self._rebuild_bm25()
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件

        Returns:
            搜索结果列表
        """
        # 编码查询
        query_embedding = self.encode_text([query])[0]
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i]  # 转换为相似度分数
            })
        
        return formatted_results

    def get_available_tags(self) -> List[str]:
        """获取当前知识库中所有已使用的标签"""
        all_metadatas = self.collection.get(include=["metadatas"])
        tags = set()
        for m in (all_metadatas.get("metadatas") or []):
            tag = m.get("tag", "")
            if tag:
                tags.add(tag)
        return sorted(tags)

    def delete_collection(self) -> None:
        """删除当前集合"""
        self.client.delete_collection(name=self.collection_name)
        print(f"已删除集合: {self.collection_name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count
        }


def create_vector_store(
    collection_name: str = "knowledge_base",
    config_path: str = "config/rag_config.yaml",
):
    """工厂：按配置选择后端。

    config/rag_config.yaml:
        vector_store:
          backend: chroma | milvus   # 默认 chroma
          milvus:
            uri: http://localhost:19530
            ...

    返回 VectorStore(Chroma) 或 MilvusVectorStore，二者接口一致。
    """
    import yaml as _yaml

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    backend = (cfg.get("vector_store", {}) or {}).get("backend", "chroma").lower()

    if backend == "milvus":
        from .milvus_store import MilvusVectorStore
        return MilvusVectorStore(collection_name, config_path)
    # 默认 chroma，保持现有行为一字不改
    return VectorStore(collection_name, config_path)
