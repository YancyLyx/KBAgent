# -*- coding: utf-8 -*-
"""Milvus 向量存储后端

与 ChromaDB 的 VectorStore 提供相同接口，新增 hybrid_search()（Milvus 内置 RRFRanker）。

设计要点：
- backend = "milvus"，供 Retriever 按后端分支（Milvus 走内置 RRF，跳过手写 RRF）
- 主键 chunk_id（VARCHAR），与 Chroma 路径 chunk_id 语义一致
- 显式标量字段：tag/source/parent_id/parent_section/parent_content/chunk_index/
  title/page_range/content_types/doc_type，方便走标量索引过滤
- BM25 用 FunctionType.BM25 + analyzer type=chinese（jieba 分词）
- 稀疏索引 SPARSE_WAND，稠密 AUTOINDEX(COSINE)
- 一致性 Bounded（入库快）
- 重导入走 upsert（chunk_id 主键冲突按 upsert 语义，不留脏数据）
- get_collection_stats 用 query count(*) 兜底（growing segment row_count 不可信）
"""

import os
from typing import List, Dict, Any, Optional

import yaml
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from sentence_transformers import SentenceTransformer


# 显式标量字段（除 chunk_id 主键、text、sparse、dense 外）
SCALAR_FIELDS = [
    "tag",
    "source",
    "parent_id",
    "parent_section",
    "parent_content",
    "chunk_index",
    "title",
    "page_range",
    "content_types",
    "doc_type",
]

# search/bm25_search/hybrid_search 统一回填的 output_fields
OUTPUT_FIELDS = ["text"] + SCALAR_FIELDS


class MilvusVectorStore:
    """Milvus 向量存储器（与 VectorStore 同接口 + hybrid_search）"""

    backend = "milvus"

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        config_path: str = "config/rag_config.yaml",
    ):
        self.config = self._load_config(config_path)
        self.collection_name = collection_name

        vs_cfg = self.config.get("vector_store", {}) or {}
        milvus_cfg = vs_cfg.get("milvus", {}) or {}
        self.uri = os.getenv("MILVUS_URI", milvus_cfg.get("uri", "http://localhost:19530"))
        self.default_partition = milvus_cfg.get("default_partition", "kb_default")
        self.consistency = milvus_cfg.get("consistency", "Bounded")
        self.dim = int(milvus_cfg.get("dim", 512))

        # embedding 模型（与 Chroma 路径同一配置）
        self.embedding_model = self._init_embedding_model()

        self.client = MilvusClient(uri=self.uri, consistency_level=self.consistency)
        self._ensure_collection()
        # 确保 default partition 存在
        if not self.client.has_partition(self.collection_name, self.default_partition):
            self.client.create_partition(self.collection_name, self.default_partition)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _init_embedding_model(self) -> SentenceTransformer:
        emb_cfg = self.config.get("embedding", {}) or {}
        model_name = emb_cfg.get("model_name", "BAAI/bge-small-zh")
        device = emb_cfg.get("device", "cpu")
        print(f"[milvus] 正在加载 Embedding 模型: {model_name}...")
        model = SentenceTransformer(model_name, device=device)
        print("[milvus] Embedding 模型加载完成")
        return model

    def _build_schema(self):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=16384,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        for name, dt, ml in [
            ("tag", DataType.VARCHAR, 64),
            ("source", DataType.VARCHAR, 512),
            ("parent_id", DataType.VARCHAR, 128),
            ("parent_section", DataType.VARCHAR, 512),
            # parent_content 给满 Milvus VARCHAR 平台上限 65535：
            # 零成本且让超长父块显式报错（而非静默截断/入库失败整篇）
            ("parent_content", DataType.VARCHAR, 65535),
            ("chunk_index", DataType.INT64, None),
            ("title", DataType.VARCHAR, 512),
            ("page_range", DataType.VARCHAR, 64),
            ("content_types", DataType.VARCHAR, 256),
            ("doc_type", DataType.VARCHAR, 64),
        ]:
            if ml is None:
                schema.add_field(name, dt)
            else:
                schema.add_field(name, dt, max_length=ml)
        schema.add_function(
            Function(
                name="bm25_text",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        return schema

    def _ensure_collection(self):
        if self.client.has_collection(self.collection_name):
            print(f"[milvus] 使用已存在集合: {self.collection_name}")
            return
        schema = self._build_schema()
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_WAND", metric_type="BM25")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[milvus] 创建新集合: {self.collection_name} (dim={self.dim})")

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode_text(self, texts: List[str]) -> List[List[float]]:
        emb_cfg = self.config.get("embedding", {}) or {}
        normalize = emb_cfg.get("normalize_embeddings", True)
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # 过滤表达式
    # ------------------------------------------------------------------
    @staticmethod
    def _esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"')

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> str:
        """把 {"tag":"制度","source":"x.pdf"} 转成 Milvus filter 表达式"""
        if not filters:
            return ""
        parts = []
        for k, v in filters.items():
            if k not in SCALAR_FIELDS:
                continue
            parts.append(f'{k} == "{self._esc(str(v))}"')
        return " and ".join(parts)

    # ------------------------------------------------------------------
    # 插入（upsert 语义，重导入不留脏数据）
    # ------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 200) -> None:
        total = len(chunks)
        print(f"[milvus] 正在添加 {total} 个文档块（upsert）...")
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.get("content", "") for c in batch]
            embeddings = self.encode_text(texts)
            rows = []
            for c, emb in zip(batch, embeddings):
                ct = c.get("content_types", [])
                ct_str = ",".join(ct) if isinstance(ct, list) else str(ct)
                rows.append(
                    {
                        "chunk_id": c.get("chunk_id", f"chunk_{i}"),
                        "text": c.get("content", ""),
                        "dense": emb,
                        "tag": c.get("tag", ""),
                        "source": c.get("source", ""),
                        "parent_id": c.get("parent_id", ""),
                        "parent_section": c.get("parent_section", ""),
                        "parent_content": c.get("parent_content", ""),
                        "chunk_index": int(c.get("chunk_index", 0) or 0),
                        "title": c.get("title", ""),
                        "page_range": c.get("page_range", ""),
                        "content_types": ct_str,
                        "doc_type": c.get("doc_type", ""),
                    }
                )
            self.client.upsert(
                collection_name=self.collection_name,
                data=rows,
                partition_name=self.default_partition,
            )
            print(f"[milvus] 已 upsert {min(i + batch_size, total)}/{total}")
        print(f"[milvus] 成功添加 {total} 个文档块")

    # ------------------------------------------------------------------
    # 结果格式化（与 Chroma VectorStore 返回形状对齐）
    # ------------------------------------------------------------------
    def _format_hit(self, hit: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        ent = hit.get("entity", {}) or {}
        metadata = {k: ent.get(k, "") for k in SCALAR_FIELDS}
        if "chunk_index" in metadata and metadata["chunk_index"] != "":
            try:
                metadata["chunk_index"] = int(metadata["chunk_index"])
            except (TypeError, ValueError):
                pass
        return {
            "id": hit.get("chunk_id", ""),
            "content": ent.get("text", ""),
            "metadata": metadata,
            "distance": hit.get("distance", 0.0),
            "score": hit.get("distance", 0.0),
            "search_type": search_type,
        }

    # ------------------------------------------------------------------
    # dense 向量检索
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qemb = self.encode_text([query])[0]
        res = self.client.search(
            collection_name=self.collection_name,
            data=[qemb],
            anns_field="dense",
            limit=top_k,
            filter=self._build_filter(filters) or None,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        return [self._format_hit(h, "vector") for h in res[0]]

    # ------------------------------------------------------------------
    # sparse BM25 检索
    # ------------------------------------------------------------------
    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse",
            limit=top_k,
            filter=self._build_filter(filters) or None,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "BM25"},
        )
        return [self._format_hit(h, "keyword") for h in res[0]]

    # ------------------------------------------------------------------
    # 混合检索（Milvus 内置 RRFRanker，单次调用，禁止二次 RRF）
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qemb = self.encode_text([query])[0]
        fetch = max(top_k * 2, 10)
        req_dense = AnnSearchRequest(
            data=[qemb],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=fetch,
            expr=self._build_filter(filters) or None,
        )
        req_sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=fetch,
            expr=self._build_filter(filters) or None,
        )
        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=RRFRanker(100),
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
        )
        return [self._format_hit(h, "hybrid") for h in res[0]]

    # ------------------------------------------------------------------
    # 删除（按 metadata，转 delete(expr)）
    # ------------------------------------------------------------------
    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        expr = self._build_filter(where)
        if not expr:
            return 0
        # 先计数（delete 不返回条数），再删除
        counted = self.client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["chunk_id"],
            limit=16384,
        )
        n = len(counted or [])
        if n:
            self.client.delete(collection_name=self.collection_name, filter=expr)
        return n

    # ------------------------------------------------------------------
    # 标签
    # ------------------------------------------------------------------
    def get_available_tags(self) -> List[str]:
        rows = self.client.query(
            collection_name=self.collection_name,
            output_fields=["tag"],
            limit=16384,
        )
        tags = set()
        for r in rows or []:
            t = r.get("tag", "")
            if t:
                tags.add(t)
        return sorted(tags)

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------
    def delete_collection(self) -> None:
        self.client.drop_collection(collection_name=self.collection_name)
        print(f"[milvus] 已删除集合: {self.collection_name}")

    def get_collection_stats(self) -> Dict[str, Any]:
        # growing segment 的 row_count 不可信，用 query count(*) 兜底
        count = 0
        try:
            res = self.client.query(
                collection_name=self.collection_name,
                output_fields=["count(*)"],
            )
            if res:
                count = int(res[0].get("count(*)", 0))
        except Exception as e:
            print(f"[milvus] count(*) 失败，回退 0: {e}")
        return {
            "collection_name": self.collection_name,
            "document_count": count,
        }
