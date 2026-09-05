# -*- coding: utf-8 -*-
"""会话记忆模块

基于 ChromaDB 的对话历史存储，支持跨会话语义回忆。
替代原有的 SQLite session_store + deque 短期记忆。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import uuid

import chromadb
from chromadb.config import Settings


# 对话记忆专用 Chroma 客户端（模块级共享，避免同进程多 PersistentClient 冲突）。
# 注意：对话记忆是"语义回忆用向量库"，**独立于知识库后端**——知识库默认已切
# Milvus 后，这里仍走 Chroma（迁移踩坑：早期直接复用 vector_store.client，
# 切 Milvus 后 client 变 MilvusClient，create_collection(name=...) 是 Chroma
# 签名 → 500，chat 拿不到 token）。
_chroma_client = None


def _memory_chroma() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        _chroma_client = chromadb.PersistentClient(
            path=db_path,
            # 与 VectorStore 完全一致的 settings，避免 SharedSystemClient 冲突
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _chroma_client


class ConversationMemory:
    """会话记忆

    使用 ChromaDB 存储对话历史（与知识库后端解耦），支持：
    - 按 session_id 精准检索历史
    - 按用户跨会话语义回忆（Semantic Recall）
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        # 只复用 vector_store 的 embedding 编码与集合命名，client 用独立 Chroma
        self.client = _memory_chroma()
        self.collection = self._get_or_create_collection()
        self._turn_counter: Dict[str, int] = {}

    def _get_or_create_collection(self):
        """获取或创建对话历史集合（Chroma）"""
        collection_name = f"{self.vector_store.collection_name}_conversations"
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> None:
        """记录一轮对话"""
        turn_idx = self._next_turn_index(session_id)
        timestamp = datetime.now().isoformat()
        # doc_id 加 uuid 后缀：即使计数恢复逻辑出现偏差（如查询失败），
        # 也不会因 ID 重复导致整轮对话写入失败
        doc_id = f"{session_id}_{turn_idx}_{uuid.uuid4().hex[:8]}"

        # 用 query embedding 编码（复用 VectorStore 的模型）
        embedding = self.vector_store.encode_text([content])[0]

        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "session_id": session_id,
                "user_id": user_id,
                "role": role,
                "turn_index": turn_idx,
                "timestamp": timestamp,
            }],
            ids=[doc_id],
        )

        self._turn_counter[session_id] = turn_idx + 1

    def _next_turn_index(self, session_id: str) -> int:
        """获取下一个轮次序号。

        优先使用内存计数；进程重启后内存计数丢失，则从 ChromaDB 现有
        记录中恢复最大 turn_index，避免 doc_id 重复导致写入失败。
        """
        if session_id in self._turn_counter:
            return self._turn_counter[session_id]
        try:
            results = self.collection.get(
                where={"session_id": session_id},
                include=["metadatas"],
            )
            max_idx = -1
            for m in (results.get("metadatas") or []):
                try:
                    max_idx = max(max_idx, int(m.get("turn_index", -1)))
                except (TypeError, ValueError):
                    continue
            self._turn_counter[session_id] = max_idx + 1
            return max_idx + 1
        except Exception:
            # 查询失败时保守从 0 开始，配合 doc_id 唯一化避免覆盖
            self._turn_counter[session_id] = 0
            return 0

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------

    def get_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """按 session_id 获取对话历史（按时间排序）"""
        results = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return []

        turns = [
            {
                "role": meta["role"],
                "content": doc,
                "timestamp": meta.get("timestamp", ""),
                "turn_index": meta.get("turn_index", 0),
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        turns.sort(key=lambda t: t["turn_index"])
        return turns[-limit:]

    def semantic_recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """跨会话语义回忆：搜索该用户历史对话中最相关的片段"""
        query_embedding = self.vector_store.encode_text([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        turns = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            turns.append({
                "content": doc,
                "role": meta.get("role", ""),
                "session_id": meta.get("session_id", ""),
                "timestamp": meta.get("timestamp", ""),
                "relevance": round(1 - dist, 3),
            })
        return turns

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的全部历史"""
        results = self.collection.get(where={"session_id": session_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        self._turn_counter.pop(session_id, None)
