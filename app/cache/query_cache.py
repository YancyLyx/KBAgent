# -*- coding: utf-8 -*-
"""语义缓存模块

对用户 query 做「精确 + 语义」两级去重：
- 精确匹配：相同 (namespace, query) 直接命中，O(1) 返回
- 语义匹配：query embedding 与缓存中同一 namespace 下已有 query 的
  余弦相似度 ≥ 阈值 → 命中，复用该 query 的答案

设计要点（回答「为什么」时用）：
1. namespace 隔离：缓存键 = (user_id 或租户, query)，防止不同用户
   问同一问题时串答案（隐私 + 权限隔离）。语义检索只在同一 namespace
   内进行，避免跨用户语义误命中。
2. TTL：每条缓存带过期时间，默认 1 小时。知识库更新后可通过
   invalidate() 主动失效，避免「文档已更新、答案还是旧的」。
3. 容量受限：LRU 淘汰；内存态实现，生产可平滑替换为 Redis
   （key 前缀 + ZSET 过期扫描，保持 get/put/invalidate 接口）。
"""

import threading
import time
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np


DEFAULT_NAMESPACE = "default"


class QueryCache:
    """LRU 语义缓存（namespace 隔离 + TTL）"""

    def __init__(
        self,
        capacity: int = 256,
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
    ):
        self.capacity = capacity
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        # key = (namespace, query) -> {answer, embedding, created_at}
        self._store: "OrderedDict[tuple, Dict]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def get(
        self,
        query: str,
        query_emb: Optional[np.ndarray] = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> Optional[str]:
        """查询缓存。

        Args:
            query: 用户原始问题
            query_emb: 查询向量（可选），用于语义命中
            namespace: 缓存隔离域（一般传 user_id / 租户 ID）
        """
        with self._lock:
            now = time.time()
            self._expire_locked(now)

            key = (namespace, query)
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key].get("answer")

            if query_emb is not None:
                hit_key, sim = self._lookup_semantic(namespace, self._norm(query_emb))
                if hit_key:
                    self._store.move_to_end(hit_key)
                    return self._store[hit_key].get("answer")
            return None

    def put(
        self,
        query: str,
        answer: str,
        query_emb: Optional[np.ndarray] = None,
        namespace: str = DEFAULT_NAMESPACE,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """写入缓存（覆盖同 (namespace, query) 旧值）"""
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        with self._lock:
            self._store[(namespace, query)] = {
                "answer": answer,
                "embedding": self._norm(query_emb) if query_emb is not None else None,
                "created_at": time.time(),
                "ttl_seconds": ttl,
            }
            self._store.move_to_end((namespace, query))
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def invalidate(
        self,
        namespace: Optional[str] = None,
        query: Optional[str] = None,
    ) -> int:
        """失效缓存。

        Args:
            namespace: 只清该命名空间；None 表示全部
            query: 只清该 query；None 表示该命名空间（或全部）都清

        Returns:
            清除的条目数
        """
        with self._lock:
            keys = list(self._store.keys())
            removed = 0
            for k in keys:
                k_ns, k_query = k
                if namespace is not None and k_ns != namespace:
                    continue
                if query is not None and k_query != query:
                    continue
                del self._store[k]
                removed += 1
            return removed

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _expire_locked(self, now: float) -> None:
        """删除已过期的条目（调用方需持有锁）"""
        expired_keys = [
            k for k, item in self._store.items()
            if now - item.get("created_at", 0) > item.get("ttl_seconds", self.ttl_seconds)
        ]
        for k in expired_keys:
            del self._store[k]

    def _lookup_semantic(self, namespace: str, query_emb: np.ndarray):
        """在同一 namespace 内查找与查询向量最相似的 query"""
        best_key, best_sim = None, 0.0
        for key, item in self._store.items():
            if key[0] != namespace:
                continue
            emb = item.get("embedding")
            if emb is None:
                continue
            sim = float(np.dot(emb, query_emb))  # 双方均归一化 → 余弦相似度
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None and best_sim >= self.threshold:
            return best_key, best_sim
        return None, 0.0

    @staticmethod
    def _norm(emb: np.ndarray) -> np.ndarray:
        """归一化向量（兼容 list / np.ndarray 入参）"""
        arr = np.asarray(emb, dtype=float)
        n = float(np.linalg.norm(arr))
        return arr / n if n > 0 else arr


# 全局共享缓存：不同会话/请求共用一份，避免「每个 Agent 实例一个空缓存」
# 导致命中率趋近于 0；也让管理端（文档删除/重索引）可以主动失效全部缓存。
_shared_cache: Optional[QueryCache] = None
_shared_cache_lock = threading.Lock()


def get_shared_cache() -> QueryCache:
    """获取进程级共享缓存单例"""
    global _shared_cache
    with _shared_cache_lock:
        if _shared_cache is None:
            _shared_cache = QueryCache()
        return _shared_cache
