# -*- coding: utf-8 -*-
"""测试 Phase 4：ConversationMemory"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def _has_model():
    try:
        from app.rag.vector_store import VectorStore
        store = VectorStore(collection_name="probe_mem_ok")
        store.delete_collection()
        return True
    except Exception:
        return False


def _cleanup(store):
    """清理基础集合 + ConversationMemory 自己的 _conversations 集合"""
    try:
        store.client.delete_collection(f"{store.collection_name}_conversations")
    except Exception:
        pass
    try:
        store.delete_collection()
    except Exception:
        pass


@pytest.mark.skipif(not _has_model(), reason="embedding model not available")
def test_memory_add_and_get():
    from app.rag.vector_store import VectorStore
    from app.memory.conversation_memory import ConversationMemory
    store = VectorStore(collection_name="test_mem_basic")
    try:
        mem = ConversationMemory(store)
        mem.add_turn("s1", "u1", "user", "你好")
        mem.add_turn("s1", "u1", "assistant", "你好！")
        assert len(mem.get_history("s1")) == 2
    finally:
        _cleanup(store)
    print("✓ test_memory_add_and_get")


@pytest.mark.skipif(not _has_model(), reason="embedding model not available")
def test_memory_semantic_recall():
    from app.rag.vector_store import VectorStore
    from app.memory.conversation_memory import ConversationMemory
    store = VectorStore(collection_name="test_mem_recall")
    try:
        mem = ConversationMemory(store)
        mem.add_turn("s1", "u1", "user", "如何创建 API Key？")
        mem.add_turn("s1", "u1", "assistant", "在设置页面创建")
        mem.add_turn("s1", "u1", "user", "退款需要多久？")
        mem.add_turn("s1", "u1", "assistant", "3-5 工作日")
        results = mem.semantic_recall("u1", "退款", top_k=1)
        assert len(results) >= 1
    finally:
        _cleanup(store)
    print("✓ test_memory_semantic_recall")


@pytest.mark.skipif(not _has_model(), reason="embedding model not available")
def test_memory_clear():
    from app.rag.vector_store import VectorStore
    from app.memory.conversation_memory import ConversationMemory
    store = VectorStore(collection_name="test_mem_clear")
    try:
        mem = ConversationMemory(store)
        mem.add_turn("s1", "u1", "user", "测试")
        mem.add_turn("s2", "u2", "user", "测试")
        mem.clear_session("s1")
        assert len(mem.get_history("s1")) == 0
        assert len(mem.get_history("s2")) == 1
    finally:
        _cleanup(store)
    print("✓ test_memory_clear")


if __name__ == "__main__":
    print("=== Phase 4 Tests ===")
    if _has_model():
        test_memory_add_and_get()
        test_memory_semantic_recall()
        test_memory_clear()
    else:
        print("  SKIP: embedding model not available")
    print("\n✅ Phase 4 complete")
