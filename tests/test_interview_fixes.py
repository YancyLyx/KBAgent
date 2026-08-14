# -*- coding: utf-8 -*-
"""完善点回归测试

覆盖面试拷打风险修复：
1. 语义缓存：namespace 隔离 / TTL / 主动失效 / 进程级共享
2. 工具安全边界：路径越界与敏感文件（.env / .git*）拒绝访问
3. 工具参数校验：缺必填参数返回纠错信息，不再生成 None.md 脏文件
4. BM25：支持 tag 过滤 + 补齐 metadata（父子分块上下文不丢失）
5. 混合检索：按 tag 检索时 keyword-only 命中不会混入其他知识库
6. 长程记忆：进程重启后 turn_index 计数恢复，doc_id 不重复
7. 演进式摘要：触发时机 = 短期记忆窗口边界
8. 会话生命周期：空闲超时 / 超量淘汰
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def _has_model():
    try:
        from app.rag.vector_store import VectorStore
        store = VectorStore(collection_name="probe_fix_ok")
        store.delete_collection()
        return True
    except Exception:
        return False


NEEDS_MODEL = pytest.mark.skipif(
    not _has_model(), reason="embedding model not available"
)


# ----------------------------------------------------------------------
# 1. 语义缓存
# ----------------------------------------------------------------------

def test_cache_namespace_isolation():
    """不同 namespace 的相同 query 不互相命中（防止跨用户串答案）"""
    from app.cache.query_cache import QueryCache
    cache = QueryCache()
    cache.put("报销流程是什么", "A 答案", namespace="user_a")
    assert cache.get("报销流程是什么", namespace="user_a") == "A 答案"
    assert cache.get("报销流程是什么", namespace="user_b") is None


def test_cache_ttl_and_invalidate():
    """TTL 过期 + 主动失效"""
    from app.cache.query_cache import QueryCache
    cache = QueryCache(ttl_seconds=1)
    cache.put("问题", "答案", namespace="u1")
    assert cache.get("问题", namespace="u1") == "答案"

    time.sleep(1.1)
    assert cache.get("问题", namespace="u1") is None

    cache.put("问题2", "答案2", namespace="u1")
    assert cache.invalidate(namespace="u1") == 1
    assert cache.get("问题2", namespace="u1") is None


def test_cache_shared_singleton():
    """不同 Agent 实例共享进程级缓存（会话间可复用答案）"""
    from app.cache.query_cache import get_shared_cache
    c1 = get_shared_cache()
    c2 = get_shared_cache()
    assert c1 is c2
    c1.clear()


# ----------------------------------------------------------------------
# 2. 工具安全边界
# ----------------------------------------------------------------------

def test_skill_path_escape_raises():
    """越界路径必须报错，而不是静默返回项目根目录"""
    from app.agent.skill_manager import SkillManager
    sm = SkillManager(None)
    with pytest.raises(ValueError):
        sm._resolve_path("../../etc/passwd")
    with pytest.raises(ValueError):
        sm._resolve_path("/etc/passwd")


def test_skill_file_tools_block_sensitive_files(tmp_path, monkeypatch):
    """文件工具拒绝读写 .env / .git* 敏感文件"""
    from app.agent.skill_manager import SkillManager
    sm = SkillManager(None)
    # 构造一个模拟项目根目录，其中包含 .env
    root = tmp_path / "proj"
    root.mkdir()
    env_file = root / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-secret", encoding="utf-8")
    monkeypatch.setattr(sm, "PROJECT_ROOT", root)

    assert "禁止" in sm._read_file(".env")
    result = sm._create_file(".env.local", "x")
    assert "禁止" in result
    result = sm._write_file(".env", "hacked")
    assert "禁止" in result
    # 文件内容未被修改
    assert env_file.read_text(encoding="utf-8") == "DEEPSEEK_API_KEY=sk-secret"


# ----------------------------------------------------------------------
# 3. 工具参数校验
# ----------------------------------------------------------------------

def test_tool_router_missing_params():
    """缺必填参数时返回纠错信息，而不是生成 None.md 脏文件"""
    from app.agent.tool_router import ToolRouter
    from app.agent.skill_manager import SkillManager
    router = ToolRouter()
    router.skill_manager = SkillManager(None)

    result = router.call_skill_tool("read_category_info", {})
    assert "参数缺失" in result
    assert "tag" in result

    result = router.call_skill_tool("search_knowledge_base", {"query": ""})
    assert "参数缺失" in result


# ----------------------------------------------------------------------
# 4/5. BM25 过滤 + metadata（需要 embedding 模型）
# ----------------------------------------------------------------------

@NEEDS_MODEL
def test_cache_semantic_namespace_isolation():
    """语义命中不跨 namespace：A 用户的问题不会命中 B 用户的缓存"""
    from app.cache.query_cache import QueryCache
    from app.rag.vector_store import VectorStore
    store = VectorStore(collection_name="test_fix_cache_ns")
    try:
        # 阈值按 embedding 分布校准（bge-small-zh 下「差旅报销怎么走」与
        # 「出差费用如何报销」余弦约 0.90，默认 0.92 偏严；测试放宽到 0.85）
        cache = QueryCache(threshold=0.85)
        emb_a = store.encode_text(["差旅报销怎么走"])[0]
        cache.put("差旅报销怎么走", "A 用户专属答案", emb_a, namespace="user_a")

        # B 用户问类似问题：即使语义相似也不能命中 A 的缓存
        emb_b = store.encode_text(["出差费用如何报销"])[0]
        assert cache.get("出差费用如何报销", emb_b, namespace="user_b") is None
        # A 自己问类似问题：语义命中自己之前的答案
        assert cache.get("出差费用如何报销", emb_b, namespace="user_a") == "A 用户专属答案"
    finally:
        store.delete_collection()


@NEEDS_MODEL
def test_bm25_filter_and_metadata():
    """BM25 支持 tag 过滤，结果带 metadata"""
    from app.rag.vector_store import VectorStore
    store = VectorStore(collection_name="test_fix_bm25")
    try:
        store.add_documents([
            {
                "content": "差旅报销标准为每晚 400 元",
                "chunk_id": "doc_a_0",
                "source": "a.md",
                "tag": "行政",
                "parent_id": "parent_a",
                "parent_content": "【父块A】差旅报销标准为每晚 400 元",
            },
            {
                "content": "差旅报销标准为每晚 600 元",
                "chunk_id": "doc_b_0",
                "source": "b.md",
                "tag": "财务",
                "parent_id": "parent_b",
                "parent_content": "【父块B】差旅报销标准为每晚 600 元",
            },
        ])

        results = store.bm25_search("差旅报销", top_k=10, filters={"tag": "行政"})
        assert len(results) == 1
        assert results[0]["metadata"]["tag"] == "行政"
        assert results[0]["metadata"]["parent_id"] == "parent_a"
        assert results[0]["search_type"] == "keyword"
    finally:
        store.delete_collection()


@NEEDS_MODEL
def test_hybrid_search_tag_isolation():
    """混合检索带 tag 过滤时，不混入其他知识库的 keyword-only 命中"""
    from app.rag.vector_store import VectorStore
    from app.rag.retriever import Retriever
    store = VectorStore(collection_name="test_fix_hybrid")
    try:
        store.add_documents([
            {
                "content": "员工持股计划的行权期为 3 年",
                "chunk_id": "d1_0",
                "source": "stock.md",
                "tag": "股权",
            },
            {
                "content": "员工持股计划的行权期说明文档",
                "chunk_id": "d2_0",
                "source": "hr.md",
                "tag": "人力",
            },
        ])
        retriever = Retriever(store)
        results = retriever.hybrid_search(
            "员工持股计划行权期", top_k=10, filters={"tag": "股权"}
        )
        assert results, "应至少有一个命中"
        for r in results:
            assert r.get("metadata", {}).get("tag", "股权") == "股权"
    finally:
        store.delete_collection()


# ----------------------------------------------------------------------
# 6. 长程记忆计数恢复
# ----------------------------------------------------------------------

@NEEDS_MODEL
def test_conversation_memory_counter_recovery():
    """模拟重启：新实例写同一会话不产生重复 doc_id"""
    from app.rag.vector_store import VectorStore
    from app.memory.conversation_memory import ConversationMemory
    store = VectorStore(collection_name="test_fix_mem")
    try:
        mem1 = ConversationMemory(store)
        mem1.add_turn("s1", "u1", "user", "第一轮问题")
        mem1.add_turn("s1", "u1", "assistant", "第一轮回答")

        # 模拟进程重启：全新 ConversationMemory（内存计数为空）
        mem2 = ConversationMemory(store)
        mem2.add_turn("s1", "u1", "user", "第二轮问题")
        mem2.add_turn("s1", "u1", "assistant", "第二轮回答")

        history = mem2.get_history("s1")
        assert len(history) == 4
        assert history[-1]["content"] == "第二轮回答"
    finally:
        # 同时清理 ConversationMemory 自己的 collection（基础集合名 + _conversations）
        try:
            store.client.delete_collection(f"{store.collection_name}_conversations")
        except Exception:
            pass
        store.delete_collection()


# ----------------------------------------------------------------------
# 7. 演进式摘要触发
# ----------------------------------------------------------------------

def test_summary_trigger_window():
    """摘要触发 = 短期记忆窗口边界（第 N、2N、3N 轮），滑出窗口的对话不丢"""
    from app.agent.agent_manager import AgentManager
    from app.memory.short_memory import ShortMemory

    # 窗口 N 轮：第 N、2N 轮触发（用会话累计轮数判断）
    assert AgentManager.summary_due(0, 5) is False
    assert AgentManager.summary_due(5, 5) is True
    assert AgentManager.summary_due(10, 5) is True
    assert AgentManager.summary_due(7, 5) is False

    # ShortMemory 只保留最近 N 轮；get_turn_count 返回窗口内轮数（恒 ≤N），
    # 这正是不能拿它当摘要触发依据的原因
    mem = ShortMemory()
    window = mem.max_history_length
    for i in range(1, window * 2 + 1):
        mem.add_turn(f"问题{i}", f"回答{i}")
    assert mem.get_turn_count() == window  # 窗口内只剩 N 轮
    history = mem.get_history()
    assert len(history) == window  # 窗口内只留 N 轮
    assert history[0]["user"] == f"问题{window + 1}"  # 前 N 轮已滑出
    # 会话累计到第 2N 轮时触发：用「旧摘要 + 窗口内 N 轮」合并，前 N 轮不丢
    assert AgentManager.summary_due(window * 2, window)


# ----------------------------------------------------------------------
# 8. 会话生命周期
# ----------------------------------------------------------------------

def test_session_eviction(monkeypatch):
    """空闲超时会话被回收；超量时按最后活跃时间淘汰"""
    from datetime import datetime, timedelta
    import app.api.chat_api as chat_api

    now = datetime.now()
    sessions = {
        "s_old": {
            "user_id": "u1",
            "last_active": (now - timedelta(minutes=40)).isoformat(),
        },
        "s_fresh": {
            "user_id": "u2",
            "last_active": (now - timedelta(minutes=1)).isoformat(),
        },
    }
    monkeypatch.setattr(chat_api, "_active_sessions", sessions)
    removed = chat_api._evict_stale_sessions(now=now)
    assert removed == 1
    assert "s_old" not in sessions
    assert "s_fresh" in sessions

    # 超量淘汰：设置上限为 1，两个活跃会话应淘汰最旧的一个
    sessions2 = {
        "s1": {"user_id": "u1", "last_active": (now - timedelta(minutes=10)).isoformat()},
        "s2": {"user_id": "u2", "last_active": (now - timedelta(minutes=5)).isoformat()},
    }
    monkeypatch.setattr(chat_api, "_active_sessions", sessions2)
    monkeypatch.setattr(chat_api, "MAX_ACTIVE_SESSIONS", 1)
    chat_api._evict_stale_sessions(now=now)
    assert "s1" not in sessions2
    assert "s2" in sessions2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
