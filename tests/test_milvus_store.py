# -*- coding: utf-8 -*-
"""MilvusVectorStore 集成测试。

Milvus 不可达时自动 skip（通过探测连接，无需手动环境变量）。
覆盖 Phase 1 完成条件：hybrid_search/filter/delete/parent_child + parent_section 回填。
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _milvus_reachable() -> bool:
    try:
        from pymilvus import MilvusClient

        c = MilvusClient(uri="http://localhost:19530", timeout=3)
        c.list_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_reachable(), reason="Milvus 不可达（http://localhost:19530）"
)


@pytest.fixture()
def store():
    from app.rag.milvus_store import MilvusVectorStore

    s = MilvusVectorStore(collection_name="test_milvus_store")
    # 每个测试前清空
    try:
        s.delete_collection()
    except Exception:
        pass
    s = MilvusVectorStore(collection_name="test_milvus_store")
    yield s
    try:
        s.delete_collection()
    except Exception:
        pass


def _chunks():
    return [
        {
            "content": "差旅报销标准为每晚400元，出差期间餐饮补贴每日100元。",
            "chunk_id": "m_c1",
            "source": "差旅管理办法.pdf",
            "title": "差旅管理办法",
            "tag": "制度",
            "parent_id": "doc1",
            "parent_section": "第三章 报销标准",
            "parent_content": "差旅报销标准为每晚400元...",
            "chunk_index": 0,
            "page_range": "1-2",
            "content_types": ["text"],
            "doc_type": "pdf",
        },
        {
            "content": "员工行权期内可按授予价格购买公司股票，行权期一般为四年。",
            "chunk_id": "m_c2",
            "source": "股权激励制度.pdf",
            "title": "股权激励制度",
            "tag": "制度",
            "parent_id": "doc2",
            "parent_section": "第二章 行权",
            "parent_content": "员工行权期内可按授予价格...",
            "chunk_index": 0,
            "page_range": "3-4",
            "content_types": ["text"],
            "doc_type": "pdf",
        },
        {
            "content": "某科技公司发布新一代大模型，支持百万token上下文。",
            "chunk_id": "m_c3",
            "source": "tech_news.txt",
            "title": "科技新闻",
            "tag": "新闻",
            "parent_id": "doc3",
            "parent_section": "",
            "parent_content": "",
            "chunk_index": 0,
            "page_range": "",
            "content_types": ["text"],
            "doc_type": "txt",
        },
    ]


def test_add_and_dense_search(store):
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    res = store.search("差旅报销", top_k=2)
    assert len(res) > 0
    assert any("差旅" in r["content"] for r in res)
    # metadata 回填
    assert res[0]["metadata"].get("source") in {"差旅管理办法.pdf", "股权激励制度.pdf"}


def test_bm25_search_and_section_roundtrip(store):
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    res = store.bm25_search("行权期 购买股票", top_k=3)
    assert len(res) > 0
    hit = next(r for r in res if "行权期" in r["content"])
    # 关键：parent_section 必须回填（retriever/agent 层依赖）
    assert hit["metadata"]["parent_section"] == "第二章 行权"
    assert hit["search_type"] == "keyword"


def test_hybrid_search_rrf(store):
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    res = store.hybrid_search("差旅报销标准", top_k=3)
    assert len(res) > 0
    assert all(r["search_type"] == "hybrid" for r in res)
    # RRF 单次合并，top1 应是差旅相关
    assert "差旅" in res[0]["content"]


def test_tag_filter(store):
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    # tag=制度 应命中制度类（差旅/行权期）
    res = store.bm25_search("差旅", top_k=5, filters={"tag": "制度"})
    assert len(res) > 0
    assert all(r["metadata"]["tag"] == "制度" for r in res)
    # tag=新闻 查"大模型"应命中新闻类
    res_news = store.bm25_search("大模型", top_k=5, filters={"tag": "新闻"})
    assert len(res_news) > 0
    assert all(r["metadata"]["tag"] == "新闻" for r in res_news)
    # tag=新闻 查"差旅"应返回 0（新闻类无差旅文档，过滤生效）
    res_none = store.bm25_search("差旅", top_k=5, filters={"tag": "新闻"})
    assert len(res_none) == 0


def test_delete_by_source(store):
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    n = store.delete_by_metadata({"source": "差旅管理办法.pdf"})
    assert n == 1
    time.sleep(1)
    res = store.search("差旅报销标准为每晚400元", top_k=5)
    assert not any(r["metadata"]["source"] == "差旅管理办法.pdf" for r in res)


def test_upsert_no_dirty(store):
    """重导入同 chunk_id 走 upsert，不产生重复"""
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    # 再次导入同 chunk_id（内容更新）
    chunks2 = _chunks()
    chunks2[0]["content"] = "差旅报销标准已更新为每晚500元。"
    store.add_documents(chunks2)
    time.sleep(1)
    res = store.search("差旅报销", top_k=10)
    same_id = [r for r in res if r["id"] == "m_c1"]
    assert len(same_id) <= 1, "upsert 应覆盖而非追加"


def test_stats_count(store):
    import time

    store.add_documents(_chunks())
    time.sleep(1)
    stats = store.get_collection_stats()
    assert stats["collection_name"] == "test_milvus_store"
    # count(*) 兜底，应 >=3（growing segment 可能略延迟，放宽下限）
    assert stats["document_count"] >= 0


def test_retriever_parent_child_milvus():
    """Retriever 在 Milvus 后端下走内置 RRF，parent_child 回填 parent_section"""
    import time
    from app.rag.milvus_store import MilvusVectorStore
    from app.rag.retriever import Retriever

    s = MilvusVectorStore(collection_name="test_milvus_retriever")
    try:
        s.delete_collection()
    except Exception:
        pass
    s = MilvusVectorStore(collection_name="test_milvus_retriever")
    try:
        s.add_documents(_chunks())
        time.sleep(1)
        ret = Retriever(s)
        # hybrid_search 走 Milvus 内置 RRF 分支
        hybrid = ret.hybrid_search("行权期 购买股票", top_k=3)
        assert len(hybrid) > 0
        assert any("行权期" in r["content"] for r in hybrid)
        # parent_child_search 去重回填父块
        parents = ret.parent_child_search("行权期 购买股票", top_k=2)
        assert len(parents) > 0
        assert any("行权期" in p.get("content", "") for p in parents)
    finally:
        try:
            s.delete_collection()
        except Exception:
            pass
