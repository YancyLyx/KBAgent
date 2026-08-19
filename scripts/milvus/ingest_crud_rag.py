# -*- coding: utf-8 -*-
"""Phase 2 任务1：CRUD-RAG 80k 新闻语料导入 Milvus crud_rag collection。

流程：
1. 读 data/crud_rag/repo/data/80000_docs/documents_dup_*（86834 行，一行一篇）
2. 去"时间戳，正文："前缀（无分隔符的行用整行，标题即有用内容）
3. 项目分块链路：Chunker(chunk_size=1000) → ParentChildChunker(child=300/overlap=30)
4. bge-small-zh-v1.5 embedding，批量 upsert 到 crud_rag collection / crud_news partition
5. 如实汇报：文档数/chunk 数/耗时/磁盘/超长报错

用 miniforge3 python。语料/产物不提交。
"""
import sys
import time
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.chunker import Chunker
from app.rag.parent_child_chunker import ParentChildChunker
from app.rag.milvus_store import MilvusVectorStore

DOCS_DIR = project_root / "data/crud_rag/repo/data/80000_docs"
COLLECTION = "crud_rag"
PARTITION = "crud_news"


def strip_prefix(line: str) -> str:
    """去掉'时间戳，正文：'前缀；无分隔符则用整行（标题即有用内容）"""
    idx = line.find("正文：")
    if idx >= 0:
        return line[idx + len("正文："):]
    return line


def main():
    t0 = time.time()

    # 1. 读取全部文档
    print("读取 80k 语料...")
    docs = []
    for fp in sorted(DOCS_DIR.glob("documents_dup_*")):
        for ln in open(fp, encoding="utf-8"):
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            docs.append(strip_prefix(ln))
    n_docs = len(docs)
    print(f"文档数: {n_docs}（耗时 {time.time()-t0:.1f}s）")

    # 2. 分块
    print("分块中...")
    chunker = Chunker()
    pc = ParentChildChunker()
    t_chunk = time.time()
    all_chunks = []
    overlong = []  # 记录超长 chunk（text>16384 或 parent_content>65535）
    start_doc = int(os.getenv("START_DOC", "0"))
    if start_doc:
        print(f"resume: 从 doc {start_doc} 开始（前 {start_doc} 个文档的 chunk 已 upsert，幂等覆盖）")
    for di, doc in enumerate(docs):
        if di < start_doc:
            continue
        meta = {"source": f"crud_{di:06d}", "title": f"crud_{di:06d}",
                "tag": "crud_rag", "doc_type": "txt"}
        parents = chunker.split_text(doc, metadata=meta)
        children = pc.chunk(parents)
        for ch in children:
            tlen = len(ch.get("content", ""))
            plen = len(ch.get("parent_content", ""))
            if tlen > 16384 or plen > 65535:
                overlong.append((di, tlen, plen))
            all_chunks.append(ch)
    n_chunks = len(all_chunks)
    print(f"待入库 chunk 数: {n_chunks}（分块耗时 {time.time()-t_chunk:.1f}s）")
    if overlong:
        print(f"⚠ 超长 chunk: {len(overlong)} 条，示例: {overlong[:3]}")
    else:
        print("无超长 chunk（text≤16384, parent_content≤65535）")

    # 3. 入库
    print(f"入库到 collection={COLLECTION} partition={PARTITION}...")
    t_ingest = time.time()
    store = MilvusVectorStore(
        collection_name=COLLECTION, config_path="config/rag_config.yaml",
        partition=PARTITION,
    )
    # 分批 upsert，捕获超长报错；磁盘充足后无需 sleep
    batch_size = 500
    errors = []
    for i in range(0, n_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        try:
            store.add_documents(batch, batch_size=len(batch))
        except Exception as e:
            errors.append((i, str(e)[:200]))
            if len(errors) <= 3:
                print(f"  批次 {i} 报错: {e}")
            # 连接类错误：等待 Milvus 恢复再继续
            if "connect" in str(e).lower() or "unavailable" in str(e).lower():
                print("  检测到连接错误，等待 30s 后继续...")
                time.sleep(30)
    ingest_dt = time.time() - t_ingest
    print(f"入库耗时: {ingest_dt:.1f}s")

    # 4. 统计
    time.sleep(2)
    stats = store.get_collection_stats()
    disk = 0
    mp = project_root / "data/milvus"
    for root, _, files in os.walk(mp):
        for f in files:
            disk += os.path.getsize(os.path.join(root, f))
    disk_mb = disk / 1024 / 1024

    print("\n" + "=" * 60)
    print("Task 1 导入汇报")
    print("=" * 60)
    print(f"文档数: {n_docs}")
    print(f"chunk 数: {n_chunks}")
    print(f"入库耗时: {ingest_dt:.1f}s")
    print(f"总耗时: {time.time()-t0:.1f}s")
    print(f"Milvus 磁盘占用: {disk_mb:.1f} MB")
    print(f"get_collection_stats: {stats}")
    print(f"超长 chunk 报错: {len(overlong)} 条")
    print(f"入库批次报错: {len(errors)} 条")
    if errors:
        print(f"首批错误: {errors[0]}")


if __name__ == "__main__":
    main()
