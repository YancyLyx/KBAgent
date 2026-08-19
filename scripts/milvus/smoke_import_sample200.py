# -*- coding: utf-8 -*-
"""Phase 2 冒烟：sample200 导入 + 检索链路验证 + 磁盘增量外推。

目的：
1. 验证分块/embedding/检索（hybrid_search）端到端链路通。
2. 实测 Milvus 磁盘增量，外推全量成本，供决策正式导入规模。
3. 用独立 collection crud_smoke，测完可删，不污染 crud_rag。

用 miniforge3 python。语料/产物不提交。
"""
import sys
import time
import os
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.chunker import Chunker
from app.rag.parent_child_chunker import ParentChildChunker
from app.rag.milvus_store import MilvusVectorStore

SAMPLE = project_root / "data/crud_rag/crud_rag_sample200.txt"
COLLECTION = "crud_smoke"
PARTITION = "smoke"


def du_milvus():
    total = 0
    mp = project_root / "data/milvus"
    for root, _, files in os.walk(mp):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def strip_prefix(line):
    idx = line.find("正文：")
    if idx >= 0:
        return line[idx + len("正文："):]
    return line


def main():
    t0 = time.time()
    docs = [strip_prefix(ln.rstrip("\n")) for ln in open(SAMPLE, encoding="utf-8") if ln.strip()]
    n_docs = len(docs)
    print(f"sample 文档数: {n_docs}")

    # 分块
    chunker = Chunker()
    pc = ParentChildChunker()
    all_chunks = []
    for di, doc in enumerate(docs):
        meta = {"source": f"smoke_{di:04d}", "title": f"smoke_{di:04d}",
                "tag": "smoke", "doc_type": "txt"}
        parents = chunker.split_text(doc, metadata=meta)
        children = pc.chunk(parents)
        all_chunks.extend(children)
    n_chunks = len(all_chunks)
    print(f"chunk 数: {n_chunks}（每文档 {n_chunks/n_docs:.2f} chunks）")

    # 磁盘 before
    disk_before = du_milvus()
    print(f"导入前 data/milvus 总占用: {disk_before/1024/1024:.1f} MB")

    # 入库（先删旧 collection）
    store = MilvusVectorStore(
        collection_name=COLLECTION, config_path="config/rag_config.yaml",
        partition=PARTITION,
    )
    try:
        store.client.drop_collection(COLLECTION)
        print(f"已删旧 {COLLECTION}")
        store = MilvusVectorStore(
            collection_name=COLLECTION, config_path="config/rag_config.yaml",
            partition=PARTITION,
        )
    except Exception:
        pass

    t_ingest = time.time()
    batch = 200
    for i in range(0, n_chunks, batch):
        store.add_documents(all_chunks[i:i + batch], batch_size=len(all_chunks[i:i + batch]))
    time.sleep(3)  # 等 growing 段可见
    ingest_dt = time.time() - t_ingest
    print(f"入库耗时: {ingest_dt:.1f}s")

    # 磁盘 after
    disk_after = du_milvus()
    delta = disk_after - disk_before
    print(f"导入后 data/milvus 总占用: {disk_after/1024/1024:.1f} MB")
    print(f"本次增量: {delta/1024/1024:.1f} MB")
    per_chunk = delta / n_chunks if n_chunks else 0
    print(f"每 chunk 占用（growing 段，未建索引）: {per_chunk/1024:.1f} KB")

    # 检索链路验证：取第 0 个文档的片段作 query
    sample_doc = docs[0]
    query = sample_doc[:30]  # 用文档开头 30 字作 query
    print(f"\n=== 检索验证 ===")
    print(f"query: {query}")
    for mode in ["dense", "bm25", "hybrid"]:
        try:
            if mode == "dense":
                hits = store.search(query, top_k=3)
            elif mode == "bm25":
                hits = store.bm25_search(query, top_k=3)
            else:
                hits = store.hybrid_search(query, top_k=3)
            print(f"\n[{mode}] 返回 {len(hits)} 条:")
            for h in hits[:3]:
                src = h.get("source") or h.get("metadata", {}).get("source")
                score = h.get("score") or h.get("distance")
                print(f"  source={src} score={score} text={h.get('content','')[:50]}")
        except Exception as e:
            print(f"[{mode}] 报错: {e}")

    # 外推
    FULL_CHUNKS = 283110
    full_growing = per_chunk * FULL_CHUNKS
    print(f"\n=== 外推（growing 段，未建索引）===")
    print(f"全量 {FULL_CHUNKS} chunks ≈ {full_growing/1024/1024:.1f} MB ({full_growing/1024/1024/1024:.2f} GB)")
    print(f"建 HNSW+BM25 索引后预计 ×1.3-1.6 ≈ {full_growing/1024/1024/1024*1.45:.1f} GB")
    print(f"总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
