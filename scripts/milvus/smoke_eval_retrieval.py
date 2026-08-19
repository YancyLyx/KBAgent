# -*- coding: utf-8 -*-
"""Task 2 检索评测冒烟：小语料 + 注入 3 个已知项，调通 HR/MRR/Recall/nDCG 计算。

目的：在正式导入完成前，用小集合验证评测逻辑正确（已知项能被召回、指标计算对）。
- 语料：sample200 的 200 文档 → crud_smoke_eval collection
- 已知项：从 merged.json questanswer_1doc 取 3 个样本，注入其 news1
- query：各样本的 question
- 验证：已知项应被召回（HR=1），指标计算无异常
"""
import sys, time, json, math
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from app.rag.chunker import Chunker
from app.rag.parent_child_chunker import ParentChildChunker
from app.rag.milvus_store import MilvusVectorStore

SAMPLE = project_root / "data/crud_rag/crud_rag_sample200.txt"
MERGED = project_root / "data/crud_rag/repo/data/crud/merged_unzip/merged.json"
COLLECTION = "crud_smoke_eval"
PARTITION = "smoke"


def strip_prefix(line):
    i = line.find("正文：")
    return line[i + len("正文："):] if i >= 0 else line


def inject(store, kid, text, chunker, pc):
    meta = {"source": kid, "title": kid, "tag": "known_item", "doc_type": "txt"}
    chunks = pc.chunk(chunker.split_text(text, metadata=meta))
    for ch in chunks:
        ch["source"] = kid
    if chunks:
        store.add_documents(chunks, batch_size=len(chunks))
    return len(chunks)


def metrics_for_query(hits, rels, top_k=10):
    """计算单 query 指标。rels=相关源文档 known_id 集合。"""
    first_pos = {}
    for idx, h in enumerate(hits):
        src = h.get("source") or h.get("metadata", {}).get("source")
        if src in rels and src not in first_pos:
            first_pos[src] = idx + 1
    found = set(first_pos.keys())
    hr = {k: 1 if any(p <= k for p in first_pos.values()) else 0 for k in (1, 3, 5)}
    mrr = 1.0 / min(first_pos.values()) if first_pos else 0.0
    recall = {k: len([p for p in first_pos.values() if p <= k]) / len(rels) for k in (1, 3, 5, 10)}
    # nDCG@10 二值相关
    dcg = sum(1.0 / math.log2(p + 1) for p in first_pos.values() if p <= 10)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rels), 10) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return hr, mrr, recall, ndcg, found, first_pos


def main():
    t0 = time.time()
    docs = [strip_prefix(ln.rstrip("\n")) for ln in open(SAMPLE, encoding="utf-8") if ln.strip()]
    data = json.load(open(MERGED, encoding="utf-8"))
    chunker = Chunker()
    pc = ParentChildChunker()

    store = MilvusVectorStore(COLLECTION, "config/rag_config.yaml", partition=PARTITION)
    try:
        store.client.drop_collection(COLLECTION)
        store = MilvusVectorStore(COLLECTION, "config/rag_config.yaml", partition=PARTITION)
    except Exception:
        pass

    # 1. 导入 200 语料
    chunks = []
    for di, doc in enumerate(docs):
        meta = {"source": f"smoke_{di:04d}", "title": f"smoke_{di:04d}", "tag": "crud_rag", "doc_type": "txt"}
        chunks.extend(pc.chunk(chunker.split_text(doc, metadata=meta)))
    print(f"语料 chunk 数: {len(chunks)}")
    for i in range(0, len(chunks), 200):
        store.add_documents(chunks[i:i+200], batch_size=len(chunks[i:i+200]))

    # 2. 注入 3 个已知项
    samples = data["questanswer_1doc"][:3]
    injected = []
    for s in samples:
        sid = s["ID"]
        kid = f"known_item:{sid}_news1"
        n = inject(store, kid, s["news1"], chunker, pc)
        injected.append((sid, kid, s["questions"], n))
        print(f"注入 {kid}: {n} chunks, question={s['questions'][:40]}")
    time.sleep(3)

    # 3. 逐 query 评测
    print("\n=== 检索评测结果 ===")
    for sid, kid, q, n in injected:
        rels = {kid}
        hits = store.hybrid_search(q, top_k=10)
        hr, mrr, recall, ndcg, found, first_pos = metrics_for_query(hits, rels)
        print(f"\n[{sid}] query={q[:30]}")
        print(f"  相关项: {rels}")
        print(f"  找到: {found}, 首次位置: {first_pos}")
        print(f"  HR@1={hr[1]} HR@3={hr[3]} HR@5={hr[5]} MRR={mrr:.3f}")
        print(f"  Recall@1={recall[1]:.2f} Recall@5={recall[5]:.2f} Recall@10={recall[10]:.2f} nDCG@10={ndcg:.3f}")
        print(f"  top3 source: {[h.get('metadata',{}).get('source') for h in hits[:3]]}")

    print(f"\n总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
