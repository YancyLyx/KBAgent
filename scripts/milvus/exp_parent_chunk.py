# -*- coding: utf-8 -*-
"""chunk 粒度实验：子块(300) vs 整篇(1 chunk)。

同 2 万篇语料 + 同 450 已知项样本，两个独立集合：
- crud_chunk_child : 复刻生产（split_text 1000 -> parent-child 300/30）检索子块
- crud_chunk_parent: 整篇一个 chunk（文档级事实完整参与匹配）

控制变量：语料子集相同、已知项相同、查询相同，只差 chunk 粒度。
指标：文档级 top-10 HR@1/HR@3/MRR/R@10/nDCG，逐子集对比。
"""
import json, os, sys, time, random
from dotenv import load_dotenv
sys_path = "/Users/yanxinluo/Documents/PycharmProjects/KBAgent"
load_dotenv(f"{sys_path}/.env")
sys.path.insert(0, sys_path)
os.environ["VECTOR_STORE_BACKEND"] = "milvus"

from pathlib import Path
from app.rag.chunker import Chunker
from app.rag.parent_child_chunker import ParentChildChunker
from app.rag.milvus_store import MilvusVectorStore

DOCS_DIR = Path(f"{sys_path}/data/crud_rag/repo/data/80000_docs")
MERGED = f"{sys_path}/data/crud_rag/repo/data/crud/merged_unzip/merged.json"
OUT = f"{sys_path}/data/crud_rag/eval/exp_parent_chunk.json"
SUB_TAG = {"questanswer_1doc": "1doc", "questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}
SUBS = ["questanswer_2docs", "questanswer_3docs", "questanswer_1doc"]
CORPUS_N = 20000
TOP_K = 10

data = json.load(open(MERGED, encoding="utf-8"))


def load_corpus_bodies():
    lines = []
    for fp in sorted(DOCS_DIR.glob("documents_dup_*")):
        with open(fp, encoding="utf-8") as f:
            lines.extend(f.readlines())
    bodies = []
    for ln in lines:
        body = ln.split("正文：", 1)[-1].strip() if "正文：" in ln else ln.strip()
        if body:
            bodies.append(body)
    random.seed(42)
    return random.sample(bodies, min(CORPUS_N, len(bodies)))


def known_sources():
    """注入集样本 -> (sub, sample, tag, rels)"""
    items = []
    for sub in SUBS:
        tag = SUB_TAG[sub]
        by_id = {s["ID"]: s for s in data[sub]}
        # 复用注入集 ID（sorted 固定）
        ids = set()
        last = None
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530", timeout=10)
        while True:
            flt = f'source like "known_item:{tag}:%"'
            if last:
                flt += f' and chunk_id > "{last}"'
            rows = client.query(collection_name="crud_rag", filter=flt, output_fields=["source", "chunk_id"], limit=3000)
            if not rows:
                break
            for r in rows:
                rest = r["source"][len(f"known_item:{tag}:"):]
                if "_news" in rest:
                    ids.add(rest.split("_news")[0])
                last = r["chunk_id"]
            if len(rows) < 3000:
                break
        for sid in sorted(ids):
            s = by_id.get(sid)
            if not s:
                continue
            rels = [f"known_item:{tag}:{s['ID']}_news{n}" for n in (1, 2, 3)
                    if isinstance(s.get(f"news{n}"), str) and s[f"news{n}"].strip()]
            if s.get("questions") and rels:
                items.append((sub, s, rels))
    return items


def whole_chunk(body, source, tag):
    return {
        "content": body, "source": source, "tag": tag, "doc_type": "txt",
        "chunk_id": f"{source}_0", "chunk_index": 0, "title": "",
        "page_range": "", "content_types": [], "parent_id": "",
        "parent_content": "", "parent_section": "",
    }


def child_chunks(body, source, tag, chunker, pc):
    meta = {"source": source, "tag": tag, "doc_type": "txt"}
    parents = chunker.split_text(body, metadata=meta)
    return pc.chunk(parents)


def build_collection(mode, store, bodies, items, chunker, pc):
    print(f"[{mode}] 构建语料 chunks...")
    chunks = []
    for i, body in enumerate(bodies):
        src = f"crudp_{i:06d}"
        if mode == "parent":
            chunks.append(whole_chunk(body, src, "crud_rag"))
        else:
            chunks.extend(child_chunks(body, src, "crud_rag", chunker, pc))
    n_corpus = len(chunks)
    print(f"[{mode}] 语料 chunks={n_corpus}，注入已知项...")
    seen = set()
    for sub, s, rels in items:
        for rel in rels:
            if rel in seen:
                continue
            seen.add(rel)
            n = int(rel.rsplit("_news", 1)[1])
            text = s[f"news{n}"]
            if mode == "parent":
                chunks.append(whole_chunk(text, rel, "crud_rag"))
            else:
                chunks.extend(child_chunks(text, rel, "crud_rag", chunker, pc))
    print(f"[{mode}] 总 chunks={len(chunks)}，开始入库（upsert，batch=500）...")
    store.add_documents(chunks, batch_size=500)
    print(f"[{mode}] 入库完成，语料 {n_corpus} + 已知项 {len(chunks)-n_corpus}")
    return n_corpus


def src_of(doc):
    if isinstance(doc.get("source"), str) and doc["source"]:
        return doc["source"]
    return (doc.get("metadata") or {}).get("source", "")


def eval_collection(store, items):
    agg = {sub: {"n": 0, "HR@1": 0, "HR@3": 0, "MRR": 0.0, "R@10": 0.0, "nDCG": 0.0} for sub in SUBS}
    import math
    for sub, s, rels in items:
        q = s["questions"]
        hits = store.hybrid_search(q, top_k=TOP_K)
        ranks = {}
        for i, h in enumerate(hits):
            src = src_of(h)
            if src in rels and src not in ranks:
                ranks[src] = i + 1
        a = agg[sub]
        a["n"] += 1
        if ranks:
            m = min(ranks.values())
            a["HR@1"] += 1 if any(p <= 1 for p in ranks.values()) else 0
            a["HR@3"] += 1 if any(p <= 3 for p in ranks.values()) else 0
            a["MRR"] += 1.0 / m
        a["R@10"] += sum(1 for p in ranks.values() if p <= 10) / len(rels)
        dcg = sum(1.0 / math.log2(p + 1) for p in ranks.values() if p <= 10)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rels), 10) + 1))
        a["nDCG"] += dcg / idcg if idcg else 0.0
    for sub in agg:
        a = agg[sub]
        if a["n"]:
            for k in ("HR@1", "HR@3"): a[k] = round(a[k] / a["n"], 4)
            for k in ("MRR", "R@10", "nDCG"): a[k] = round(a[k] / a["n"], 4)
    return agg


def main():
    t0 = time.time()
    bodies = load_corpus_bodies()
    items = known_sources()
    print(f"语料 {len(bodies)} 篇，评测样本 {len(items)}（2doc/3doc/1doc）")
    chunker = Chunker(f"{sys_path}/config/rag_config.yaml")
    pc = ParentChildChunker(f"{sys_path}/config/rag_config.yaml")

    results = {}
    for mode in ("child", "parent"):
        col = f"crud_chunk_{mode}"
        store = MilvusVectorStore(col, f"{sys_path}/config/rag_config.yaml", partition="kb_default")
        n_corpus = build_collection(mode, store, bodies, items, chunker, pc)
        agg = eval_collection(store, items)
        results[mode] = {"corpus_chunks": n_corpus, "agg": agg}
        print(f"\n===== {mode} 集合指标 =====")
        for sub in SUBS:
            a = agg[sub]
            print(f"  {sub:<22} HR@1={a['HR@1']:.3f} HR@3={a['HR@3']:.3f} MRR={a['MRR']:.3f} R@10={a['R@10']:.3f} nDCG={a['nDCG']:.3f}")

    print("\n===== chunk 粒度对比（child vs parent） =====")
    print(f"{'子集':<22}{'指标':<7}{'child':>8}{'parent':>8}{'diff':>8}")
    for sub in SUBS:
        for k in ("HR@1", "HR@3", "MRR", "R@10", "nDCG"):
            c = results["child"]["agg"][sub][k]
            p = results["parent"]["agg"][sub][k]
            print(f"{sub:<22}{k:<7}{c:>8.3f}{p:>8.3f}{p-c:>+8.3f}")

    out = {"meta": {"corpus_n": len(bodies), "samples": len(items),
                    "elapsed_s": round(time.time() - t0, 1)}, "results": results}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果存 {OUT}")


if __name__ == "__main__":
    main()
