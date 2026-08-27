# -*- coding: utf-8 -*-
"""全量父块验证：86,834 篇整篇粒度 + 450 已知项 → 450 查询检索层指标。

对照基线 = crud_rag（子块 283,110 chunks）全量池：2docs HR@1 0.327 / 3docs 0.293 / 1doc 0.733。
若父块全量池收益保持（2docs 明显、其余不降），则产品分块策略可决策改为整篇/语义父块。
"""
import json, os, sys, time
from dotenv import load_dotenv
sys_path = "/Users/yanxinluo/Documents/PycharmProjects/KBAgent"
load_dotenv(f"{sys_path}/.env")
sys.path.insert(0, sys_path)
os.environ["VECTOR_STORE_BACKEND"] = "milvus"

from pathlib import Path
from pymilvus import MilvusClient
from app.rag.milvus_store import MilvusVectorStore

DOCS_DIR = Path(f"{sys_path}/data/crud_rag/repo/data/80000_docs")
MERGED = f"{sys_path}/data/crud_rag/repo/data/crud/merged_unzip/merged.json"
OUT = f"{sys_path}/data/crud_rag/eval/exp_parent_full.json"
SUB_TAG = {"questanswer_1doc": "1doc", "questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}
SUBS = ["questanswer_2docs", "questanswer_3docs", "questanswer_1doc"]
COLLECTION = "crud_parent_full"
TOP_K = 10

data = json.load(open(MERGED, encoding="utf-8"))
client = MilvusClient(uri="http://localhost:19530", timeout=10)


def load_all_bodies():
    lines = []
    for fp in sorted(DOCS_DIR.glob("documents_dup_*")):
        with open(fp, encoding="utf-8") as f:
            lines.extend(f.readlines())
    bodies = []
    for ln in lines:
        body = ln.split("正文：", 1)[-1].strip() if "正文：" in ln else ln.strip()
        if body:
            bodies.append(body)
    return bodies


def injected_samples():
    items = []
    for sub in SUBS:
        tag = SUB_TAG[sub]
        by_id = {s["ID"]: s for s in data[sub]}
        ids, last = set(), None
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


def whole_chunk(body, source):
    return {
        "content": body, "source": source, "tag": "crud_rag", "doc_type": "txt",
        "chunk_id": f"{source}_0", "chunk_index": 0, "title": "",
        "page_range": "", "content_types": [], "parent_id": "",
        "parent_content": "", "parent_section": "",
    }


def src_of(doc):
    if isinstance(doc.get("source"), str) and doc["source"]:
        return doc["source"]
    return (doc.get("metadata") or {}).get("source", "")


def main():
    t0 = time.time()
    bodies = load_all_bodies()
    items = injected_samples()
    print(f"全量语料 {len(bodies)} 篇，样本 {len(items)}")

    store = MilvusVectorStore(COLLECTION, f"{sys_path}/config/rag_config.yaml", partition="kb_default")
    chunks = [whole_chunk(b, f"crudp_{i:06d}") for i, b in enumerate(bodies)]
    seen = set()
    for sub, s, rels in items:
        for rel in rels:
            if rel in seen:
                continue
            seen.add(rel)
            n = int(rel.rsplit("_news", 1)[1])
            chunks.append(whole_chunk(s[f"news{n}"], rel))
    print(f"总 chunks={len(chunks)}（语料 {len(bodies)} + 已知项 {len(seen)}），入库...")
    store.add_documents(chunks, batch_size=500)
    print("入库完成，开始评测...")

    import math
    agg = {sub: {"n": 0, "HR@1": 0, "HR@3": 0, "MRR": 0.0, "R@10": 0.0, "nDCG": 0.0} for sub in SUBS}
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

    baseline = {"questanswer_2docs": {"HR@1": 0.327, "MRR": 0.527, "R@10": 0.740},
                "questanswer_3docs": {"HR@1": 0.293, "MRR": 0.511, "R@10": 0.631},
                "questanswer_1doc": {"HR@1": 0.733, "MRR": 0.832, "R@10": 0.980}}
    print("\n===== 全量父块 vs 全量子块基线（crud_rag） =====")
    print(f"{'子集':<22}{'指标':<7}{'子块基线':>9}{'父块全量':>9}{'diff':>8}")
    for sub in SUBS:
        for k in ("HR@1", "MRR", "R@10"):
            b = baseline[sub][k]
            p = agg[sub][k]
            print(f"{sub:<22}{k:<7}{b:>9.3f}{p:>9.3f}{p-b:>+8.3f}")

    out = {"meta": {"corpus": len(bodies), "samples": len(items),
                    "collection": COLLECTION, "elapsed_s": round(time.time() - t0, 1)},
           "parent_full": agg}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果存 {OUT}")


if __name__ == "__main__":
    main()
