# -*- coding: utf-8 -*-
"""Step1 诊断：多查询分解合并策略修复对比（老=先到先得，新=候选放大+RRF）。

样本：库中真实注入的 2doc/3doc 样本（各 8 条）。
链路：走与产品一致 RAGPipeline.retrieve（混合+精排+Autocut+MMR），tag=None。
指标：合并后 top-3 的 HR@1 / HR@3 / MRR（文档级：source 命中即算）。
门槛：新合并 MRR/HR@3 必须 >= 老合并，否则不合入产品。
"""
import json, os, asyncio, random, sys
from dotenv import load_dotenv
sys_path = "/Users/yanxinluo/Documents/PycharmProjects/KBAgent"
load_dotenv(f"{sys_path}/.env")
sys.path.insert(0, sys_path)

os.environ["VECTOR_STORE_BACKEND"] = "milvus"
os.environ["E2E_COLLECTION"] = "crud_rag"

from openai import AsyncOpenAI
from pymilvus import MilvusClient
from app.rag.rag_pipeline import RAGPipeline
from app.rag.milvus_store import MilvusVectorStore
from app.rag.result_merger import rrf_merge
from app.rag.query_expansion import decompose_query, generate_hypothetical_doc

MERGED = f"{sys_path}/data/crud_rag/repo/data/crud/merged_unzip/merged.json"
data = json.load(open(MERGED, encoding="utf-8"))
store = MilvusVectorStore("crud_rag", f"{sys_path}/config/rag_config.yaml", partition="known_items")
client = MilvusClient(uri="http://localhost:19530", timeout=10)
pipe = RAGPipeline(config_path=f"{sys_path}/config/rag_config.yaml", collection_name="crud_rag")
llm = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
model = os.getenv("LLM_MODEL", "deepseek-v4-pro")

SUB_TAG = {"questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}

def injected_ids(tag):
    ids, last = set(), None
    while True:
        flt = f'source like "known_item:{tag}:%"'
        if last: flt += f' and chunk_id > "{last}"'
        rows = client.query(collection_name="crud_rag", filter=flt, output_fields=["source","chunk_id"], limit=3000)
        if not rows: break
        for r in rows:
            rest = r["source"][len(f"known_item:{tag}:"):]
            if "_news" in rest: ids.add(rest.split("_news")[0])
            last = r["chunk_id"]
        if len(rows) < 3000: break
    return ids

def rels_of(s, tag):
    return [f"known_item:{tag}:{s['ID']}_news{n}" for n in (1,2,3)
            if isinstance(s.get(f"news{n}"), str) and s[f"news{n}"].strip()]

def source_of(doc):
    if isinstance(doc.get("source"), str) and doc["source"]:
        return doc["source"]
    return (doc.get("metadata") or {}).get("source", "")

def ranks_in(hits, rels):
    out = {}
    for i, h in enumerate(hits[:10]):
        src = source_of(h)
        if src in rels and src not in out:
            out[src] = i + 1
    return out

def metrics(ranks):
    if not ranks:
        return {"HR@1": 0, "HR@3": 0, "MRR": 0.0}
    m = min(ranks.values())
    return {"HR@1": 1 if m == 1 else 0, "HR@3": 1 if m <= 3 else 0, "MRR": round(1.0 / m, 3)}

def old_merge(results_per_query, top_k=3):
    merged, seen = [], set()
    for ctxs in results_per_query:
        for c in ctxs:
            key = c.get("content", "")[:80]
            if key and key not in seen:
                seen.add(key)
                merged.append(c)
    return merged[:top_k]

def first_seen_top3_ranks(candidates):
    per = []
    for cq in candidates:
        per.append(pipe.retrieve(cq, top_k=3, use_rerank=True, tag=None))
    return ranks_in(old_merge(per), rels)

def rrf_top3_ranks(candidates):
    per = []
    for cq in candidates:
        per.append(pipe.retrieve(cq, top_k=9, use_rerank=True, tag=None))
    merged = rrf_merge([p for p in per if p], top_k=12)
    # 与产品路径一致的仲裁：原 query 精排 top3（阈值/Autocut 沿用管线配置）
    merged = pipe.reranker.rerank(
        candidates[0], merged, top_k=3,
        threshold=pipe.min_rerank_score_default,
        autocut=pipe.autocut_enabled,
        drop_ratio=pipe.autocut_drop_ratio,
    )
    return ranks_in(merged, rels)

random.seed(42)
agg = {}
for sub, tag in SUB_TAG.items():
    ids = list(injected_ids(tag))
    by_id = {s["ID"]: s for s in data[sub]}
    samples = [by_id[i] for i in sorted(ids) if i in by_id][: (8 if sub == "questanswer_2docs" else 16)]
    a = {"old": {"HR@1":0,"HR@3":0,"MRR":0.0}, "new": {"HR@1":0,"HR@3":0,"MRR":0.0}}
    details = []
    for s in samples:
        q = s.get("questions",""); rels = rels_of(s, tag)
        if not q or not rels: continue
        subs = asyncio.run(decompose_query(llm, model, q))
        hyde = asyncio.run(generate_hypothetical_doc(llm, model, q))
        candidates = [q] + [x for x in subs if x != q]
        candidates = list(dict.fromkeys(candidates))
        if hyde: candidates.append(hyde)
        old_r = first_seen_top3_ranks(candidates)
        new_r = rrf_top3_ranks(candidates)
        mo, mn = metrics(old_r), metrics(new_r)
        for k in a["old"]: a["old"][k] += mo[k]
        for k in a["new"]: a["new"][k] += mn[k]
        details.append((q[:36], old_r, new_r))
    n = len(samples)
    for k in a["old"]: a["old"][k] = round(a["old"][k]/n, 3)
    for k in a["new"]: a["new"][k] = round(a["new"][k]/n, 3)
    agg[sub] = a
    print(f"\n=== {sub} (n={n}) ===")
    print(f"  OLD(先到先得, top3): {a['old']}")
    print(f"  NEW(候选9+RRF, top3): {a['new']}")
    for q, o, nn in details:
        print(f"    {q} old={o} new={nn}")

print("\n===== 门槛检查（NEW >= OLD 才合入） =====")
ok = True
for sub in agg:
    for k in ("MRR", "HR@3", "HR@1"):
        diff = agg[sub]["new"][k] - agg[sub]["old"][k]
        flag = "PASS" if diff >= 0 else "FAIL"
        if diff < 0: ok = False
        print(f"  {sub} {k}: old={agg[sub]['old'][k]} new={agg[sub]['new'][k]} diff={diff:+.3f} {flag}")
print("结论:", "通过，可合入产品" if ok else "未通过，先分析再改")
