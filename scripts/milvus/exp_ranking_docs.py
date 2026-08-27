# -*- coding: utf-8 -*-
"""实验1+2：2doc/3doc 排序问题的候选放宽 vs dense 优先路由。

检索层实验（不经过精排/阈值/Autocut/MMR）：
- base   : store.hybrid_search（每路 fetch=20，现状）
- fetch30: 每路 fetch=30 → RRF → top-10
- fetch50: 每路 fetch=50 → RRF → top-10
- dense  : 仅 dense（已有 Task2 数据背书）
- router : LLM 判断"跨文档聚合"→ true 走 dense，false 走 hybrid(base)

同时输出 router 的 LLM 判断准确率（对照 1doc/2docs/3docs 已知标签）。
样本：库中注入集 sorted ID（150/子集），全量 450。
"""
import json, os, sys, asyncio, time
from dotenv import load_dotenv
sys_path = "/Users/yanxinluo/Documents/PycharmProjects/KBAgent"
load_dotenv(f"{sys_path}/.env")
sys.path.insert(0, sys_path)
os.environ["VECTOR_STORE_BACKEND"] = "milvus"

from openai import AsyncOpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from app.rag.milvus_store import MilvusVectorStore

MERGED = f"{sys_path}/data/crud_rag/repo/data/crud/merged_unzip/merged.json"
OUT = f"{sys_path}/data/crud_rag/eval/exp_ranking_docs.json"
SUB_TAG = {"questanswer_1doc": "1doc", "questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}
SUBS = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
TOP_K = 10

data = json.load(open(MERGED, encoding="utf-8"))
store = MilvusVectorStore("crud_rag", f"{sys_path}/config/rag_config.yaml", partition="known_items")
client = MilvusClient(uri="http://localhost:19530", timeout=10)
llm = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
model = os.getenv("LLM_MODEL", "deepseek-v4-pro")


def injected_ids(tag):
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
    return sorted(ids)


def rels_of(s, tag):
    return [f"known_item:{tag}:{s['ID']}_news{n}" for n in (1, 2, 3)
            if isinstance(s.get(f"news{n}"), str) and s[f"news{n}"].strip()]


def src_of(doc):
    if isinstance(doc.get("source"), str) and doc["source"]:
        return doc["source"]
    return (doc.get("metadata") or {}).get("source", "")


def ranks_of(hits, rels):
    out = {}
    for i, h in enumerate(hits[:TOP_K]):
        src = src_of(h)
        if src in rels and src not in out:
            out[src] = i + 1
    return out


def metrics(ranks, nrels):
    if not ranks:
        return {"HR@1": 0, "HR@3": 0, "MRR": 0.0, "R@10": 0.0, "nDCG": 0.0}
    import math
    m = min(ranks.values())
    hr1 = 1 if any(p <= 1 for p in ranks.values()) else 0
    hr3 = 1 if any(p <= 3 for p in ranks.values()) else 0
    recall = sum(1 for p in ranks.values() if p <= 10) / nrels
    dcg = sum(1.0 / math.log2(p + 1) for p in ranks.values() if p <= 10)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(nrels, 10) + 1))
    return {"HR@1": hr1, "HR@3": hr3, "MRR": round(1.0 / m, 4), "R@10": round(recall, 4),
            "nDCG": round(dcg / idcg if idcg else 0.0, 4)}


def hybrid_fetch(q, fetch):
    qemb = store.encode_text([q])[0]
    req_d = AnnSearchRequest(data=[qemb], anns_field="dense",
                             param={"metric_type": "COSINE"}, limit=fetch)
    req_s = AnnSearchRequest(data=[q], anns_field="sparse",
                             param={"metric_type": "BM25"}, limit=fetch)
    res = client.hybrid_search(collection_name="crud_rag", reqs=[req_d, req_s],
                               ranker=RRFRanker(100), limit=TOP_K,
                               output_fields=["text", "tag", "source", "parent_id",
                                              "parent_section", "parent_content",
                                              "chunk_index", "title", "page_range",
                                              "content_types", "doc_type"])
    out = []
    for h in res[0]:
        ent = h.get("entity", {}) or {}
        out.append({"source": ent.get("source", ""), "content": ent.get("text", "")})
    return out


def dense_top10(q):
    hits = store.search(q, top_k=TOP_K)
    return [{"source": src_of(h)} for h in hits]


async def judge_multi_doc(q):
    prompt = (
        "判断下面的问题是否需要综合多篇文档/多个事实来源的信息才能完整回答"
        "（跨文档聚合）。单主题单事实问题（答案在一处）输出 false。\n"
        '只输出 JSON：{"multi_doc": true/false}\n\n'
        f"问题：{q}"
    )
    try:
        resp = await llm.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=50,
        )
        txt = (resp.choices[0].message.content or "").strip()
        if "true" in txt.lower():
            return True
        if "false" in txt.lower():
            return False
    except Exception:
        pass
    return None


async def main():
    t0 = time.time()
    samples = {}
    for sub in SUBS:
        by_id = {s["ID"]: s for s in data[sub]}
        samples[sub] = [by_id[i] for i in injected_ids(SUB_TAG[sub]) if i in by_id]

    agg = {variant: {sub: {"n": 0, "HR@1": 0, "HR@3": 0, "MRR": 0.0, "R@10": 0.0, "nDCG": 0.0}
                     for sub in SUBS} for variant in ("base", "fetch30", "fetch50", "dense", "router")}
    judge_stats = {"1doc_true": 0, "1doc_n": 0, "2doc_true": 0, "2doc_n": 0,
                   "3doc_true": 0, "3doc_n": 0, "none": 0}

    for sub in SUBS:
        tag = SUB_TAG[sub]
        for s in samples[sub]:
            q = s.get("questions", "")
            rels = rels_of(s, tag)
            if not q or not rels:
                continue
            hits_base = hybrid_fetch(q, 20)
            hits_30 = hybrid_fetch(q, 30)
            hits_50 = hybrid_fetch(q, 50)
            hits_dense = dense_top10(q)
            md = await judge_multi_doc(q)
            hits_router = hits_dense if md is True else hits_base
            rs = {"base": ranks_of(hits_base, rels), "fetch30": ranks_of(hits_30, rels),
                  "fetch50": ranks_of(hits_50, rels), "dense": ranks_of(hits_dense, rels),
                  "router": ranks_of(hits_router, rels)}
            for v, r in rs.items():
                m = metrics(r, len(rels))
                a = agg[v][sub]
                a["n"] += 1
                for k in ("HR@1", "HR@3"): a[k] += m[k]
                for k in ("MRR", "R@10", "nDCG"): a[k] += m[k]
            # 判断准确率（2docs/3docs 应为 true，1doc 应为 false）
            key = f"{SUB_TAG[sub]}_n"
            judge_stats[key] += 1
            if md is True:
                judge_stats[f"{SUB_TAG[sub]}_true"] += 1
            elif md is None:
                judge_stats["none"] += 1

    for v in agg:
        for sub in agg[v]:
            a = agg[v][sub]
            if a["n"]:
                for k in ("HR@1", "HR@3"): a[k] = round(a[k] / a["n"], 4)
                for k in ("MRR", "R@10", "nDCG"): a[k] = round(a[k] / a["n"], 4)

    print("\n===== 检索层排序实验（top-10 文档级） =====")
    print(f"{'子集':<22}{'variant':<9}{'HR@1':>7}{'HR@3':>7}{'MRR':>7}{'R@10':>7}{'nDCG':>7}")
    for sub in SUBS:
        for v in ("base", "fetch30", "fetch50", "dense", "router"):
            a = agg[v][sub]
            print(f"{sub:<22}{v:<9}{a['HR@1']:>7.3f}{a['HR@3']:>7.3f}{a['MRR']:>7.3f}"
                  f"{a['R@10']:>7.3f}{a['nDCG']:>7.3f}")

    print("\n===== LLM 跨文档判断准确率 =====")
    for sub in ("1doc", "2doc", "3doc"):
        n = judge_stats[f"{sub}_n"]
        true_n = judge_stats[f"{sub}_true"]
        expect = "false" if sub == "1doc" else "true"
        acc = (n - true_n) / n if expect == "false" else true_n / n
        print(f"  {sub}: true率={true_n}/{n}={true_n/n:.3f}（期望{expect}）")
    print(f"  judge 无输出: {judge_stats['none']}")

    out = {"meta": {"note": "检索层实验：候选放宽 vs dense 优先路由；不经精排",
                    "elapsed_s": round(time.time() - t0, 1)}, "agg": agg, "judge": judge_stats}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果存 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
