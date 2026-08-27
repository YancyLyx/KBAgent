# -*- coding: utf-8 -*-
"""Step2：完整链路检索评测（增强开 vs 关，三档对比）。

T1 裸 store：store.hybrid_search（Task 2 原口径，无任何增强）
T2 RAGPipeline：混合检索 + 精排 + 阈值 + Autocut + MMR
T3 产品等价链路：LLM 判断拆分 → 多路并行 retrieve(top-9) → RRF(top-12)
   → 原 query 精排仲裁；LLM 判断无需拆分时走 Self-RAG（充分性判断 → 改写补检）
   → 最终精排。指标统一 top-10 文档级（source 命中）。

样本：库中真实注入的 150/150/150（sorted ID 固定，落盘 task2_sample_ids.json）。
产物：data/crud_rag/eval/task2_full_pipeline_compare.json（不提交）。
"""
import json, os, asyncio, sys, time
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
OUT_DIR = f"{sys_path}/data/crud_rag/eval"
SUB_TAG = {"questanswer_1doc": "1doc", "questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}
SUBS = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
N_LIMIT = int(os.getenv("N_LIMIT", "0"))  # 0 = 全量 150/子集；>0 仅取前 N（冒烟）

data = json.load(open(MERGED, encoding="utf-8"))
store = MilvusVectorStore("crud_rag", f"{sys_path}/config/rag_config.yaml", partition="known_items")
client = MilvusClient(uri="http://localhost:19530", timeout=10)
pipe = RAGPipeline(config_path=f"{sys_path}/config/rag_config.yaml", collection_name="crud_rag")
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


def ranks_of(hits, rels, top=10):
    out = {}
    for i, h in enumerate(hits[:top]):
        src = src_of(h)
        if src in rels and src not in out:
            out[src] = i + 1
    return out


def metrics(ranks, nrels):
    if not ranks:
        return {"HR@1": 0, "HR@3": 0, "HR@5": 0, "MRR": 0.0, "Recall@10": 0.0, "nDCG@10": 0.0}
    m = min(ranks.values())
    hr = {k: 1 if any(p <= k for p in ranks.values()) else 0 for k in (1, 3, 5)}
    recall = sum(1 for p in ranks.values() if p <= 10) / nrels
    dcg = sum(1.0 / __import__("math").log2(p + 1) for p in ranks.values() if p <= 10)
    idcg = sum(1.0 / __import__("math").log2(i + 1) for i in range(1, min(nrels, 10) + 1))
    return {"HR@1": hr[1], "HR@3": hr[3], "HR@5": hr[5],
            "MRR": round(1.0 / m, 4), "Recall@10": round(recall, 4),
            "nDCG@10": round(dcg / idcg if idcg else 0.0, 4)}


async def t1_ranks(q, rels):
    return ranks_of(store.hybrid_search(q, top_k=10), rels)


def t2_ranks(q, rels):
    hits = pipe.retrieve(q, top_k=10, use_rerank=True, tag=None)
    return ranks_of(hits, rels)


async def t3_ranks(q, rels):
    """产品等价：拆得动走 expansion；拆不动走 Self-RAG。"""
    subs = await decompose_query(llm, model, q)
    if len(subs) > 1 or (len(subs) == 1 and subs[0] != q):
        candidates = [q] + [x for x in subs if x != q]
        candidates = list(dict.fromkeys(candidates))
        hyde = await generate_hypothetical_doc(llm, model, q)
        if hyde:
            candidates.append(hyde)
        gathered = await asyncio.gather(*(
            asyncio.to_thread(pipe.retrieve, cq, 9, True, None) for cq in candidates
        ), return_exceptions=True)
        merged = rrf_merge([g for g in gathered if isinstance(g, list)], top_k=12)
        if not merged:
            return {}
        final = pipe.reranker.rerank(
            q, merged, top_k=10,
            threshold=getattr(pipe, "min_rerank_score_default", None),
            autocut=getattr(pipe, "autocut_enabled", False),
            drop_ratio=getattr(pipe, "autocut_drop_ratio", 0.3),
        )
        return ranks_of(final, rels)

    # Self-RAG 路径：首次检索 → 充分性判断 → 不足则改写补检 → RRF → 精排
    first = pipe.retrieve(q, top_k=3, use_rerank=True, tag=None)
    if not first:
        return {}
    prompt = (
        "你是检索质量评估器。判断下面的检索片段是否足以回答用户问题。\n\n"
        f"用户问题：{q}\n\n检索片段：\n{first}\n\n"
        '仅输出 JSON：{"sufficient": true/false, '
        '"rewritten_query": "不足时改写后的检索词（足够时填空）"}'
    )
    resp = await llm.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=150,
    )
    try:
        judge = json.loads((resp.choices[0].message.content or "").strip())
    except Exception:
        judge = {}
    if judge.get("sufficient") is True:
        return ranks_of(first, rels)
    rewritten = str(judge.get("rewritten_query") or "").strip()
    if not rewritten or rewritten == q:
        return ranks_of(first, rels)
    rounds = [first]
    seen = {c.get("content", "")[:80] for c in first}
    for _ in range(1):
        second = pipe.retrieve(rewritten, top_k=3, use_rerank=True, tag=None)
        if not second:
            break
        new_keys = {c.get("content", "")[:80] for c in second} - seen
        if not new_keys:
            break
        seen |= new_keys
        rounds.append(second)
    merged = rrf_merge(rounds, top_k=12)
    final = pipe.reranker.rerank(
        q, merged, top_k=10,
        threshold=getattr(pipe, "min_rerank_score_default", None),
        autocut=getattr(pipe, "autocut_enabled", False),
        drop_ratio=getattr(pipe, "autocut_drop_ratio", 0.3),
    )
    return ranks_of(final, rels)


async def main():
    limit = N_LIMIT or 150
    samples_all = {}
    for sub in SUBS:
        by_id = {s["ID"]: s for s in data[sub]}
        samples_all[sub] = [by_id[i] for i in injected_ids(SUB_TAG[sub]) if i in by_id][:limit]

    # 落盘样本 ID（可复现性）
    ids_out = {sub: [s["ID"] for s in samples_all[sub]] for sub in SUBS}
    with open(f"{OUT_DIR}/task2_sample_ids.json", "w", encoding="utf-8") as f:
        json.dump({"seed_fixed": "injected_set_sorted", "subs": ids_out}, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    agg = {tier: {sub: {"n": 0, "HR@1": 0, "HR@3": 0, "HR@5": 0, "MRR": 0.0, "Recall@10": 0.0, "nDCG@10": 0.0}
                  for sub in SUBS} for tier in ("T1_raw", "T2_pipeline", "T3_full")}
    per_query = []
    for sub in SUBS:
        tag = SUB_TAG[sub]
        for s in samples_all[sub]:
            q = s.get("questions", "")
            rels = rels_of(s, tag)
            if not q or not rels:
                continue
            r1 = await t1_ranks(q, rels)
            r2 = t2_ranks(q, rels)
            r3 = await t3_ranks(q, rels)
            m1, m2, m3 = metrics(r1, len(rels)), metrics(r2, len(rels)), metrics(r3, len(rels))
            for tier, m in (("T1_raw", m1), ("T2_pipeline", m2), ("T3_full", m3)):
                a = agg[tier][sub]
                a["n"] += 1
                for k in ("HR@1", "HR@3", "HR@5"): a[k] += m[k]
                for k in ("MRR", "Recall@10", "nDCG@10"): a[k] += m[k]
            per_query.append({"sub": sub, "ID": s["ID"], "q": q[:60],
                              "T1": r1, "T2": r2, "T3": r3})
    for tier in agg:
        for sub in agg[tier]:
            a = agg[tier][sub]
            if a["n"]:
                for k in ("HR@1", "HR@3", "HR@5"): a[k] = round(a[k] / a["n"], 4)
                for k in ("MRR", "Recall@10", "nDCG@10"): a[k] = round(a[k] / a["n"], 4)

    print("\n===== 三档对比（文档级 top-10 口径） =====")
    print(f"{'子集':<22}{'档位':<12}{'n':>5}{'HR@1':>8}{'HR@3':>8}{'HR@5':>8}{'MRR':>8}{'R@10':>8}{'nDCG':>8}")
    for sub in SUBS:
        for tier in ("T1_raw", "T2_pipeline", "T3_full"):
            a = agg[tier][sub]
            print(f"{sub:<22}{tier:<12}{a['n']:>5}{a['HR@1']:>8.3f}{a['HR@3']:>8.3f}"
                  f"{a['HR@5']:>8.3f}{a['MRR']:>8.3f}{a['Recall@10']:>8.3f}{a['nDCG@10']:>8.3f}")

    out = {"meta": {"paradigm": "known_item_injection", "sample_source": "injected_set_sorted",
                    "top_k": 10, "limit_per_sub": limit, "elapsed_s": round(time.time() - t0, 1)},
           "agg": agg, "per_query_count": len(per_query)}
    with open(f"{OUT_DIR}/task2_full_pipeline_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果已存 task2_full_pipeline_compare.json + task2_sample_ids.json")


if __name__ == "__main__":
    asyncio.run(main())
