# -*- coding: utf-8 -*-
"""实验：per-fact 子查询 top-k 并集 + chunk 粒度诊断。

假设：2doc/3doc 低分是"一个 query 押不中多篇"，把问题拆成原子事实子查询、
每路取 top-2 来源做并集，覆盖度应高于单 query 的 R@10。

同时输出 chunk 诊断：相关文档在 base top-10 里命中几块、最佳块排名——
若普遍只有 1 块且排名低，chunk 粒度（300 字子块）就是嫌疑放大项。

检索层实验，不做精排；指标=覆盖度（相关文档被并集覆盖的比例）。
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
OUT = f"{sys_path}/data/crud_rag/eval/exp_perfact_union.json"
SUB_TAG = {"questanswer_1doc": "1doc", "questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}
SUBS = ["questanswer_2docs", "questanswer_3docs", "questanswer_1doc"]

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


def hybrid_fetch(q, fetch):
    qemb = store.encode_text([q])[0]
    req_d = AnnSearchRequest(data=[qemb], anns_field="dense",
                             param={"metric_type": "COSINE"}, limit=fetch)
    req_s = AnnSearchRequest(data=[q], anns_field="sparse",
                             param={"metric_type": "BM25"}, limit=fetch)
    res = client.hybrid_search(collection_name="crud_rag", reqs=[req_d, req_s],
                               ranker=RRFRanker(100), limit=10,
                               output_fields=["text", "source", "parent_id"])
    out = []
    for h in res[0]:
        ent = h.get("entity", {}) or {}
        out.append({"source": ent.get("source", ""), "chunk_id": h.get("chunk_id", ""),
                    "parent_id": ent.get("parent_id", "")})
    return out


async def decompose_facts(q):
    prompt = (
        "把下面的问题拆成独立的原子事实子问题。每个子问题只问一个事实/一个"
        "信息来源点，尽量短、聚焦。不要输出多余文字，只输出 JSON 数组。\n\n"
        f"问题：{q}"
    )
    try:
        resp = await llm.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=300,
        )
        txt = (resp.choices[0].message.content or "").strip()
        start, end = txt.find("["), txt.rfind("]")
        if start < 0 or end < 0:
            return [q]
        subs = json.loads(txt[start:end + 1])
        return [str(x).strip() for x in subs if str(x).strip()][:4]
    except Exception:
        return [q]


def union_sources(fact_hits):
    """每路取前 2 个不同来源，按出现顺序并集"""
    seen, out = set(), []
    for hits in fact_hits:
        cnt = 0
        for h in hits:
            src = h.get("source", "")
            if src and src not in seen:
                seen.add(src)
                out.append(src)
                cnt += 1
                if cnt >= 2:
                    break
    return out


async def main():
    t0 = time.time()
    agg = {sub: {"n": 0, "base_r10": 0.0, "union_cov": 0.0, "base_found_any": 0,
                 "union_found_any": 0, "chunk_single": 0, "chunk_multi": 0,
                 "best_chunk_ranks": []} for sub in SUBS}
    for sub in SUBS:
        tag = SUB_TAG[sub]
        by_id = {s["ID"]: s for s in data[sub]}
        for sid in injected_ids(tag):
            s = by_id.get(sid)
            if not s:
                continue
            q = s.get("questions", "")
            rels = rels_of(s, tag)
            if not q or not rels:
                continue
            # base top-10（chunk 级，含 chunk 诊断）
            base = hybrid_fetch(q, 20)
            base_srcs = []
            for r in base:
                if r["source"] not in base_srcs:
                    base_srcs.append(r["source"])
            base_r10 = sum(1 for rel in rels if rel in base_srcs[:10]) / len(rels)
            found_any_base = 1 if any(rel in base_srcs[:10] for rel in rels) else 0

            # chunk 诊断：相关文档在 top-10 命中块数与最佳块排名
            for rel in rels:
                pos = [i + 1 for i, h in enumerate(base) if h.get("source") == rel]
                if pos:
                    if len(pos) > 1:
                        agg[sub]["chunk_multi"] += 1
                    else:
                        agg[sub]["chunk_single"] += 1
                    agg[sub]["best_chunk_ranks"].append(min(pos))

            # per-fact 并集
            facts = await decompose_facts(q)
            fact_hits = [hybrid_fetch(f, 5) for f in facts]
            union = union_sources(fact_hits)
            union_cov = sum(1 for rel in rels if rel in union) / len(rels)
            found_any_union = 1 if any(rel in union for rel in rels) else 0

            a = agg[sub]
            a["n"] += 1
            a["base_r10"] += base_r10
            a["union_cov"] += union_cov
            a["base_found_any"] += found_any_base
            a["union_found_any"] += found_any_union

    print("\n===== per-fact 并集 vs base（覆盖度，文档级） =====")
    print(f"{'子集':<22}{'n':>5}{'baseR@10':>9}{'unionCov':>9}{'base有任一篇':>12}{'union有任一篇':>12}")
    for sub in SUBS:
        a = agg[sub]
        if a["n"] == 0:
            continue
        print(f"{sub:<22}{a['n']:>5}{a['base_r10']/a['n']:>9.3f}{a['union_cov']/a['n']:>9.3f}"
              f"{a['base_found_any']/a['n']:>12.3f}{a['union_found_any']/a['n']:>12.3f}")

    print("\n===== chunk 粒度诊断（相关文档在 base top-10 的表现） =====")
    for sub in SUBS:
        a = agg[sub]
        if a["n"] == 0:
            continue
        ranks = sorted(a["best_chunk_ranks"])
        med = ranks[len(ranks) // 2] if ranks else 0
        print(f"  {sub}: 命中块数=1 的样本 {a['chunk_single']} / 多块 {a['chunk_multi']}"
              f" | 最佳块排名 p50={med} p90={ranks[int(len(ranks)*0.9)] if ranks else 0}")

    out = {"meta": {"note": "per-fact 子查询 top-2 并集覆盖度 + chunk 粒度诊断；检索层不精排",
                    "elapsed_s": round(time.time() - t0, 1)}, "agg": {
        k: {kk: vv for kk, vv in v.items() if kk != "best_chunk_ranks"}
        for k, v in agg.items()}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n耗时 {time.time()-t0:.0f}s，结果存 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
