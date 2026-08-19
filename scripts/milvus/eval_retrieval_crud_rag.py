# -*- coding: utf-8 -*-
"""Phase 2 任务2：CRUD-RAG 检索层评测（A1 已知项检索）。

范式：已知项注入（known-item injection），非 golden 文档命中。
干扰项池 = 86,834 篇 / 283,110 chunks（crud_rag / crud_news partition）。
对每个抽样样本的源文档（news1..newsN）走同一分块 pipeline 注入 crud_rag
的 known_items partition，打标 tag=crud_rag、source=known_item:{ID}_news{n}，
chunk_id 由分块器派生为 known_item:{ID}_news{n}_0_child_0（与 crud_ 前缀不冲突）。
用 question 作 query，hybrid_search（Milvus RRFRanker，生产路径），另跑 dense-only 对比。
指标：HR@1/3/5（文档级，top-k 含该样本任一 known item 即命中）、Recall@k
（该样本 N 个源文档被 top-k 覆盖比例）、MRR（样本级首命中倒数排名）、nDCG@10（多相关项）。
产物存 data/crud_rag/eval/（不提交）。miniforge3 python。
"""
import sys
import time
import json
import random
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.chunker import Chunker
from app.rag.parent_child_chunker import ParentChildChunker
from app.rag.milvus_store import MilvusVectorStore

SPLIT = project_root / "data/crud_rag/repo/data/crud_split/split_merged.json"
COLLECTION = "crud_rag"
KNOWN_PARTITION = "known_items"
CORPUS_PARTITION = "crud_news"
TOP_K = 10
READ_SUBS = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
SEED = 42
SAMPLE_PER_TASK = int(__import__("os").getenv("SAMPLE_PER_TASK", "150"))
OUT_DIR = project_root / "data/crud_rag/eval"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = json.load(open(SPLIT, encoding="utf-8"))

    # ---- Step 1: 抽样 ----
    print("=" * 60)
    print("Step 1: 抽样（seed=%d, 每任务 %d 条）" % (SEED, SAMPLE_PER_TASK))
    print("=" * 60)
    rng = random.Random(SEED)
    samples = {}
    total_q = 0
    for sub in READ_SUBS:
        pool = data[sub]
        n = min(SAMPLE_PER_TASK, len(pool))
        picked = rng.sample(pool, n)
        samples[sub] = picked
        nq = sum(1 for s in picked if isinstance(s.get("questions"), str) and s["questions"].strip())
        total_q += nq
        print(f"  {sub}: 池 {len(pool)} -> 抽 {len(picked)}, 有效 question {nq}")
    print(f"  三子集合计 question: {total_q}")

    # ---- Step 2: 已知项注入 ----
    print("\n" + "=" * 60)
    print("Step 2: 已知项注入")
    print("=" * 60)
    chunker = Chunker()
    pc = ParentChildChunker()
    inject_store = MilvusVectorStore(
        collection_name=COLLECTION, config_path="config/rag_config.yaml",
        partition=KNOWN_PARTITION,
    )
    # 注入前 count
    pre_count = inject_store.client.query(COLLECTION, output_fields=["count(*)"])[0]["count(*)"]
    print(f"注入前 count(*): {pre_count}")

    all_inject = []
    expected_sources = {}  # sub -> {sample_idx: [known_id...]}
    SUB_TAG = {"questanswer_1doc": "1doc", "questanswer_2docs": "2doc", "questanswer_3docs": "3doc"}
    for sub in READ_SUBS:
        expected_sources[sub] = {}
        for si, s in enumerate(samples[sub]):
            sid = s["ID"]
            stag = SUB_TAG[sub]
            rels = []
            for n in (1, 2, 3):
                key = f"news{n}"
                if key not in s:
                    continue
                text = s[key]
                if not isinstance(text, str) or not text.strip():
                    continue
                # 含子集前缀，避免同一 ID 跨子集 news 内容不同导致 chunk_id 碰撞覆盖
                kid = f"known_item:{stag}:{sid}_news{n}"
                rels.append(kid)
                meta = {"source": kid, "title": kid, "tag": "crud_rag", "doc_type": "txt"}
                parents = chunker.split_text(text, metadata=meta)
                children = pc.chunk(parents)
                for ch in children:
                    ch["source"] = kid
                    ch["tag"] = "crud_rag"
                    # chunk_id 已由分块器派生为 {source}_0_child_0，确保唯一
                    all_inject.append(ch)
            expected_sources[sub][si] = rels
    n_inject = len(all_inject)
    print(f"已知项 chunk 数: {n_inject}")
    print(f"已知项源文档数: {sum(len(v) for sub in expected_sources for v in expected_sources[sub].values())}")

    # 批量 upsert
    batch = 500
    for i in range(0, n_inject, batch):
        inject_store.add_documents(all_inject[i:i + batch], batch_size=len(all_inject[i:i + batch]))
    time.sleep(3)
    post_count = inject_store.client.query(COLLECTION, output_fields=["count(*)"])[0]["count(*)"]
    print(f"注入后 count(*): {post_count}")
    print(f"对账: post - pre = {post_count - pre_count} (应 == {n_inject})")
    ok_reconcile = (post_count - pre_count) == n_inject
    print(f"对账结果: {'PASS' if ok_reconcile else 'FAIL'}")

    # 已知项 tag 分布
    n_known_tag = 0
    n_known_src = 0
    # 用 source 前缀统计已知项 chunk
    r = inject_store.client.query(
        COLLECTION, filter='source like "known_item:%"',
        output_fields=["chunk_id"], limit=16384,
    )
    # 分页统计
    known_count = 0
    last_id = ""
    while True:
        flt = 'source like "known_item:%"'
        if last_id:
            flt = f'source like "known_item:%" and chunk_id > "{last_id}"'
        r = inject_store.client.query(COLLECTION, filter=flt, output_fields=["chunk_id"], limit=16384)
        if not r:
            break
        known_count += len(r)
        last_id = r[-1]["chunk_id"]
        if len(r) < 16384:
            break
    print(f"已知项 chunk（source like known_item:%）: {known_count} (应 == {n_inject})")

    # ---- Step 3: 检索评测 ----
    print("\n" + "=" * 60)
    print("Step 3: 检索评测（hybrid 生产路径 + dense-only 对比）")
    print("=" * 60)
    search_store = MilvusVectorStore(
        collection_name=COLLECTION, config_path="config/rag_config.yaml",
        partition=CORPUS_PARTITION,  # search 不限 partition，跨 crud_news+known_items
    )

    def eval_one(query, rels, mode):
        """mode: 'hybrid' or 'dense'. 返回各指标。rels=已知项 source id 列表"""
        try:
            if mode == "hybrid":
                hits = search_store.hybrid_search(query, top_k=TOP_K)
            else:
                hits = search_store.search(query, top_k=TOP_K)
        except Exception:
            return None
        # 取每个 hit 的 source
        retrieved = []
        for h in hits:
            src = h.get("source") or h.get("metadata", {}).get("source")
            retrieved.append(src)
        # 首次出现位置（按源文档去重）
        first_pos = {}
        for idx, src in enumerate(retrieved):
            if src in rels and src not in first_pos:
                first_pos[src] = idx + 1
        found = set(first_pos.keys())
        res = {
            "HR@1": 1 if any(p <= 1 for p in first_pos.values()) else 0,
            "HR@3": 1 if any(p <= 3 for p in first_pos.values()) else 0,
            "HR@5": 1 if any(p <= 5 for p in first_pos.values()) else 0,
            "MRR": (1.0 / min(first_pos.values())) if first_pos else 0.0,
            "Recall@1": len([p for p in first_pos.values() if p <= 1]) / len(rels),
            "Recall@3": len([p for p in first_pos.values() if p <= 3]) / len(rels),
            "Recall@5": len([p for p in first_pos.values() if p <= 5]) / len(rels),
            "Recall@10": len([p for p in first_pos.values() if p <= 10]) / len(rels),
        }
        # nDCG@10：二值相关，按首次出现位置
        dcg = sum(1.0 / (1 + p) for p in first_pos.values() if p <= 10)
        idcg = sum(1.0 / (1 + i) for i in range(1, min(len(rels), 10) + 1))
        res["nDCG@10"] = dcg / idcg if idcg > 0 else 0.0
        return res

    results = {}
    for mode in ["hybrid", "dense"]:
        print(f"\n--- {mode} ---")
        mode_results = {}
        for sub in READ_SUBS:
            agg = {k: 0.0 for k in
                   ["HR@1", "HR@3", "HR@5", "MRR", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "nDCG@10"]}
            n = 0
            t_sub = time.time()
            for si, s in enumerate(samples[sub]):
                q = s.get("questions")
                if not isinstance(q, str) or not q.strip():
                    continue
                rels = expected_sources[sub][si]
                if not rels:
                    continue
                r = eval_one(q, rels, mode)
                if r is None:
                    continue
                for k in agg:
                    agg[k] += r[k]
                n += 1
            if n == 0:
                continue
            for k in agg:
                agg[k] /= n
            agg["sub"] = sub
            agg["n"] = n
            agg["eval_time_s"] = time.time() - t_sub
            mode_results[sub] = agg
            print(f"  {sub}: n={n} HR@1={agg['HR@1']:.3f} HR@3={agg['HR@3']:.3f} HR@5={agg['HR@5']:.3f}"
                  f" MRR={agg['MRR']:.3f} R@5={agg['Recall@5']:.3f} R@10={agg['Recall@10']:.3f}"
                  f" nDCG@10={agg['nDCG@10']:.3f} ({agg['eval_time_s']:.1f}s)")
        results[mode] = mode_results

    # ---- Step 4: 汇报 ----
    print("\n" + "=" * 60)
    print("Step 4: Task 2 检索评测汇报（A1 已知项注入范式）")
    print("=" * 60)
    print(f"干扰项池: 86,834 篇 / 283,110 chunks (crud_rag / crud_news)")
    print(f"已知项: {n_inject} chunks, {sum(len(v) for sub in expected_sources for v in expected_sources[sub].values())} 源文档 (known_items partition)")
    print(f"抽样: seed={SEED}, 每任务 {SAMPLE_PER_TASK}, 合计 question {total_q}")
    print(f"对账: count {pre_count} -> {post_count}, 增量 {post_count-pre_count} == 注入 {n_inject}: {'PASS' if ok_reconcile else 'FAIL'}")
    print(f"范式说明: 已知项注入，非 golden 文档命中，不与 67 条 QA HR@1 对比")
    print()
    for mode in ["hybrid", "dense"]:
        if mode not in results:
            continue
        print(f"[{mode}]")
        print(f"{'子集':<20}{'n':>6}{'HR@1':>8}{'HR@3':>8}{'HR@5':>8}{'MRR':>8}{'R@5':>8}{'R@10':>8}{'nDCG@10':>9}")
        for sub in READ_SUBS:
            if sub not in results[mode]:
                continue
            r = results[mode][sub]
            print(f"{r['sub']:<20}{r['n']:>6}{r['HR@1']:>8.3f}{r['HR@3']:>8.3f}{r['HR@5']:>8.3f}"
                  f"{r['MRR']:>8.3f}{r['Recall@5']:>8.3f}{r['Recall@10']:>8.3f}{r['nDCG@10']:>9.3f}")
    print(f"\n总耗时: {time.time()-t0:.1f}s")

    out = {
        "paradigm": "known_item_injection",
        "distractor_pool": {"docs": 86834, "chunks": 283110, "partition": "crud_news"},
        "known_items": {"chunks": n_inject, "partition": "known_items"},
        "sampling": {"seed": SEED, "per_task": SAMPLE_PER_TASK, "total_questions": total_q},
        "reconcile": {"pre_count": pre_count, "post_count": post_count, "inject": n_inject, "pass": ok_reconcile},
        "results": results,
    }
    json.dump(out, open(OUT_DIR / "task2_retrieval_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"结果已存: {OUT_DIR / 'task2_retrieval_results.json'}")


if __name__ == "__main__":
    main()
