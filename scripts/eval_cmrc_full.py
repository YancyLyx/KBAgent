#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMRC2018 完整验证集评测（3219 条）+ HotpotQA 长文档评测

用法：
  python3 scripts/eval_cmrc_full.py --cmrc-only    # 只跑 CMRC 全量
  python3 scripts/eval_cmrc_full.py --hotpot-only  # 只跑 HotpotQA
  python3 scripts/eval_cmrc_full.py                # 跑全部
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

project_root = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════
# Chunker（复用项目配置）
# ═══════════════════════════════════════════════════════════

class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: dict = None) -> List[Dict]:
        if metadata is None:
            metadata = {}
        chunks = self._recursive_split(text)
        return [{
            "content": c,
            "chunk_id": f"{metadata.get('source', 'doc')}_{i}",
            "chunk_index": i,
            "metadata": dict(metadata) if metadata else {},
        } for i, c in enumerate(chunks)]

    def _recursive_split(self, text: str) -> List[str]:
        return self._split_with_separators(text, ["\n\n", "\n", "。", "，", " ", ""], 0)

    def _split_with_separators(self, text: str, seps: List[str], depth: int) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if depth >= len(seps):
            return self._split_by_size(text)
        sep = seps[depth]
        if sep == "":
            return self._split_by_size(text)
        chunks, current = [], ""
        for part in text.split(sep):
            if not part.strip():
                continue
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_with_separators(part, seps, depth + 1))
                else:
                    current = part.strip()
        if current:
            chunks.append(current)
        merged = []
        for c in chunks:
            if merged and len(merged[-1]) < self.chunk_size * 0.5:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)
        return merged

    def _split_by_size(self, text: str) -> List[str]:
        r, overlap = [], min(self.chunk_overlap, self.chunk_size // 2)
        start = 0
        while start < len(text):
            c = text[start:start + self.chunk_size]
            if c.strip():
                r.append(c)
            start += self.chunk_size - overlap
        return r


# ═══════════════════════════════════════════════════════════
# Embedding & BM25
# ═══════════════════════════════════════════════════════════

class EmbeddingEngine:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  加载 BGE: {model_name} → {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                o = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(o.last_hidden_state.size()).float()
            emb = (o.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)


class SimpleBM25:
    def __init__(self, corpus: List[str]):
        self.corpus = [self._tokenize(d) for d in corpus]
        self.avg_dl = sum(len(d) for d in self.corpus) / max(len(self.corpus), 1)
        n = len(self.corpus)
        df = {}
        for doc in self.corpus:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'[\w]+', text.lower())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        k1, b = 1.5, 0.75
        qt = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.corpus):
            s, dl = 0.0, len(doc)
            for q in qt:
                if q not in self.idf:
                    continue
                tf = doc.count(q)
                s += self.idf[q] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avg_dl))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════

class LightRetriever:
    def __init__(self, docs: List[Dict], embedder: EmbeddingEngine):
        self.docs = docs
        self.contents = [d["content"] for d in docs]
        self.embedder = embedder
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[SimpleBM25] = None

    def build_index(self):
        print("  建向量索引...")
        t0 = time.perf_counter()
        self.embeddings = self.embedder.encode(self.contents, batch_size=16)
        print(f"    完成: {len(self.contents)} 条, {time.perf_counter()-t0:.1f}s")
        self.bm25 = SimpleBM25(self.contents)

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        idxs = np.argsort(-np.dot(self.embeddings, q_emb))[:top_k]
        return [{"id": str(i), "content": self.contents[i], "score": float(np.dot(self.embeddings, q_emb)[i]),
                 "metadata": self.docs[i].get("metadata", {}), "search_type": "vector"} for i in idxs]

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        sims = np.dot(self.embeddings, q_emb)
        vi = np.argsort(-sims)[:top_k * 2]
        bm = self.bm25.search(query, top_k=top_k * 2) if self.bm25 else []
        rrf = {}
        for r, i in enumerate(vi):
            rrf[int(i)] = rrf.get(int(i), 0) + 1.0 / (rrf_k + r + 1)
        for r, (i, _) in enumerate(bm):
            rrf[i] = rrf.get(i, 0) + 1.0 / (rrf_k + r + 1)
        sidx = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": str(i), "content": self.contents[i], "score": s,
                 "metadata": self.docs[i].get("metadata", {}), "search_type": "hybrid"}
                for i, s in sidx]


# ═══════════════════════════════════════════════════════════
# Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════

class LightReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  加载 Reranker: {model_name} → {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        import torch
        if not docs:
            return []
        pairs = [[query, d["content"]] for d in docs]
        inp = self.tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            sc = self.model(**inp).logits.squeeze(-1).cpu().numpy()
        sc = [float(sc)] if sc.ndim == 0 else sc.tolist()
        for i, d in enumerate(docs):
            d["rerank_score"] = sc[i] if i < len(sc) else 0.0
        return sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def hit_rate(results, gts, k):
    hits = 0
    for res, gt in zip(results, gts):
        if any(gt in r.get("content", "") for r in res[:k]):
            hits += 1
    return hits / len(results) if results else 0.0

def mrr(results, gts):
    rrs = []
    for res, gt in zip(results, gts):
        for rk, r in enumerate(res, 1):
            if gt in r.get("content", ""):
                rrs.append(1.0 / rk)
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0

def rel_label(content, query, gt):
    if gt in content:
        return 2
    return 1 if len(set(re.findall(r'[\w]+', query.lower())) & set(re.findall(r'[\w]+', content.lower()))) >= 2 else 0

def ndcg(results, gts, queries, k):
    ndcgs = []
    for res, gt, q in zip(results, gts, queries):
        rels = [rel_label(r.get("content", ""), q, gt) for r in res[:k]]
        # DCG = sum(rel_i / log2(i+1)), i从0开始
        dcg_val = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        ideal = sum(sorted(rels, reverse=True)[i] / math.log2(i + 2) for i in range(len(rels)))
        ndcgs.append(dcg_val / ideal if ideal > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ═══════════════════════════════════════════════════════════
# CMRC 评测
# ═══════════════════════════════════════════════════════════

def run_cmrc():
    print("\n" + "█" * 60)
    print("  CMRC2018 完整验证集评测 (3219 queries)")
    print("█" * 60)

    ds = load_dataset("clue", "cmrc2018", trust_remote_code=True)
    data = ds["validation"]  # 3219 条
    print(f"\n[1] 加载: {len(data)} queries")

    queries = [s["question"] for s in data]
    gt = [s["answers"]["text"][0] for s in data]
    ctxs = [s["context"] for s in data]

    chunker = Chunker(500, 100)
    docs = []
    for i, c in enumerate(ctxs):
        docs.extend(chunker.split_text(c, {"source": f"d{i}"}))
    print(f"[2] 文档块: {len(docs)} (chunk_size=500)")

    # 验证保留率
    retained = sum(1 for i, a in enumerate(gt)
                   if any(a in d["content"] for d in docs if d["metadata"]["source"] == f"d{i}"))
    print(f"    答案保留率: {retained}/{len(gt)} = {retained/len(gt):.1%}")

    embedder = EmbeddingEngine()
    retriever = LightRetriever(docs, embedder)
    retriever.build_index()

    print(f"\n[3] 检索中...")
    rv, rh = [], []
    lt_v, lt_h = [], []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        rv.append(retriever.vector_search(q, 10))
        lt_v.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        rh.append(retriever.hybrid_search(q, 10))
        lt_h.append(time.perf_counter() - t0)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(queries)}")

    print(f"    [3b] 重排序中...")
    reranker = LightReranker()
    rr, lt_r = [], []
    for i, (q, h) in enumerate(zip(queries, rh)):
        t0 = time.perf_counter()
        rr.append(reranker.rerank(q, h, 5))
        lt_r.append(time.perf_counter() - t0)
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(queries)}")

    return _print_results("CMRC2018", queries, gt, rv, rh, rr, lt_v, lt_h, lt_r, len(docs))


# ═══════════════════════════════════════════════════════════
# HotpotQA 评测
# ═══════════════════════════════════════════════════════════

def run_hotpot(max_samples: int = 1000):
    print("\n" + "█" * 60)
    print(f"  HotpotQA 验证集评测 ({max_samples} queries, 英文·多文档)")
    print("█" * 60)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    data = ds["validation"]
    if max_samples < len(data):
        import random; random.seed(42)
        idxs = sorted(random.sample(range(len(data)), max_samples))
        data = data.select(idxs)

    queries, gt, all_ctx_texts = [], [], []
    for s in data:
        queries.append(s["question"])
        gt.append(s["answer"])
        all_sents = []
        for sent_list in s["context"]["sentences"]:
            all_sents.extend(sent_list)
        all_ctx_texts.append(" ".join(all_sents))

    print(f"\n[1] 加载: {len(queries)} queries")

    chunker = Chunker(500, 100)
    docs = []
    for i, c in enumerate(all_ctx_texts):
        docs.extend(chunker.split_text(c, {"source": f"d{i}"}))
    print(f"[2] 文档块: {len(docs)} (chunk_size=500)")

    # 验证保留率 (只看非 yes/no 的)
    retained, total_non_yn = 0, 0
    for i, (a, c) in enumerate(zip(gt, all_ctx_texts)):
        if a.lower() in ("yes", "no"):
            continue
        total_non_yn += 1
        doc_chunks = [d for d in docs if d["metadata"]["source"] == f"d{i}"]
        if any(a in d["content"] for d in doc_chunks):
            retained += 1
    print(f"    答案保留率(非yes/no): {retained}/{total_non_yn} = {retained/total_non_yn:.1%}")

    embedder = EmbeddingEngine("BAAI/bge-small-en-v1.5")  # 英文版
    retriever = LightRetriever(docs, embedder)
    retriever.build_index()

    print(f"\n[3] 检索中...")
    rv, rh = [], []
    lt_v, lt_h = [], []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        rv.append(retriever.vector_search(q, 10))
        lt_v.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        rh.append(retriever.hybrid_search(q, 10))
        lt_h.append(time.perf_counter() - t0)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(queries)}")

    print(f"    [3b] 重排序中...")
    reranker = LightReranker("BAAI/bge-reranker-base")  # 英文也适用
    rr, lt_r = [], []
    for i, (q, h) in enumerate(zip(queries, rh)):
        t0 = time.perf_counter()
        rr.append(reranker.rerank(q, h, 5))
        lt_r.append(time.perf_counter() - t0)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(queries)}")

    return _print_results("HotpotQA", queries, gt, rv, rh, rr, lt_v, lt_h, lt_r, len(docs))


# ═══════════════════════════════════════════════════════════
# 结果输出
# ═══════════════════════════════════════════════════════════

def _print_results(name, queries, gt, rv, rh, rr, lv, lh, lr, n_docs):
    strategies = {
        "纯向量检索 (Vector)": (rv, lv),
        "混合检索 (BM25+RRF)": (rh, lh),
        "混合+Cross-Encoder精排": (rr, lr),
    }

    report = {}
    print(f"\n{'='*60}")
    print(f"  {name} 评测结果")
    print(f"{'='*60}\n")

    hdr = f"{'策略':<28} | {'HR@1':<7} | {'HR@3':<7} | {'HR@5':<7} | {'MRR':<7} | {'NDCG@10':<8} | {'P50(ms)':<8} | {'P99(ms)'}"
    print(hdr)
    print("-" * len(hdr))

    for sname, (res, lat) in strategies.items():
        h1 = hit_rate(res, gt, 1)
        h3 = hit_rate(res, gt, 3)
        h5 = hit_rate(res, gt, 5)
        mr = mrr(res, gt)
        n10 = ndcg(res, gt, queries, 10)
        p50 = round(float(np.median(lat) * 1000), 1) if lat else 0
        p99 = round(float(np.percentile(lat, 99) * 1000), 1) if lat else 0
        report[sname] = {"HR@1": h1, "HR@3": h3, "HR@5": h5, "MRR": mr, "NDCG@10": n10, "P50(ms)": p50, "P99(ms)": p99}
        print(f"{sname:<28} | {h1:<7.2%} | {h3:<7.2%} | {h5:<7.2%} | {mr:<7.4f} | {n10:<8.4f} | {p50:<8} | {p99}")

    # 提升率
    keys = list(strategies.keys())
    b, p = report[keys[0]], report[keys[-1]]
    print(f"\n提升率（{keys[-1]} vs {keys[0]}）：")
    for m in ["HR@1", "HR@3", "HR@5", "MRR", "NDCG@10"]:
        if b.get(m, 0) > 0:
            pct = (p[m] - b[m]) / b[m] * 100
            print(f"  {m}: +{pct:.1f}%")

    report["_meta"] = {
        "dataset": name,
        "num_queries": len(queries),
        "num_docs": n_docs,
        "chunk_size": 500, "chunk_overlap": 100,
    }

    out = project_root / "data" / f"eval_report_{name.lower().replace(' ','_')}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告保存: {out}")
    return report


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmrc-only", action="store_true")
    ap.add_argument("--hotpot-only", action="store_true")
    ap.add_argument("--hotpot-samples", type=int, default=500)
    args = ap.parse_args()

    if args.hotpot_only:
        run_hotpot(args.hotpot_samples)
    elif args.cmrc_only:
        run_cmrc()
    else:
        run_cmrc()
        run_hotpot(args.hotpot_samples)
