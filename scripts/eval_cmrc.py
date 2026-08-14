#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMRC2018 评测脚本

在 CMRC2018 中文阅读理解数据集上评测 KB-Agent RAG 系统的检索效果。

评测设计：
- 知识库：CMRC2018 validation set 的 context 按项目 chunk 策略分块后建索引
- Query：CMRC2018 的 question
- Ground Truth：answers.text[0]（标准答案）是否在检索结果中
- 三种策略对比：纯向量 / 混合检索(RRF) / 混合+Cross-Encoder精排

脚本完全独立运行，仅依赖 transformers + torch + datasets + numpy。
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

# ─── 复现项目的 Chunker 逻辑 ────────────────────────────

class Chunker:
    """复现项目 app/rag/chunker.py 的 RecursiveCharacterTextSplitter 逻辑
    chunk_size=500, chunk_overlap=100, separators=["\n\n","\n","。","，"," ",""]
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: dict = None) -> List[Dict]:
        if metadata is None:
            metadata = {}
        chunks = self._recursive_split(text)
        result = []
        for idx, chunk in enumerate(chunks):
            result.append({
                "content": chunk,
                "chunk_id": f"{metadata.get('source', 'doc')}_{idx}",
                "chunk_index": idx,
                "metadata": dict(metadata) if metadata else {}
            })
        return result

    def _recursive_split(self, text: str) -> List[str]:
        """RecursiveCharacterTextSplitter 简化实现"""
        separators = ["\n\n", "\n", "。", "，", " ", ""]
        return self._split_with_separators(text, separators, 0)

    def _split_with_separators(self, text: str, separators: List[str], depth: int) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if depth >= len(separators):
            return self._split_by_size(text)

        sep = separators[depth]
        if sep == "":
            return self._split_by_size(text)

        chunks = []
        parts = text.split(sep)
        current = ""

        for part in parts:
            if not part.strip():
                continue
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_with_separators(part, separators, depth + 1)
                    chunks.extend(sub_chunks)
                else:
                    current = part.strip()

        if current:
            chunks.append(current)

        # 合并过短的块（处理 overlap）
        merged = []
        for c in chunks:
            if merged and len(merged[-1]) < self.chunk_size * 0.5:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)
        return merged

    def _split_by_size(self, text: str) -> List[str]:
        chunks = []
        overlap_size = min(self.chunk_overlap, self.chunk_size // 2)
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - overlap_size
        return chunks


# ─── Embedding 引擎 ─────────────────────────────────────

class EmbeddingEngine:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  加载 BGE 模型 ({model_name}) → {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print(f"  模型加载完成")

    @staticmethod
    def _mean_pooling(last_hidden, attention_mask):
        import torch
        token_emb = last_hidden
        mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        return (token_emb * mask).sum(1) / mask.sum(1)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in batch]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embs = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs)


# ─── BM25 ───────────────────────────────────────────────

class SimpleBM25:
    def __init__(self, corpus: List[str]):
        self.corpus = [self._tokenize(d) for d in corpus]
        self.avg_dl = sum(len(d) for d in self.corpus) / max(len(self.corpus), 1)
        n_docs = len(self.corpus)
        self.idf = {}
        df = {}
        for doc in self.corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'[\w]+', text.lower())

    def score(self, query: List[str], doc: List[str]) -> float:
        k1, b = 1.5, 0.75
        score = 0.0
        dl = len(doc)
        for q in query:
            if q not in self.idf:
                continue
            tf = doc.count(q)
            score += self.idf[q] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avg_dl))
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_tokens = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.corpus):
            s = self.score(q_tokens, doc)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─── 检索器 ─────────────────────────────────────────────

class LightRetriever:
    def __init__(self, docs: List[Dict], embedder: EmbeddingEngine):
        self.docs = docs
        self.contents = [d["content"] for d in docs]
        self.embedder = embedder
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[SimpleBM25] = None

    def build_index(self):
        print("  构建向量索引...")
        t0 = time.perf_counter()
        self.embeddings = self.embedder.encode(self.contents, batch_size=16)
        print(f"  向量索引完成: {len(self.contents)} 条, 耗时 {time.perf_counter() - t0:.1f}s")
        print("  构建 BM25 索引...")
        t0 = time.perf_counter()
        self.bm25 = SimpleBM25(self.contents)
        print(f"  BM25 索引完成, 耗时 {time.perf_counter() - t0:.1f}s")

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        sims = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(-sims)[:top_k]
        return [{
            "id": str(idx),
            "content": self.contents[idx],
            "metadata": self.docs[idx].get("metadata", {}),
            "score": float(sims[idx]),
            "search_type": "vector",
        } for idx in top_indices]

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        sims = np.dot(self.embeddings, q_emb)
        vec_indices = np.argsort(-sims)[:top_k * 2]
        bm25_results = self.bm25.search(query, top_k=top_k * 2) if self.bm25 else []

        rrf_scores = {}
        for rank, idx in enumerate(vec_indices):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{
            "id": str(idx),
            "content": self.contents[idx],
            "metadata": self.docs[idx].get("metadata", {}),
            "score": score,
            "search_type": "hybrid",
        } for idx, score in sorted_indices[:top_k]]


# ─── Cross-Encoder 重排序器 ─────────────────────────────

class LightReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  加载 Cross-Encoder 模型 ({model_name}) → {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        print(f"  模型加载完成")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        import torch
        if not documents:
            return []
        pairs = [[query, d["content"]] for d in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = outputs.logits.squeeze(-1).cpu().numpy().tolist()
        if isinstance(scores, float):
            scores = [scores]
        for i, d in enumerate(documents):
            d["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0
        sorted_docs = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return sorted_docs[:top_k]


# ─── 指标计算 ───────────────────────────────────────────

def hit_rate_at_k(results_list: List[List[Dict]], ground_truths: List[str], k: int) -> float:
    """正确答案是否在 top-k 检索结果中（按 content 匹配）"""
    hits = 0
    for results, gt in zip(results_list, ground_truths):
        contents = [r.get("content", "") for r in results[:k]]
        if any(gt in c for c in contents):
            hits += 1
    return hits / len(results_list) if results_list else 0.0


def mrr(results_list: List[List[Dict]], ground_truths: List[str]) -> float:
    """第一个正确答案出现位置的倒数平均"""
    rrs = []
    for results, gt in zip(results_list, ground_truths):
        found = False
        for rank, r in enumerate(results, 1):
            if gt in r.get("content", ""):
                rrs.append(1.0 / rank)
                found = True
                break
        if not found:
            rrs.append(0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


def _relevance_label(content: str, query: str, gt_answer: str) -> int:
    if gt_answer in content:
        return 2
    q_words = set(re.findall(r'[\w]+', query.lower()))
    c_words = set(re.findall(r'[\w]+', content.lower()))
    return 1 if len(q_words & c_words) >= 2 else 0


def dcg(scores: List[int], k: int) -> float:
    scores = scores[:k]
    if not scores:
        return 0.0
    result = float(scores[0])
    for i, s in enumerate(scores[1:], 2):
        result += s / math.log2(i)
    return result


def ndcg_at_k(results_list: List[List[Dict]], ground_truths: List[str],
              queries: List[str], k: int) -> float:
    ndcgs = []
    for results, gt, q in zip(results_list, ground_truths, queries):
        rels = [_relevance_label(r.get("content", ""), q, gt) for r in results[:k]]
        actual = dcg(rels, k)
        ideal = dcg(sorted(rels, reverse=True), k)
        ndcgs.append(actual / ideal if ideal > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ─── 主流程 ─────────────────────────────────────────────

def run_evaluation(split: str = "validation", max_samples: int = 500):
    print("=" * 65)
    print("  CMRC2018 RAG 检索效果评测")
    print("=" * 65)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据集: {split}, max_samples={max_samples}")
    ds = load_dataset("clue", "cmrc2018", trust_remote_code=True)
    data = ds[split]
    if max_samples and max_samples < len(data):
        data = data.select(range(max_samples))
    print(f"  → {len(data)} 条 query")

    queries = [s["question"] for s in data]
    gt_answers = [s["answers"]["text"][0] for s in data]
    raw_contexts = [s["context"] for s in data]

    # 2. 用项目一致的 Chunker 构建文档库
    print(f"\n[2/4] 构建文档库 (Chunker: chunk_size=500, overlap=100)")
    chunker = Chunker(chunk_size=500, chunk_overlap=100)
    all_docs = []
    for i, ctx in enumerate(raw_contexts):
        chunks = chunker.split_text(ctx, metadata={"source": f"cmrc_doc_{i}"})
        all_docs.extend(chunks)
    print(f"  → {len(all_docs)} 个文档块")

    # 验证 Chunker 对答案的保留率（上限基准）
    gt_retained = 0
    for i, ans in enumerate(gt_answers):
        doc_chunks = [d for d in all_docs if d["metadata"]["source"] == f"cmrc_doc_{i}"]
        if any(ans in d["content"] for d in doc_chunks):
            gt_retained += 1
    retention = gt_retained / len(gt_answers)
    print(f"  → 答案保留率: {gt_retained}/{len(gt_answers)} = {retention:.1%} (理论 Hit Rate 上限)")

    # 3. 建索引
    print(f"\n[3/4] 初始化 Embedding + BM25 索引...")
    embedder = EmbeddingEngine()
    retriever = LightRetriever(all_docs, embedder)
    retriever.build_index()

    # 4. 执行检索
    print(f"\n[4/4] 执行三种检索策略...")

    results_vector: List[List[Dict]] = []
    results_hybrid: List[List[Dict]] = []
    results_rerank: List[List[Dict]] = []
    latencies = {"vector": [], "hybrid": [], "rerank": []}

    total = len(queries)
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        r1 = retriever.vector_search(q, top_k=10)
        latencies["vector"].append(time.perf_counter() - t0)
        results_vector.append(r1)

        t0 = time.perf_counter()
        r2 = retriever.hybrid_search(q, top_k=10)
        latencies["hybrid"].append(time.perf_counter() - t0)
        results_hybrid.append(r2)

        if (i + 1) % 100 == 0:
            print(f"  向量 + 混合检索进度: {i + 1}/{total}")

    # 重排序
    print("  执行 Cross-Encoder 重排序...")
    reranker = LightReranker()
    for i, (q, r2) in enumerate(zip(queries, results_hybrid)):
        t0 = time.perf_counter()
        r3 = reranker.rerank(q, r2, top_k=5)
        latencies["rerank"].append(time.perf_counter() - t0)
        results_rerank.append(r3)
        if (i + 1) % 100 == 0:
            print(f"  重排序进度: {i + 1}/{total}")

    # 5. 计算指标
    print("\n" + "=" * 65)
    print("  评测结果")
    print("=" * 65)

    strategies = {
        "纯向量检索 (Vector)": results_vector,
        "混合检索 (BM25+Vector, RRF)": results_hybrid,
        "混合+Cross-Encoder精排": results_rerank,
    }

    report = {}
    for name, res in strategies.items():
        h1 = hit_rate_at_k(res, gt_answers, 1)
        h3 = hit_rate_at_k(res, gt_answers, 3)
        h5 = hit_rate_at_k(res, gt_answers, 5)
        m = mrr(res, gt_answers)
        n5 = ndcg_at_k(res, gt_answers, queries, 5)
        n10 = ndcg_at_k(res, gt_answers, queries, 10)

        report[name] = {
            "Hit Rate@1": round(h1, 4),
            "Hit Rate@3": round(h3, 4),
            "Hit Rate@5": round(h5, 4),
            "MRR": round(m, 4),
            "NDCG@5": round(n5, 4),
            "NDCG@10": round(n10, 4),
        }

        if "Vector" in name and "混合" not in name:
            lt = latencies["vector"]
            report[name]["P50延迟(ms)"] = round(float(np.median(lt) * 1000), 1)
            report[name]["P99延迟(ms)"] = round(float(np.percentile(lt, 99) * 1000), 1)
        elif "混合" in name and "精排" not in name:
            lt = latencies["hybrid"]
            report[name]["P50延迟(ms)"] = round(float(np.median(lt) * 1000), 1)
            report[name]["P99延迟(ms)"] = round(float(np.percentile(lt, 99) * 1000), 1)
        elif "精排" in name:
            total_times = [a + b for a, b in zip(latencies["hybrid"], latencies["rerank"])]
            report[name]["P50延迟(ms)"] = round(float(np.median(total_times) * 1000), 1)
            report[name]["P99延迟(ms)"] = round(float(np.percentile(total_times, 99) * 1000), 1)

    # 输出表格
    header = f"{'策略':<28} | {'HR@1':<7} | {'HR@3':<7} | {'HR@5':<7} | {'MRR':<7} | {'NDCG@5':<7} | {'NDCG@10':<8} | {'P50(ms)':<8} | {'P99(ms)'}"
    print()
    print(header)
    print("-" * len(header))
    for name in strategies:
        r = report[name]
        print(
            f"{name:<28} | "
            f"{r['Hit Rate@1']:<7.2%} | "
            f"{r['Hit Rate@3']:<7.2%} | "
            f"{r['Hit Rate@5']:<7.2%} | "
            f"{r['MRR']:<7.4f} | "
            f"{r['NDCG@5']:<7.4f} | "
            f"{r['NDCG@10']:<8.4f} | "
            f"{r.get('P50延迟(ms)', '-'):<8} | "
            f"{r.get('P99延迟(ms)', '-')}"
        )

    # 提升率
    keys = list(strategies.keys())
    base, best = report[keys[0]], report[keys[-1]]
    print(f"\n提升率（{keys[-1]} vs {keys[0]}）：")
    for metric in ["Hit Rate@1", "Hit Rate@3", "Hit Rate@5", "MRR", "NDCG@5", "NDCG@10"]:
        if base.get(metric, 0) > 0:
            pct = (best[metric] - base[metric]) / base[metric] * 100
            print(f"  {metric}: +{pct:.1f}%")

    print(f"\n  数据集: CMRC2018 ({split}, {len(queries)} queries)")
    print(f"  文档块: {len(all_docs)} (Chunker: chunk_size=500, overlap=100)")
    print(f"  答案保留率: {retention:.1%}")
    print(f"  模型: BAAI/bge-small-zh-v1.5 + BAAI/bge-reranker-base")

    # 保存
    out_path = Path(__file__).parent.parent / "data" / "cmrc_eval_report.json"
    report["_meta"] = {
        "dataset": f"clue/cmrc2018 ({split})",
        "num_queries": len(queries),
        "num_docs": len(all_docs),
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "reranker_model": "BAAI/bge-reranker-base",
        "gt_retention_rate": retention,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {out_path}")
    print("评测完成!")

    return report


if __name__ == "__main__":
    run_evaluation()
