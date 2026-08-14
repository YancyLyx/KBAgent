#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test-corpus 评测脚本

用 test-corpus 的企业行政文档 + 测试问题，评估 RAG 检索效果。
"""

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─── Chunker（复用项目配置） ────────────────────────────

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


# ─── Embedding ──────────────────────────────────────────

class EmbeddingEngine:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  BGE: {model_name} → {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts[i:i+batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
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
            s = 0.0
            for t in set(qt):
                if t in self.idf:
                    tf = doc.count(t)
                    s += self.idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(doc) / self.avg_dl))
            scores.append((i, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


class LightRetriever:
    def __init__(self, docs: List[Dict], embedder: EmbeddingEngine):
        self.docs = docs
        self.contents = [d["content"] for d in docs]
        self.embedder = embedder
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[SimpleBM25] = None

    def build_index(self):
        print(f"  建索引: {len(self.contents)} 条", flush=True)
        self.embeddings = self.embedder.encode(self.contents, batch_size=16)
        self.bm25 = SimpleBM25(self.contents)

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        idxs = np.argsort(-np.dot(self.embeddings, q_emb))[:top_k]
        return [{"id": str(i), "content": self.contents[i],
                 "score": float(np.dot(self.embeddings, q_emb)[i]),
                 "metadata": self.docs[i].get("metadata", {}),
                 "search_type": "vector"} for i in idxs]

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


class LightReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Reranker: {model_name} → {self.device}", flush=True)
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


# ─── 指标 ──────────────────────────────────────────────

def hit_rate(results, gts, k):
    hits = 0
    for res, gt in zip(results, gts):
        gt_str = str(gt) if gt else ""
        if any(gt_str in r.get("content", "") for r in res[:k]):
            hits += 1
    return hits / len(results) if results else 0.0


def mrr(results, gts):
    rrs = []
    for res, gt in zip(results, gts):
        gt_str = str(gt) if gt else ""
        for rk, r in enumerate(res, 1):
            if gt_str in r.get("content", ""):
                rrs.append(1.0 / rk)
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


def ndcg(results, gts, queries, k):
    ndcgs = []
    for res, gt, q in zip(results, gts, queries):
        gt_str = str(gt) if gt else ""
        rels = []
        for r in res[:k]:
            if gt_str in r.get("content", ""):
                rels.append(2)
            elif len(set(re.findall(r'[\w]+', q.lower())) & set(re.findall(r'[\w]+', r.get("content", "").lower()))) >= 2:
                rels.append(1)
            else:
                rels.append(0)
        dcg_val = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        ideal = sum(sorted(rels, reverse=True)[i] / math.log2(i + 2) for i in range(len(rels)))
        ndcgs.append(dcg_val / ideal if ideal > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ─── 构建测试 QA 对 ─────────────────────────────────────

def build_test_qa(doc_dir: Path) -> List[Tuple[str, str]]:
    """从 test-corpus 文档中构建测试 (query, answer) 对"""
    qa_pairs = [
        # 02-差旅与报销管理办法-2025版
        ("一线城市出差住宿费每晚上限是多少？", "550 元"),
        ("出差伙食补助每天多少钱？", "130 元"),
        ("报销审批额度上限是多少？", "8000 元"),
        ("报销时限是多少天？", "60 天"),
        ("报销打款需要几个工作日？", "3 个工作日"),
        ("市内交通补贴每天上限是多少？", "120 元"),
        ("员工乘坐高铁优先选择什么座位？", "二等座"),
        ("飞行距离多少公里以上可乘坐经济舱？", "800 公里"),
        ("部门总监及以上可乘坐什么座位？", "高铁一等座"),
        ("出差申请须填写什么？", "预算金额"),
        ("超预算多少以上须重新说明？", "20%"),
        ("出差途中因不可抗力滞留费用如何报销？", "按本办法报销"),

        # 03-考勤与请假管理制度
        ("工龄满12年每年有几天年假？", "10 天"),
        ("试用期员工有带薪年假吗？", "无"),
        ("病假期间工资怎么计算？", "基本工资的 70%"),
        ("事假期间发工资吗？", "无薪假"),
        ("婚假多少天？", "10 天"),
        ("丧假多少天？", "3 天"),
        ("年假满20年以上每年多少天？", "15 天"),
        ("标准工作时间是几点到几点？", "09:00–18:00"),
        ("请假超过10个工作日应报哪里备案？", "人事部"),
        ("员工请假最小单位是什么？", "半天"),
        ("婚假应在结婚后多长时间内休完？", "一年内"),

        # 04-远程办公管理规定
        ("远程办公每周最多申请几天？", "2 天"),
        ("远程办公需要什么条件？", "绩效考核不低于 B"),
        ("公司为远程办公员工配发什么设备？", "笔记本电脑"),

        # 05-员工入职与转正管理办法
        ("试用期一般是多久？", "3 个月"),
        ("新员工入职引导安排在什么时间？", "入职当日"),

        # 06-IT设备与办公用品管理规定
        ("笔记本电脑多久更换一次？", "4 年"),
        ("IT设备报修后应在多长时间内响应？", "2 个工作日"),

        # 07-薪酬与绩效管理制度
        ("绩效工资占比多少？", "20%"),

        # 09-生育与陪产假管理规定
        ("陪产假有几天？", "15 天"),
        ("育儿假每年多少天？", "5 天"),

        # 10-员工持股与股权激励计划
        ("股权锁定期是多久？", "3 年"),
    ]
    return qa_pairs


# ─── 主流程 ─────────────────────────────────────────────

def evaluate():
    print("=" * 60, flush=True)
    print("  test-corpus RAG 检索评测", flush=True)
    print("=" * 60, flush=True)

    doc_dir = project_root / "test-corpus"

    # 1. 加载文档
    print(f"\n[1/4] 加载文档...", flush=True)
    md_files = sorted([f for f in doc_dir.iterdir() if f.suffix == '.md' and f.name != 'README.md'])
    print(f"  共 {len(md_files)} 篇文档", flush=True)

    docs = []
    for f in md_files:
        with open(f) as fh:
            text = fh.read()
        docs.append({"filename": f.name, "content": text, "title": f.stem})
        print(f"    {f.name}: {len(text)} 字符", flush=True)

    # 2. Chunker 分块
    print(f"\n[2/4] Chunker 分块 (chunk_size=500, overlap=100)...", flush=True)
    chunker = Chunker(500, 100)
    all_chunks = []
    for d in docs:
        chunks = chunker.split_text(d["content"], {"source": d["filename"], "title": d["title"]})
        all_chunks.extend(chunks)
    print(f"  共 {len(all_chunks)} 个文档块", flush=True)

    # 3. 准备 QA 对
    print(f"\n[3/4] 准备测试问题...", flush=True)
    qa_pairs = build_test_qa(doc_dir)
    print(f"  共 {len(qa_pairs)} 条测试问题", flush=True)

    # 验证答案在文档中的保留率
    retained = 0
    for q, a in qa_pairs:
        if any(a in d["content"] for d in all_chunks):
            retained += 1
    print(f"  答案保留率: {retained}/{len(qa_pairs)} = {retained/len(qa_pairs):.1%}", flush=True)

    # 4. 建索引
    print(f"\n[4/4] 建索引 + 检索...", flush=True)
    embedder = EmbeddingEngine()
    retriever = LightRetriever(all_chunks, embedder)
    retriever.build_index()

    queries = [q for q, a in qa_pairs]
    gt = [a for q, a in qa_pairs]

    # 三种策略检索
    rv, rh, rr = [], [], []
    lv, lh, lr = [], [], []

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        rv.append(retriever.vector_search(q, 10))
        lv.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rh.append(retriever.hybrid_search(q, 10))
        lh.append(time.perf_counter() - t0)

        if (i + 1) % 10 == 0:
            print(f"  检索进度: {i+1}/{len(queries)}", end="\r", flush=True)
    print(f"  检索完成: {len(queries)} 条", flush=True)

    print(f"  重排序中...", flush=True)
    reranker = LightReranker()
    for i, (q, h) in enumerate(zip(queries, rh)):
        t0 = time.perf_counter()
        rr.append(reranker.rerank(q, h, 5))
        lr.append(time.perf_counter() - t0)

    # 5. 计算结果
    print(f"\n{'='*60}", flush=True)
    print(f"  评测结果", flush=True)
    print(f"{'='*60}\n", flush=True)

    strategies = {
        "纯向量检索": (rv, lv),
        "混合检索 (BM25+RRF)": (rh, lh),
        "混合+Cross-Encoder精排": (rr, lr),
    }

    hdr = f"{'策略':<28} | {'HR@1':<7} | {'HR@3':<7} | {'MRR':<7} | {'P50(ms)'}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    report = {}
    for sname, (res, lat) in strategies.items():
        h1 = hit_rate(res, gt, 1)
        h3 = hit_rate(res, gt, 3)
        m = mrr(res, gt)
        p50 = round(float(np.median(lat) * 1000), 1) if lat else 0
        report[sname] = {"HR@1": h1, "HR@3": h3, "MRR": m}
        print(f"{sname:<28} | {h1:<7.2%} | {h3:<7.2%} | {m:<7.4f} | {p50}", flush=True)

    keys = list(strategies.keys())
    b, p = report[keys[0]], report[keys[-1]]
    print(f"\n提升率（{keys[-1]} vs {keys[0]}）：", flush=True)
    for m in ["HR@1", "HR@3", "MRR"]:
        if b.get(m, 0) > 0:
            pct = (p[m] - b[m]) / b[m] * 100
            print(f"  {m}: +{pct:.1f}%", flush=True)

    print(f"\n  文档: {len(md_files)} 篇, {len(all_chunks)} 个块", flush=True)
    print(f"  Query: {len(queries)} 条", flush=True)

    # 逐条输出结果
    print(f"\n{'='*60}", flush=True)
    print(f"  逐条检索结果（混合+精排 top-1）", flush=True)
    print(f"{'='*60}\n", flush=True)
    for i, q in enumerate(queries):
        top1 = rr[i][0]["content"] if rr[i] else ""
        hit = "✅" if gt[i] in top1 else "❌"
        print(f"  Q: {q}", flush=True)
        print(f"  A: {gt[i]}", flush=True)
        print(f"  Hit: {hit}", flush=True)

    # 保存报告
    out_dir = project_root / "data"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "eval_report_testcorpus.json"
    report["_meta"] = {
        "doc_count": len(md_files),
        "chunk_count": len(all_chunks),
        "query_count": len(queries),
        "gt_retention": retained / len(qa_pairs) if qa_pairs else 0,
    }
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out}", flush=True)
    print("评测完成!", flush=True)


if __name__ == "__main__":
    evaluate()
