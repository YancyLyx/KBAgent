#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test-corpus 评测脚本

用 test-corpus 的企业行政文档 + 测试问题，评估 RAG 检索效果。
"""

import hashlib
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

# ─── 项目真实 MarkdownChunker（按标题语义分块）+ ParentChildChunker ──

def merge_elements_into_chunks(elements, chunk_size, *, metadata=None, source_name="doc"):
    if metadata is None:
        metadata = {}
    source_label = metadata.get("source", source_name)
    chunks = []
    section_stack = []
    buf_text, buf_size, buf_pages, buf_types = [], 0, set(), set()
    chunk_idx = 0
    def flush():
        nonlocal buf_text, buf_size, buf_pages, buf_types, chunk_idx
        if not buf_text:
            return
        sec_path = " > ".join(s for s in section_stack if s)
        pages = sorted(buf_pages)
        chunks.append({
            "content": "\n\n".join(buf_text),
            "chunk_id": f"{source_label}_{chunk_idx}",
            "chunk_index": chunk_idx,
            "section": sec_path,
            "page_range": (
                f"{pages[0]}-{pages[-1]}" if len(pages) > 1
                else str(pages[0]) if pages else ""
            ),
            "content_types": sorted(buf_types),
            **metadata,
        })
        chunk_idx += 1
        buf_text, buf_size, buf_pages, buf_types = [], 0, set(), set()
    for elem in elements:
        if elem.get("type") == "heading":
            flush()
            lvl = elem.get("level", 2)
            while len(section_stack) < lvl:
                section_stack.append("")
            section_stack[lvl - 1] = elem["text"]
            section_stack = section_stack[:lvl]
            continue
        text = elem["text"]
        etype = elem["type"]
        if buf_size + len(text) > chunk_size and buf_size > 0:
            flush()
        buf_text.append(text)
        buf_size += len(text)
        buf_pages.add(elem.get("page_num", 1))
        buf_types.add(etype)
    flush()
    return chunks


class MarkdownChunker:
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    def chunk_text(self, text: str, metadata: dict = None) -> List[Dict]:
        if metadata is None:
            metadata = {}
        elements = self._parse_md(text)
        return merge_elements_into_chunks(
            elements, self.chunk_size,
            metadata=metadata,
            source_name=metadata.get("source", "md"),
        )
    def _parse_md(self, text: str) -> List[Dict]:
        lines = text.split("\n")
        elements: List[Dict] = []
        i = 0
        in_code_block = False
        code_block_buffer: List[str] = []
        code_block_lang = ""
        _para_break = True
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_block_buffer = []
                    code_block_lang = line.strip()[3:].strip()
                    i += 1
                    continue
                else:
                    in_code_block = False
                    code_text = "\n".join(code_block_buffer)
                    if code_text.strip():
                        label = "【代码块" + (" - " + code_block_lang if code_block_lang else "") + "】\n"
                        elements.append({"type": "code_block", "text": label + code_text, "page_num": 1, "level": 0})
                    i += 1
                    continue
            if in_code_block:
                code_block_buffer.append(line)
                i += 1
                continue
            trimmed = line.strip()
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                elements.append({"type": "heading", "text": heading_text, "level": level, "page_num": 1})
                i += 1
                continue
            if "|" in line and i + 1 < len(lines) and re.match(r'^\|?[-:| ]+\|?$', lines[i + 1].strip()):
                table_lines = []
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                table_text = "\n".join(table_lines)
                elements.append({"type": "table", "text": "【表格】\n" + table_text, "page_num": 1, "level": 0})
                continue
            if re.match(r'^-{3,}$', trimmed) or re.match(r'^\*{3,}$', trimmed) or re.match(r'^_{3,}$', trimmed):
                i += 1
                continue
            if trimmed:
                if elements and elements[-1]["type"] == "text" and not _para_break:
                    elements[-1]["text"] += "\n" + line
                else:
                    elements.append({"type": "text", "text": line, "page_num": 1, "level": 0})
                _para_break = False
            else:
                _para_break = True
            i += 1
        return elements


class ParentChildChunkerSimple:
    def __init__(self, child_chunk_size: int = 150, child_chunk_overlap: int = 30):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
    def chunk(self, parents: List[Dict]) -> List[Dict]:
        children_out: List[Dict] = []
        for parent in parents:
            parent_text = parent.get("content", "")
            if not parent_text:
                continue
            parent_id = hashlib.md5(parent_text.encode("utf-8")).hexdigest()[:12]
            child_texts = self._recursive_split_text(parent_text)
            for ci, child_text in enumerate(child_texts):
                children_out.append({
                    "content": child_text,
                    "parent_id": parent_id,
                    "parent_content": parent_text,
                    "parent_section": parent.get("section", ""),
                    "section": parent.get("section", ""),
                    "page_range": parent.get("page_range", ""),
                    "content_types": parent.get("content_types", []),
                    "source": parent.get("source", "unknown"),
                    "chunk_id": f"{parent.get('chunk_id', 'doc')}_child_{ci}",
                    "chunk_index": ci,
                })
        return children_out
    def _recursive_split_text(self, text: str) -> List[str]:
        return self._split_with_seps(text, ["\n\n", "\n", "。", "，", " ", ""], 0)
    def _split_with_seps(self, text: str, seps: List[str], depth: int) -> List[str]:
        if len(text) <= self.child_chunk_size:
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
            if len(candidate) <= self.child_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.child_chunk_size:
                    chunks.extend(self._split_with_seps(part, seps, depth + 1))
                else:
                    current = part.strip()
        if current:
            chunks.append(current)
        merged = []
        for c in chunks:
            if merged and len(merged[-1]) < self.child_chunk_size * 0.5:
                merged[-1] = merged[-1] + "\n" + c
            else:
                merged.append(c)
        return merged
    def _split_by_size(self, text: str) -> List[str]:
        r, overlap = [], min(self.child_chunk_overlap, self.child_chunk_size // 2)
        start = 0
        while start < len(text):
            c = text[start:start + self.child_chunk_size]
            if c.strip():
                r.append(c)
            start += self.child_chunk_size - overlap
        return r

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
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文 2-gram 分词 + 数字/英文保持原样"""
        tokens = []
        text = text.lower()
        for segment in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9.]+', text):
            if re.match(r'^[\u4e00-\u9fff]+$', segment):
                # 中文：生成重叠 2-gram
                for i in range(len(segment) - 1):
                    tokens.append(segment[i:i+2])
                # 也保留单字
                for c in segment:
                    tokens.append(c)
            elif re.match(r'^[a-zA-Z0-9.]+$', segment):
                tokens.append(segment)
        return tokens

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
                 "chunk_id": self.docs[i].get("chunk_id", str(i)),
                 "metadata": self.docs[i].get("metadata", {}),
                 "search_type": "vector",
                 "parent_content": self.docs[i].get("parent_content", "")} for i in idxs]

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 关键词检索"""
        bm = self.bm25.search(query, top_k=top_k) if self.bm25 else []
        return [{
            "chunk_id": self.docs[i].get("chunk_id", str(i)),
            "content": self.contents[i],
            "score": round(float(s), 6),
            "metadata": self.docs[i].get("metadata", {}),
            "search_type": "bm25",
            "parent_content": self.docs[i].get("parent_content", ""),
        } for i, s in bm]

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
                 "metadata": self.docs[i].get("metadata", {}), "search_type": "hybrid",
                 "parent_content": self.docs[i].get("parent_content", "")}
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

def jdump(obj, path: Path):
    """保存 JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _check_hit(results: List[Dict], gt: str) -> tuple:
    """检查答案是否在检索结果中（子块内容 + 父块内容）"""
    for idx, r in enumerate(results):
        if gt in r.get("content", ""):
            return True, idx
        if gt in r.get("parent_content", ""):
            return True, idx
    return False, None



def hit_rate(results, gts, k):
    hits = 0
    for res, gt in zip(results, gts):
        gt_str = str(gt) if gt else ""
        if any(gt_str in r.get("content", "") or gt_str in r.get("parent_content", "") for r in res[:k]):
            hits += 1
    return hits / len(results) if results else 0.0


def mrr(results, gts):
    rrs = []
    for res, gt in zip(results, gts):
        gt_str = str(gt) if gt else ""
        for rk, r in enumerate(res, 1):
            if gt_str in r.get("content", "") or gt_str in r.get("parent_content", ""):
                rrs.append(1.0 / rk)
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs) if rrs else 0.0


def precision_at_k(results, gts, k):
    """Precision@k：top-k 中与标准答案相关的文档占比。

    本项目 QA 是"单答案"（answer_gt 为原文片段），所以相关文档数最多 = 命中
    该片段的 chunk 数（父子分块下同一父块的多个子块都可能含答案）。
    P@k 在此场景衡量的是"噪音稀释程度"——top-k 里无关内容占比越高，
    说明 Rerank/Embedding 区分度不足，噪声越可能带偏 LLM。
    """
    precisions = []
    for res, gt in zip(results, gts):
        gt_str = str(gt) if gt else ""
        relevant = sum(
            1 for r in res[:k]
            if gt_str in r.get("content", "")
            or gt_str in r.get("parent_content", "")
        )
        precisions.append(relevant / k if res else 0.0)
    return sum(precisions) / len(precisions) if precisions else 0.0


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
        # ── 02-差旅与报销管理办法-2025版 ──
        ("一线城市出差住宿费每晚上限是多少？", "550 元"),
        ("员工出差期间餐补每天多少钱？", "130 元"),
        ("出差申请超预算多少须重新说明？", "20%"),
        ("公司支持电子发票通过什么方式提交？", "线上提交"),
        ("出差结束后应在多长时间内完成报销？", "60 天"),
        ("报销审批额度上限是多少？", "8000 元"),
        ("报销款项几个工作日内发放？", "3 个工作日"),
        ("市内交通补贴单日上限是多少？", "120 元"),
        ("出差乘坐经济舱的飞行距离条件是什么？", "800 公里"),
        ("二线城市出差住宿标准上限是多少？", "420 元"),
        ("其他城市出差住宿标准上限是多少？", "320 元"),
        ("部门总监及以上可乘坐什么等级座位？", "高铁一等座"),
        ("出差伙食补助超过半天不足一天按多少钱计发？", "65 元"),
        ("出差申请单新增了哪项填写内容？", "预算金额"),
        ("员工乘坐高铁优先选择什么座位？", "二等座"),
        ("出差途中因不可抗力滞留费用如何报销？", "按本办法报销"),

        # ── 03-考勤与请假管理制度 ──
        ("员工月补卡不超过几次？", "3 次"),
        ("单次事假超过几天须部门负责人审批？", "3 天"),
        ("连续请假超几个工作日应报人事部备案？", "10 个工作日"),
        ("当月迟到超过几次起扣全勤奖？", "3 次"),
        ("婚假多少天？", "10 天"),
        ("丧假多少天？", "3 天"),
        ("事假是带薪假还是无薪假？", "无薪假"),
        ("累计工龄满1年不满10年每年几天年假？", "5 天"),
        ("累计工龄满10年不满20年每年几天年假？", "10 天"),
        ("累计工龄满20年以上每年几天年假？", "15 天"),
        ("月累计病假超3天超出部分按基本工资多少计发？", "70%"),
        ("病假证明须提供什么级别医院？", "二级及以上医院"),
        ("法定节假日加班按多少倍工资支付？", "3 倍工资"),
        ("休息日加班优先安排什么？", "调休"),
        ("工作日加班调休比例是多少？", "1:1"),
        ("年假满20年以上每年多少天？", "15 天"),
        ("标准工作时间是几点到几点？", "09:00–18:00"),
        ("请假超过10个工作日应报哪里备案？", "人事部"),
        ("员工请假最小单位是什么？", "半天"),
        ("婚假应在结婚后多长时间内休完？", "一年内"),
        ("试用期员工有带薪年假吗？", "无"),

        # ── 04-远程办公管理规定 ──
        ("远程办公每周最多申请几天？", "2 天"),
        ("申请远程办公须转正满几个月？", "3 个月"),
        ("公司为远程办公员工配发什么设备？", "笔记本电脑"),

        # ── 05-员工入职与转正管理办法 ──
        ("试用期最长不超过几个月？", "6 个月"),
        ("试用期一般是多久？", "3 个月"),
        ("试用期工资为转正后工资的多少？", "90%"),
        ("新员工入职引导安排在什么时间？", "入职当日"),

        # ── 06-IT设备与办公用品管理规定 ──
        ("笔记本电脑多久更换一次？", "4 年"),
        ("常规办公用品每月什么日期集中领用？", "5 日"),
        ("单价多少以上的办公用品须审批？", "200 元"),
        ("IT设备报修后多长时间内响应？", "2 个工作日"),

        # ── 07-薪酬与绩效管理制度 ──
        ("绩效工资占月薪比例是多少？", "20%"),
        ("年度调薪窗口在什么时候？", "每年 4 月"),
        ("月薪发放日期是每月几号？", "15 日"),

        # ── 08-员工福利与商业保险管理办法 ──
        ("公司额外提供什么商业保险？", "补充商业医疗保险"),
        ("补充商业医疗保险保额是多少？", "50 万元"),
        ("公司每年提供几次免费体检？", "1 次"),

        # ── 09-生育与陪产假管理规定 ──
        ("女员工产假多少天？", "158 天"),
        ("陪产假有几天？", "15 天"),
        ("育儿假每年多少天？", "5 天"),
        ("孕期每月可享几天带薪产检假？", "1 天"),
        ("陪产假需在配偶分娩后多长时间内休完？", "3 个月"),

        # ── 10-员工持股与股权激励计划 ──
        ("激励对象司龄要求多久？", "2 年"),
        ("股权锁定期是多久？", "1 年"),
        ("未归属部分离职后怎么处理？", "作废"),

        # ── 长答案 QA（用于验证精排效果） ──
        ("公司的带薪年假是怎么规定的？",
         "第十八条 带薪年假按员工累计工龄确定：\n- 累计工龄满 1 年不满 10 年：每年 5 天\n- 累计工龄满 10 年不满 20 年：每年 10 天\n- 累计工龄满 20 年以上：每年 15 天\n\n第十九条 累计工龄以员工依法可证明的工作年限为依据。员工入职时应提供社保缴纳记录、离职证明、劳动合同或其他可证明工作年限的材料；无法提供有效证明的，公司可按本公司可核实工龄计算。"),
        ("哪些情况下公司不批准远程办公申请？",
         "第九条 以下情形原则上不批准远程办公：\n- 岗位需现场接待客户、管理前台、收发快递、维护办公设施或处理纸质档案；\n- 当前参与重大上线、紧急故障处理、集中培训、盘点、审计或其他需现场协作的事项；\n- 员工近期考勤异常、补卡频繁、响应不及时或交付质量不稳定；\n- 员工远程办公环境无法满足网络、安静办公、保密和设备安全要求；\n- 部门负责人认为远程安排将明显影响团队协作或业务连续性。"),
        ("员工住院后如何申请商业保险理赔？",
         "第九条 员工发生住院、门诊特殊病、重大疾病或其他可理赔事项时，应优先按基本医保流程结算，再根据保险公司要求提交发票、费用清单、诊断证明、出院小结、医保结算单、银行卡信息和其他材料。"),
        ("试用期目标包括哪些方面？",
         "第二十一条 用人部门应在试用期开始时与员工确认试用期目标。目标可包括工作业绩、交付质量、学习进度、协作表现、客户反馈、合规意识、沟通响应和岗位技能掌握情况。目标应尽量具体、可观察、可评估。"),
        ("工作日加班如何调休？",
         "第三十七条 加班须提前申请并经直属上级批准。工作日加班可申请调休（1:1），调休须在 3 个月内使用完毕。"),
    ]
    return qa_pairs


# ─── 主流程 ─────────────────────────────────────────────

def evaluate():
    print("=" * 65, flush=True)
    print("  test-corpus 透明评测（MarkdownChunker + ParentChildChunker）", flush=True)
    print("=" * 65, flush=True)

    doc_dir = project_root / "test-corpus"
    out_dir = project_root / "data" / "testcorpus_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══ Step 0: 加载文档 ═══
    print(f"\n[Step 0] 加载文档...", flush=True)
    md_files = sorted([f for f in doc_dir.iterdir() if f.suffix == '.md' and f.name != 'README.md'])
    print(f"  共 {len(md_files)} 篇文档", flush=True)
    docs = []
    for f in md_files:
        with open(f) as fh:
            text = fh.read()
        docs.append({"filename": f.name, "content": text, "title": f.stem})
        print(f"    {f.name}: {len(text)} 字符", flush=True)

    # ═══ Step 1: MarkdownChunker（按标题语义分块） ═══
    print(f"\n[Step 1] MarkdownChunker: 按标题层级分 semantic block...", flush=True)
    md_chunker = MarkdownChunker(chunk_size=1000)
    all_parents = []
    for d in docs:
        parents = md_chunker.chunk_text(d["content"], {"source": d["filename"]})
        for p in parents:
            p["doc_file"] = d["filename"]
        all_parents.extend(parents)
    print(f"  ✅ {len(all_parents)} 个父块", flush=True)

    # 保存每个父块
    parents_data = []
    for i, p in enumerate(all_parents):
        parents_data.append({
            "index": i,
            "chunk_id": p.get("chunk_id", ""),
            "doc_file": p.get("doc_file", ""),
            "section": p.get("section", ""),
            "len": len(p.get("content", "")),
            "content_types": p.get("content_types", []),
            "content": p.get("content", ""),
        })
    jdump(parents_data, out_dir / "step1_parent_chunks.json")

    # ═══ Step 2: ParentChildChunker ═══
    print(f"\n[Step 2] ParentChildChunker: 子块切分 (child_size=300, overlap=30)...", flush=True)
    pc_chunker = ParentChildChunkerSimple(child_chunk_size=150, child_chunk_overlap=30)
    all_children = pc_chunker.chunk(all_parents)
    print(f"  ✅ {len(all_children)} 个子块", flush=True)

    # 统计每篇文档的子块分布
    from collections import Counter
    doc_chunk_counts = Counter(c.get("source", "") for c in all_children)
    for doc, cnt in sorted(doc_chunk_counts.items()):
        print(f"    {doc}: {cnt} 子块", flush=True)

    child_lengths = [len(c.get("content", "")) for c in all_children]
    print(f"  子块长度: min={min(child_lengths)} 平均={sum(child_lengths)//len(child_lengths):.0f} max={max(child_lengths)}", flush=True)

    # 保存子块（完整内容）
    children_data = []
    for i, c in enumerate(all_children):
        children_data.append({
            "index": i,
            "chunk_id": c.get("chunk_id"),
            "parent_id": c.get("parent_id"),
            "source": c.get("source"),
            "section": c.get("section"),
            "len": len(c.get("content", "")),
            "content": c.get("content"),
            "parent_content_len": len(c.get("parent_content", "")),
        })
    jdump(children_data, out_dir / "step2_child_chunks.json")

    # 用表格打印几个典型子块
    print(f"\n  典型子块示例（前 5 个）：", flush=True)
    for i, c in enumerate(all_children[:5]):
        print(f"    ┌─ index={i} chunk_id={c['chunk_id']} parent_id={c['parent_id'][:8]}...", flush=True)
        print(f"    ├─ source={c.get('source','')} section={c.get('section','')}", flush=True)
        print(f"    ├─ len={len(c.get('content',''))}", flush=True)
        print(f"    └─content: {c.get('content','')[:120]}...", flush=True)
        print(flush=True)

    # ═══ Step 3: QA 对 ═══
    print(f"\n[Step 3] 准备测试问题...", flush=True)
    qa_pairs = build_test_qa(doc_dir)
    queries = [q for q, a in qa_pairs]
    gt = [a for q, a in qa_pairs]
    print(f"  ✅ {len(qa_pairs)} 条测试问题", flush=True)

    # 验证答案在子块中的保留率
    retained = 0
    for q, a in qa_pairs:
        if any(a in c.get("content", "") for c in all_children):
            retained += 1
    print(f"  答案保留率: {retained}/{len(qa_pairs)} = {retained/len(qa_pairs):.1%}", flush=True)

    # 保存 QA 对
    qa_data = []
    for i, (q, a) in enumerate(qa_pairs):
        match_idx = None
        for ci, c in enumerate(all_children):
            if a in c.get("content", ""):
                match_idx = ci
                break
        qa_data.append({
            "index": i, "query": q, "answer_gt": a,
            "retained": any(a in c.get("content", "") for c in all_children),
            "located_in_child_chunk": match_idx,
        })
    jdump(qa_data, out_dir / "step3_qa_pairs.json")

    # ═══ Step 4: 建索引 + 检索 ═══
    print(f"\n[Step 4] 建索引 + 执行三种检索策略...", flush=True)
    print(f"  索引对象: 子块（ParentChildChunker 产出），共 {len(all_children)} 条", flush=True)
    embedder = EmbeddingEngine()
    retriever = LightRetriever(all_children, embedder)
    retriever.build_index()

    # 存储检索过程的详细信息
    per_query_bm25 = []
    per_query_vector = []
    per_query_hybrid = []
    per_query_reranked = []
    rv, rb, rh, rr = [], [], [], []
    lv, lb, lh, lr = [], [], [], []

    reranker = LightReranker()

    for qi, q in enumerate(queries):
        if (qi + 1) % 5 == 0:
            print(f"  Query {qi+1}/{len(queries)}", flush=True)

        # A) 纯向量检索 top-10
        t0 = time.perf_counter()
        vec_res = retriever.vector_search(q, 10)
        lv.append(time.perf_counter() - t0)
        rv.append(vec_res)

        # B) BM25 纯关键词检索 top-10
        t0 = time.perf_counter()
        bm25_res = retriever.bm25_search(q, 10)
        lb.append(time.perf_counter() - t0)
        rb.append(bm25_res)

        # C) 混合检索：BM25 + 向量 + RRF 融合 top-10
        rrf_k = 60
        t0 = time.perf_counter()
        vec_wide = retriever.vector_search(q, 20)
        bm25_wide = retriever.bm25_search(q, 20)
        rrf_scores = {}
        for rk, r in enumerate(vec_wide):
            did = r["chunk_id"]
            rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k + rk + 1)
        for rk, r in enumerate(bm25_wide):
            did = r["chunk_id"]
            rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (rrf_k + rk + 1)
        hyb_tmp = [r for r in vec_wide if r["chunk_id"] in rrf_scores]
        for r in hyb_tmp:
            r["score"] = rrf_scores.get(r["chunk_id"], 0)
            r["search_type"] = "hybrid"
        hyb_tmp.sort(key=lambda x: x["score"], reverse=True)
        hyb_res = hyb_tmp[:10]
        lh.append(time.perf_counter() - t0)
        rh.append(hyb_res)

        # D) Cross-Encoder 精排 top-5
        t0 = time.perf_counter()
        reranked = reranker.rerank(q, hyb_res, 5)
        lr.append(time.perf_counter() - t0)
        rr.append(reranked)

        # 记录详细的逐条过程
        vec_hit, vec_hit_pos = _check_hit(vec_res, gt[qi])
        bm25_hit, bm25_hit_pos = _check_hit(bm25_res, gt[qi])
        hyb_hit, hyb_hit_pos = _check_hit(hyb_res, gt[qi])
        rer_hit, rer_hit_pos = _check_hit(reranked, gt[qi])

        per_query_bm25.append({
            "query_index": qi, "query": q, "answer_gt": gt[qi],
            "latency_ms": round(lb[-1] * 1000, 1),
            "hit": bm25_hit, "hit_position": bm25_hit_pos,
            "top_k": [
                {"rank": rk+1, "chunk_id": r.get("chunk_id",""), "score": round(r.get("score",0),4),
                 "source": r.get("metadata",{}).get("source",""),
                 "content_preview": r.get("content","")[:150],
                 "parent_content_preview": r.get("parent_content","")[:150]}
                for rk, r in enumerate(bm25_res)
            ],
        })

        per_query_vector.append({
            "query_index": qi, "query": q, "answer_gt": gt[qi],
            "latency_ms": round(lv[-1] * 1000, 1),
            "hit": vec_hit, "hit_position": vec_hit_pos,
            "top_k": [
                {"rank": rk+1, "chunk_id": r.get("chunk_id",""), "score": round(r.get("score",0),4),
                 "source": r.get("metadata",{}).get("source",""),
                 "content_preview": r.get("content","")[:150],
                 "parent_content_preview": r.get("parent_content","")[:150]}
                for rk, r in enumerate(vec_res)
            ],
        })

        per_query_hybrid.append({
            "query_index": qi, "query": q, "answer_gt": gt[qi],
            "latency_ms": round(lh[-1] * 1000, 1),
            "hit": hyb_hit, "hit_position": hyb_hit_pos,
            "bm25_results": [
                {"rank": rk+1, "chunk_id": r.get("chunk_id",""), "score": round(r.get("score",0),4),
                 "content_preview": r.get("content","")[:100]}
                for rk, r in enumerate(vec_wide)
            ],
            "bm25_results": [
                {"rank": rk+1, "chunk_id": r.get("chunk_id",""), "score": round(r.get("score",0),4),
                 "content_preview": r.get("content","")[:100],
                 "parent_content_preview": r.get("parent_content","")[:150]}
                for rk, r in enumerate(bm25_wide)
            ],
            "rrf_fused": [
                {"rank": rk+1, "chunk_id": r.get("chunk_id",""), "score": round(r.get("score",0),4),
                 "vector_rank": r.get("vector_rank"), "bm25_rank": r.get("bm25_rank"),
                 "content_preview": r.get("content","")[:150],
                 "parent_content_preview": r.get("parent_content","")[:150]}
                for rk, r in enumerate(hyb_res)
            ],
        })

        per_query_reranked.append({
            "query_index": qi, "query": q, "answer_gt": gt[qi],
            "latency_ms": round(lr[-1] * 1000, 1),
            "hit": rer_hit, "hit_position": rer_hit_pos,
            "reranked": [
                {"rank": rk+1, "chunk_id": r.get("chunk_id",""),
                 "rerank_score": round(r.get("rerank_score",0),4),
                 "original_hybrid_score": round(r.get("score",0),4),
                 "content_preview": r.get("content","")[:200],
                 "parent_content_preview": r.get("parent_content","")[:200]}
                for rk, r in enumerate(reranked)
            ],
        })

    jdump(per_query_bm25, out_dir / "step4_bm25_search.json")
    jdump(per_query_vector, out_dir / "step4_vector_search.json")
    jdump(per_query_hybrid, out_dir / "step4_hybrid_search.json")
    jdump(per_query_reranked, out_dir / "step4_reranked_results.json")
    print(f"  检索过程已保存到 {out_dir}/step4_*", flush=True)

    # ═══ Step 5: 计算指标 ═══
    print(f"\n{'='*65}", flush=True)
    print(f"  评测结果汇总", flush=True)
    print(f"{'='*65}\n", flush=True)

    strategies = {
        "纯向量检索": rv,
        "混合检索 (BM25+向量+RRF)": rh,
        "混合+RRF+Cross-Encoder精排": rr,
    }

    hdr = (
        f"{'策略':<28} | {'HR@1':<7} | {'HR@3':<7} | {'HR@5':<7} | {'HR@10':<8} | "
        f"{'P@1':<7} | {'P@3':<7} | {'MRR':<7} | {'P50(ms)'}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    report = {}
    for sname, res in strategies.items():
        h1 = hit_rate(res, gt, 1)
        h3 = hit_rate(res, gt, 3)
        h5 = hit_rate(res, gt, 5)
        h10 = hit_rate(res, gt, 10)
        m = mrr(res, gt)
        if sname == "纯向量检索":
            p50 = round(float(np.median(lv) * 1000), 1)
        elif "精排" in sname:
            p50 = round(float(np.median(lr) * 1000), 1)
        else:
            p50 = round(float(np.median(lh) * 1000), 1)
        p1 = precision_at_k(res, gt, 1)
        p3 = precision_at_k(res, gt, 3)
        report[sname] = {
            "HR@1": h1, "HR@3": h3, "HR@5": h5, "HR@10": h10,
            "P@1": p1, "P@3": p3, "MRR": m,
        }
        print(
            f"{sname:<28} | {h1:<7.2%} | {h3:<7.2%} | {h5:<7.2%} | {h10:<8.2%} | "
            f"{p1:<7.2%} | {p3:<7.2%} | {m:<7.4f} | {p50}",
            flush=True,
        )

    keys = list(strategies.keys())
    b, p = report[keys[0]], report[keys[-1]]
    print(f"\n提升率（{keys[-1]} vs {keys[0]}）：", flush=True)
    for m in ["HR@1", "HR@3", "HR@5", "P@1", "MRR"]:
        if b.get(m, 0) > 0:
            pct = (p[m] - b[m]) / b[m] * 100
            print(f"  {m}: +{pct:.1f}%", flush=True)

    print(f"\n  文档: {len(md_files)} 篇", flush=True)
    print(f"  父块: {len(all_parents)} 个 (MarkdownChunker, chunk_size=1000)", flush=True)
    print(f"  子块: {len(all_children)} 个 (ParentChildChunker, child_size=300, overlap=30)", flush=True)
    print(f"  Query: {len(queries)} 条", flush=True)
    print(f"  答案保留率: {retained}/{len(qa_pairs)} = {retained/len(qa_pairs):.1%}", flush=True)
    print(f"  嵌入模型: BAAI/bge-small-zh-v1.5", flush=True)
    print(f"  精排模型: BAAI/bge-reranker-base", flush=True)

    # 保存指标汇总
    report["_meta"] = {
        "doc_count": len(md_files),
        "parent_chunk_count": len(all_parents),
        "child_chunk_count": len(all_children),
        "query_count": len(queries),
        "gt_retention_rate": retained / len(qa_pairs) if qa_pairs else 0,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "reranker_model": "BAAI/bge-reranker-base",
        "chunker": "MarkdownChunker (semantic blocks, chunk_size=1000) -> ParentChildChunkerSimple (child_size=300, overlap=30)",
    }
    jdump(report, out_dir / "step5_metrics_summary.json")

    # 保存逐条命中对比表
    detailed = []
    for i, (q, a) in enumerate(zip(queries, gt)):
        detailed.append({
            "index": i, "query": q, "answer_gt": a,
            "纯向量_命中": per_query_vector[i]["hit"],
            "BM25_命中": per_query_bm25[i]["hit"],
            "混合_命中": per_query_hybrid[i]["hit"],
            "精排_命中": per_query_reranked[i]["hit"],
        })
    jdump(detailed, out_dir / "step5_detailed_results.json")

    # 生成人类可读的 Markdown 报告
    lines = [
        f"# test-corpus 透明评测报告\n",
        f"\n",
        f"## 数据概览\n",
        f"| 项目 | 数值 |\n",
        f"|---|---|\n",
        f"| 文档数 | {len(md_files)} 篇 |\n",
        f"| 父块数（MarkdownChunker） | {len(all_parents)} 个 |\n",
        f"| 子块数（ParentChildChunker） | {len(all_children)} 个 |\n",
        f"| Query 数 | {len(queries)} 条 |\n",
        f"| 答案保留率 | {retained}/{len(qa_pairs)} = {retained/len(qa_pairs):.1%} |\n",
        f"\n",
        f"## 三种策略对比\n",
        f"| 策略 | HR@1 | HR@3 | HR@5 | HR@10 | P@1 | P@3 | MRR | P50(ms) |\n",
        f"|---|---|---|---|---|---|---|---|---|\n",
    ]
    for sname in strategies:
        m = report[sname]
        if sname == "纯向量检索":
            p50_ms = round(float(np.median(lv) * 1000), 1)
        elif "精排" in sname:
            p50_ms = round(float(np.median(lr) * 1000), 1)
        else:
            p50_ms = round(float(np.median(lh) * 1000), 1)
        lines.append(
            f"| {sname} | {m['HR@1']:.2%} | {m['HR@3']:.2%} | {m['HR@5']:.2%} | "
            f"{m['HR@10']:.2%} | {m['P@1']:.2%} | {m['P@3']:.2%} | "
            f"{m['MRR']:.4f} | {p50_ms}ms |\n"
        )
    lines.append(f"\n")
    lines.append(f"## 逐条命中情况（V=纯向量, H=混合+RRF, R=精排）\n")
    lines.append(f"| # | Query | V | H | R |\n")
    lines.append(f"|---|---|---|---|---|\n")
    for d in detailed:
        v = "✅" if d["纯向量_命中"] else "❌"
        b = "✅" if d["BM25_命中"] else "❌"
        h = "✅" if d["混合_命中"] else "❌"
        r = "✅" if d["精排_命中"] else "❌"
        lines.append(f"| {d['index']} | {d['query'][:40]}... | {v} | {h} | {r} |\n")

    with open(out_dir / "step5_report.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n所有过程数据已保存至: {out_dir}/", flush=True)
    print(f"  step1_parent_chunks.json       — 父块（完整内容）", flush=True)
    print(f"  step2_child_chunks.json        — 子块（完整内容，含 parent_id）", flush=True)
    print(f"  step3_qa_pairs.json            — QA 对", flush=True)
    print(f"  step4_vector_search.json       — 纯向量检索逐条过程", flush=True)
    print(f"  step4_hybrid_search.json       — 混合检索（向量+BM25+RRF）逐条过程", flush=True)
    print(f"  step4_reranked_results.json    — 精排后逐条结果", flush=True)
    print(f"  step5_metrics_summary.json     — 指标汇总", flush=True)
    print(f"  step5_detailed_results.json    — 逐条命中对比", flush=True)
    print(f"  step5_report.md                — 人类可读报告", flush=True)
    print("评测完成!", flush=True)


if __name__ == "__main__":
    evaluate()
