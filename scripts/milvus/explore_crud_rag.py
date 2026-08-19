# -*- coding: utf-8 -*-
"""Phase 2 任务0：CRUD-RAG 数据探查。

回答：
1. 80k 文档格式（一行一篇？）+ 单篇字符数分布
2. merged.json / split_merged.json 字段结构 + 四任务样本数分布
3. 每个测试样本能否定位到源文档（有无显式 doc id / 证据文本）
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent / "data/crud_rag/repo"
DOCS_DIR = REPO / "data/80000_docs"
MERGED = REPO / "data/crud/merged_unzip/merged.json"
SPLIT = REPO / "data/crud_split/split_merged.json"


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return int(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def dist(vals):
    if not vals:
        return "n=0"
    s = sorted(vals)
    return f"n={len(s)} min={s[0]} p50={pct(s,50)} p95={pct(s,95)} p99={pct(s,99)} max={s[-1]} mean={int(sum(s)/len(s))}"


# ----------------------------------------------------------------------
# 1. 80k 文档格式与长度分布
# ----------------------------------------------------------------------
print("=" * 70)
print("1. 80000_docs 文档格式与长度分布")
print("=" * 70)
doc_files = sorted(DOCS_DIR.glob("documents_dup_*"))
print(f"文档分片文件数: {len(doc_files)}")
# 抽样：每个分片取前若干行 + 随机
all_lines = []
for fp in doc_files:
    with open(fp, "r", encoding="utf-8") as f:
        all_lines.extend(f.readlines())
print(f"总行数（=文档数）: {len(all_lines)}")
print(f"首行前 80 字符: {all_lines[0][:80]!r}")
# 一行一篇：检查是否都以"正文："分隔
has_zhengwen = sum(1 for l in all_lines if "正文：" in l)
print(f'含"正文："的行数: {has_zhengwen} ({has_zhengwen/len(all_lines)*100:.1f}%)')

# 单篇字符数（去掉时间戳前缀后的正文长度）
lens = []
for l in all_lines:
    body = l.split("正文：", 1)[-1].strip() if "正文：" in l else l.strip()
    lens.append(len(body))
print(f"正文长度分布: {dist(lens)}")

# 抽样 200 篇用于 chunk 长度统计（写文件供 inspect_chunk_lengths 用）
random.seed(42)
sample_docs = random.sample(all_lines, min(200, len(all_lines)))
sample_file = REPO.parent / "crud_rag_sample200.txt"
with open(sample_file, "w", encoding="utf-8") as f:
    f.writelines(sample_docs)
print(f"已写抽样200篇到: {sample_file}")

# ----------------------------------------------------------------------
# 2. merged.json 字段结构与四任务样本数
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. merged.json 字段结构与四任务样本数")
print("=" * 70)
data = json.load(open(MERGED, encoding="utf-8"))
print(f"顶层 keys: {list(data.keys())}")
for k, v in data.items():
    print(f"\n--- {k} (类型={type(v).__name__}, 数量={len(v)}) ---")
    if isinstance(v, list) and v:
        s = v[0]
        if isinstance(s, dict):
            print(f"  字段: {list(s.keys())}")
            # 打印每个字段的值的前 120 字符
            for fk, fv in s.items():
                sv = json.dumps(fv, ensure_ascii=False) if not isinstance(fv, str) else fv
                print(f"  {fk}: {sv[:120]!r}")
    elif isinstance(v, dict):
        print(f"  dict keys: {list(v.keys())[:10]}")

# ----------------------------------------------------------------------
# 3. 四任务映射 + ground truth 可定位性
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. 四任务映射与 ground truth 可定位性")
print("=" * 70)
# CRUD-RAG 论文四任务：
# Create = 续写（continuing_writing）/ 摘要（event_summary）
# Read = 问答（questanswer_*）
# Update = 幻觉句子检测（hallu_modified）
# Delete = 冗余检测（？可能在 split_merged 或 questanswer_2docs/3docs 里）
task_map = {
    "event_summary": "Create(摘要)",
    "continuing_writing": "Create(续写)",
    "hallu_modified": "Update(幻觉检测)",
    "questanswer_1doc": "Read(单文档QA)",
    "questanswer_2docs": "Read(双文档QA)",
    "questanswer_3docs": "Read(三文档QA)",
}
for k, task in task_map.items():
    v = data.get(k, [])
    print(f"\n[{task}] {k}: {len(v)} 条")
    if v:
        s = v[0]
        # 检查是否有显式 doc id / 证据文本
        has_text = "text" in s
        has_id = "ID" in s
        text_len = len(s.get("text", "")) if has_text else 0
        print(f"  首条: has_text={has_text} has_ID={has_id} text长度={text_len}")
        print(f"  text 前 100 字符: {s.get('text','')[:100]!r}")

# ----------------------------------------------------------------------
# 4. split_merged.json 结构
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. split_merged.json 结构（论文切分）")
print("=" * 70)
with open(SPLIT, "r", encoding="utf-8") as f:
    head = f.read(500)
print(f"首 300 字符: {head[:300]!r}")
try:
    split = json.load(open(SPLIT, encoding="utf-8"))
    print(f"顶层类型: {type(split).__name__}")
    if isinstance(split, dict):
        print(f"keys: {list(split.keys())}")
        for k, v in split.items():
            print(f"  {k}: {type(v).__name__} 数量={len(v) if hasattr(v,'__len__') else '?'}")
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"    首条字段: {list(v[0].keys())}")
    elif isinstance(split, list):
        print(f"list 数量: {len(split)}, 首条字段: {list(split[0].keys()) if isinstance(split[0],dict) else '?'}")
except Exception as e:
    print(f"load 失败: {e}")
