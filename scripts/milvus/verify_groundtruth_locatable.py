# -*- coding: utf-8 -*-
"""决定性验证：CRUD-RAG 测试样本的证据文本能否在 80000_docs 语料中定位。

这决定 HR@1/Recall 能不能算。
- event_summary.text / questanswer_*.newsN / continuing_writing.beginning
  应在 80k 语料中（同"时间戳，正文："格式）
- hallu_modified.newsBeginning 格式不同（新华社2016，ID=doc_XXXXXX），
  可能不在 80k 语料 → 需单独确认
"""
import json
from pathlib import Path

REPO = Path(__file__).parent.parent.parent / "data/crud_rag/repo"
DOCS_DIR = REPO / "data/80000_docs"
MERGED = REPO / "data/crud/merged_unzip/merged.json"

# 加载全部 80k 文档行，建两个索引：完整行集合 + 行首前缀(40字符)集合
print("加载 80k 语料...")
lines = []
for fp in sorted(DOCS_DIR.glob("documents_dup_*")):
    with open(fp, "r", encoding="utf-8") as f:
        for ln in f:
            lines.append(ln.rstrip("\n"))
print(f"语料行数: {len(lines)}")
full_set = set(lines)
prefix40_set = {ln[:40] for ln in lines}
prefix60_set = {ln[:60] for ln in lines}

data = json.load(open(MERGED, encoding="utf-8"))


def check_full(samples, field, label, n=20):
    hit = 0
    for s in samples[:n]:
        t = s.get(field, "")
        if not t:
            continue
        if t in full_set:
            hit += 1
    print(f"  [{label}] {field} 精确匹配 {hit}/{n}")
    return hit


def check_prefix(samples, field, label, n=20, plen=40):
    hit = 0
    for s in samples[:n]:
        t = s.get(field, "")
        if not t:
            continue
        if t[:plen] in (prefix40_set if plen == 40 else prefix60_set):
            hit += 1
    print(f"  [{label}] {field} 前{plen}字符匹配 {hit}/{n}")
    return hit


print("\n=== event_summary.text 是否在 80k 语料 ===")
check_full(data["event_summary"], "text", "event_summary", 20)
check_prefix(data["event_summary"], "text", "event_summary", 20, 40)

print("\n=== questanswer_1doc.news1 是否在 80k 语料 ===")
check_full(data["questanswer_1doc"], "news1", "qa1", 20)
check_prefix(data["questanswer_1doc"], "news1", "qa1", 20, 40)

print("\n=== questanswer_2docs.news1/news2 是否在 80k 语料 ===")
check_full(data["questanswer_2docs"], "news1", "qa2-news1", 20)
check_full(data["questanswer_2docs"], "news2", "qa2-news2", 20)

print("\n=== continuing_writing.beginning 是否在 80k 语料（前缀匹配）===")
check_prefix(data["continuing_writing"], "beginning", "cont", 20, 40)
check_prefix(data["continuing_writing"], "beginning", "cont", 20, 60)

print("\n=== hallu_modified.newsBeginning 是否在 80k 语料 ===")
check_full(data["hallu_modified"], "newsBeginning", "hallu", 20)
check_prefix(data["hallu_modified"], "newsBeginning", "hallu", 20, 40)
# hallu 的 ID 格式不同，看一下
print(f"  hallu ID 示例: {[s['ID'] for s in data['hallu_modified'][:5]]}")
print(f"  event_summary ID 示例: {[s['ID'] for s in data['event_summary'][:5]]}")

# 额外：event_summary.text 与 questanswer.news1 是否同源（都来自80k）
print("\n=== 抽查 event_summary[0].text 在语料中的实际行 ===")
t0 = data["event_summary"][0]["text"]
for ln in lines:
    ln.startswith(t0[:40])
matches = [ln for ln in lines if ln[:40] == t0[:40]]
print(f"  前40字符匹配到 {len(matches)} 行")
if matches:
    print(f"  语料行前 80: {matches[0][:80]!r}")
    print(f"  样本text前 80: {t0[:80]!r}")
    print(f"  完全相等: {matches[0] == t0}")
