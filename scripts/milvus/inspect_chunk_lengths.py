# -*- coding: utf-8 -*-
"""统计 text 与 parent_content 长度分布，为 Milvus VARCHAR max_length 选值提供依据。

对 test-corpus（10 篇制度文档，MD）跑完整链路：语义分块 → 父子分块。
输出 text / parent_content 的 min/p50/p95/p99/max 字符长度。
CRUD-RAG 语料未下载，其抽样统计留待 Phase 2 入库前补做（脚本预留入口）。
"""

import sys
from pathlib import Path
import statistics

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.markdown_chunker import MarkdownChunker
from app.rag.parent_child_chunker import ParentChildChunker


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def stats(vals):
    if not vals:
        return {"n": 0, "min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "min": s[0],
        "p50": int(percentile(s, 50)),
        "p95": int(percentile(s, 95)),
        "p99": int(percentile(s, 99)),
        "max": s[-1],
        "mean": int(statistics.mean(s)),
    }


def run_corpus(corpus_dir, label):
    print(f"\n{'='*60}\n{label}: {corpus_dir}\n{'='*60}")
    md_chunker = MarkdownChunker()
    pc_chunker = ParentChildChunker()

    files = sorted(Path(corpus_dir).glob("*.md"))
    # 跳过 README
    files = [f for f in files if f.name.lower() != "readme.md"]
    print(f"文档数: {len(files)}")

    text_lens = []
    parent_lens = []
    per_file_max_parent = []

    for fp in files:
        text = fp.read_text(encoding="utf-8")
        parents = md_chunker.chunk_md(str(fp), metadata={"source": fp.name})
        children = pc_chunker.chunk(parents)
        for ch in children:
            text_lens.append(len(ch.get("content", "")))
            parent_lens.append(len(ch.get("parent_content", "")))
        if parents:
            mp = max(len(p.get("content", "")) for p in parents)
            per_file_max_parent.append((fp.name, mp))

    print(f"\n子块总数: {len(text_lens)}")
    print("\n--- text（子块 content）长度分布 ---")
    st = stats(text_lens)
    for k, v in st.items():
        print(f"  {k}: {v}")

    print("\n--- parent_content（父块全文）长度分布 ---")
    sp = stats(parent_lens)
    for k, v in sp.items():
        print(f"  {k}: {v}")

    print("\n--- 每篇文档最大父块长度（top 长尾）---")
    for name, mp in sorted(per_file_max_parent, key=lambda x: x[1], reverse=True):
        print(f"  {mp:>6}  {name}")

    return st, sp


if __name__ == "__main__":
    st_text, st_parent = run_corpus("test-corpus", "test-corpus（制度文档）")

    print("\n" + "=" * 60)
    print("max_length 选值建议")
    print("=" * 60)
    text_rec = max(16384, int(st_text["p99"] * 4))
    # parent_content：若 max 接近或超过 32767 长尾，直接给 65535 平台上限
    if st_parent["max"] >= 20000 or st_parent["p99"] >= 16384:
        parent_rec = 65535
        parent_reason = "存在超长父块，直接给 Milvus VARCHAR 平台上限 65535"
    else:
        parent_rec = max(16384, int(st_parent["p99"] * 2))
        parent_reason = f"无明显长尾，取 p99({st_parent['p99']}) 的 2 倍向上取整"
    print(f"  text max_length          = {text_rec}  (p99={st_text['p99']} 的 4 倍向上取整，至少 16384)")
    print(f"  parent_content max_length= {parent_rec}  ({parent_reason})")
