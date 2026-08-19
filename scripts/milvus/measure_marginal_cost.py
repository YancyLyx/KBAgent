# -*- coding: utf-8 -*-
"""边际成本测量：导入 1500 文档到 crud_rag，测规模化下每 chunk 真实磁盘增量。"""
import sys, os, time
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from app.rag.chunker import Chunker
from app.rag.parent_child_chunker import ParentChildChunker
from app.rag.milvus_store import MilvusVectorStore

DOCS_DIR = project_root / "data/crud_rag/repo/data/80000_docs"
START = int(os.getenv("START_DOC", "24000"))
N = int(os.getenv("N_DOCS", "1500"))


def du_milvus():
    t = 0
    for r, _, fs in os.walk(project_root / "data/milvus"):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t


def strip_prefix(line):
    i = line.find("正文：")
    return line[i + 3:] if i >= 0 else line  # 3 = len("正文：")? 实际5; 修正下面


def main():
    # 读取全部，取切片
    docs = []
    for fp in sorted(DOCS_DIR.glob("documents_dup_*")):
        for ln in open(fp, encoding="utf-8"):
            ln = ln.rstrip("\n")
            if ln.strip():
                docs.append(ln)
    seg = docs[START:START + N]
    # 正确去前缀
    def sp(l):
        i = l.find("正文：")
        return l[i + len("正文："):] if i >= 0 else l
    seg = [sp(s) for s in seg]
    print(f"测量文档: {len(seg)}（doc {START}..{START+len(seg)}）")

    chunker = Chunker()
    pc = ParentChildChunker()
    chunks = []
    for di, doc in enumerate(seg):
        idx = START + di
        meta = {"source": f"crud_{idx:06d}", "title": f"crud_{idx:06d}", "tag": "crud_news", "doc_type": "txt"}
        for ch in pc.chunk(chunker.split_text(doc, metadata=meta)):
            chunks.append(ch)
    print(f"chunk 数: {len(chunks)}")

    before = du_milvus()
    print(f"导入前 data/milvus: {before/1024/1024:.1f} MB")
    store = MilvusVectorStore("crud_rag", "config/rag_config.yaml", partition="crud_news")
    t = time.time()
    for i in range(0, len(chunks), 500):
        store.add_documents(chunks[i:i+500], batch_size=len(chunks[i:i+500]))
    time.sleep(3)
    print(f"入库耗时: {time.time()-t:.1f}s")
    after = du_milvus()
    delta = after - before
    print(f"导入后 data/milvus: {after/1024/1024:.1f} MB")
    print(f"增量: {delta/1024/1024:.1f} MB")
    print(f"边际每 chunk: {delta/len(chunks)/1024:.1f} KB")
    full = delta / len(chunks) * 283110
    print(f"外推全量 283110 chunks: {full/1024/1024/1024:.2f} GB（growing 段）")
    print(f"建索引后 ×1.4: {full/1024/1024/1024*1.4:.1f} GB")


if __name__ == "__main__":
    main()
