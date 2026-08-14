#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test-corpus 制度文档入库脚本

走线上同款管线（MarkdownChunker 章节分块 + ParentChildChunker 父子分块 +
VectorStore 入库），并按主题打 tag，供多知识库路由（Skill 系统）演示。

用法：
  python scripts/ingest_test_corpus.py            # 增量入库
  python scripts/ingest_test_corpus.py --reset    # 先清空 knowledge_base 再入库
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.rag_pipeline import RAGPipeline
from app.rag.vector_store import VectorStore
from app.agent.skill_manager import SkillManager


# 文件名前缀 → 知识库分类（tag）
TAG_BY_PREFIX = {
    "01": "差旅报销",
    "02": "差旅报销",
    "03": "考勤请假",
    "04": "远程办公",
    "05": "入职转正",
    "06": "IT设备",
    "07": "薪酬绩效",
    "08": "福利保险",
    "09": "生育假期",
    "10": "股权激励",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="先清空 knowledge_base 再入库")
    args = parser.parse_args()

    corpus = project_root / "test-corpus"
    md_files = sorted(
        f for f in corpus.iterdir()
        if f.suffix == ".md" and f.name != "README.md"
    )
    if not md_files:
        print("test-corpus 目录没有可入库的 md 文档")
        return

    if args.reset:
        print("清空 knowledge_base 集合...")
        store = VectorStore()
        store.delete_collection()

    rag = RAGPipeline()
    for path in md_files:
        prefix = path.name[:2]
        tag = TAG_BY_PREFIX.get(prefix, "综合")
        rag.add_file(
            str(path),
            metadata={"tag": tag, "source": path.name},
        )
        print(f"  ✅ {path.name} -> tag={tag}")

    total_children = rag.vector_store.collection.count()
    print(f"\n入库完成，当前子块总数：{total_children}")

    # 同步 Skill 分类参考文件（每个 tag 生成 references/{tag}.md + 更新 skill.md）
    print("同步 Skill 分类参考...")
    sm = SkillManager(rag.vector_store)
    sm.sync_references()
    print("分类参考：", rag.vector_store.get_available_tags())


if __name__ == "__main__":
    main()
