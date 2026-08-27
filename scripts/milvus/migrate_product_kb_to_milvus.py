# -*- coding: utf-8 -*-
"""把产品知识库（test-corpus 10 篇制度文档）迁入 Milvus knowledge_base。

tag 映射取自 Chroma knowledge_base 的 ground truth（2026-08-27 读取）：
01/02->差旅报销 03->考勤请假 04->远程办公 05->入职转正 06->IT设备
07->薪酬绩效 08->福利保险 09->生育假期 10->股权激励
走产品同一管线（MarkdownChunker -> ParentChildChunker -> MilvusVectorStore），
chunk_id 由 source 派生，幂等可重跑。
"""
import os, sys
from dotenv import load_dotenv
sys_path = "/Users/yanxinluo/Documents/PycharmProjects/KBAgent"
load_dotenv(f"{sys_path}/.env")
sys.path.insert(0, sys_path)
os.environ["VECTOR_STORE_BACKEND"] = "milvus"

from app.rag.rag_pipeline import RAGPipeline

DOC_TAGS = {
    "01-差旅与报销管理办法-2023版.md": "差旅报销",
    "02-差旅与报销管理办法-2025版.md": "差旅报销",
    "03-考勤与请假管理制度.md": "考勤请假",
    "04-远程办公管理规定.md": "远程办公",
    "05-员工入职与转正管理办法.md": "入职转正",
    "06-IT设备与办公用品管理规定.md": "IT设备",
    "07-薪酬与绩效管理制度.md": "薪酬绩效",
    "08-员工福利与商业保险管理办法.md": "福利保险",
    "09-生育与陪产假管理规定.md": "生育假期",
    "10-员工持股与股权激励计划.md": "股权激励",
}

pipe = RAGPipeline(config_path=f"{sys_path}/config/rag_config.yaml", collection_name="knowledge_base")
print("backend:", getattr(pipe.vector_store, "backend", "?"), "| collection:", pipe.vector_store.collection_name)
for fname, tag in DOC_TAGS.items():
    path = f"{sys_path}/test-corpus/{fname}"
    print(f"\n>>> ingest {fname} (tag={tag})")
    pipe.add_file(path, metadata={"tag": tag, "doc_type": "md"})
print("\n迁移完成")
