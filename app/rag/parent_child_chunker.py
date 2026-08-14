# -*- coding: utf-8 -*-
"""父子分块模块

将语义级分块（parent）再切分为更细粒度的子块（child），
实现「用子块做精确检索，返回父块做 LLM 上下文」的检索策略。
"""

from typing import List, Dict, Any, Optional
import hashlib
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentChildChunker:
    """父子分块器

    输入：PDFChunker / MarkdownChunker 产出的语义级 chunks（parent）
    输出：子块（child）列表，每个子块带 parent_id / parent_content 元数据
    """

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        pc_cfg = config.get("parent_child", {})
        self.enabled = pc_cfg.get("enabled", True)
        self.child_chunk_size = pc_cfg.get("child_chunk_size", 150)
        self.child_chunk_overlap = pc_cfg.get("child_chunk_overlap", 30)
        self.separators = pc_cfg.get(
            "separators", ["\n\n", "\n", "。", "，", " ", ""]
        )

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def chunk(
        self, parents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """对 parent chunks 进行父子分块

        Args:
            parents: PDFChunker / MarkdownChunker 输出的语义 block

        Returns:
            child chunks 列表，每个 chunk 的 metadata 中附带 parent 级信息
        """
        if not self.enabled:
            # 未启用父子分块时，直接返回 parents（作为 child 自身）
            for p in parents:
                p["parent_id"] = ""
                p["parent_content"] = ""
            return parents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        children_out: List[Dict[str, Any]] = []

        for parent in parents:
            parent_text = parent.get("content", "")
            if not parent_text:
                continue

            # 生成 parent_id（基于 content hash，相同内容始终映射到同一 parent）
            parent_id = hashlib.md5(parent_text.encode("utf-8")).hexdigest()[:12]

            # 对 parent 文本做子级切分
            child_texts = splitter.split_text(parent_text)

            for ci, child_text in enumerate(child_texts):
                child_chunk = {
                    "content": child_text,
                    "parent_id": parent_id,
                    "parent_content": parent_text,
                    "parent_section": parent.get("section", ""),

                    # 继承 parent 元数据
                    "section": parent.get("section", ""),
                    "page_range": parent.get("page_range", ""),
                    "content_types": parent.get("content_types", []),
                    "source": parent.get("source", "unknown"),

                    # child 自身标识
                    "chunk_id": f"{parent.get('chunk_id', 'doc')}_child_{ci}",
                    "chunk_index": ci,
                }
                # 继承 parent 的其余元数据（tag/title/doc_path 等）：
                # 否则子块入库后 tag 丢失，多知识库过滤失效
                for k, v in parent.items():
                    if k not in child_chunk and k != "content":
                        child_chunk[k] = v
                children_out.append(child_chunk)

        return children_out
