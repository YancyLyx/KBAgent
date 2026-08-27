# -*- coding: utf-8 -*-
"""DOCX 文档分块模块

用 python-docx 抽取段落与表格，按标题样式（Heading/标题）做章节边界，
与 MarkdownChunker 走同一套 merge_elements_into_chunks 逻辑。
表格序列化为 Markdown 风格文本（表头 | 行），保证检索可命中列名与数值。
"""

from typing import List, Dict, Any, Optional
import re
import yaml

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from .chunk_utils import merge_elements_into_chunks

HEADING_LVL_RE = re.compile(r"(\d+)")


class DocxChunker:
    """DOCX 分块器：段落 + 表格 → 章节语义块"""

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        chunk_cfg = config.get("chunk_strategy", {})
        self.chunk_size = chunk_cfg.get("chunk_size", 500)

    @staticmethod
    def _serialize_table(table: Table) -> str:
        """表格 → Markdown 风格文本（列名=值 或 管道表格）"""
        rows = []
        for row in table.rows:
            cells = [c.text.strip().replace("|", "/") for c in row.cells]
            rows.append(cells)
        if not rows:
            return ""
        # 表头 + 数据行：管道表格
        lines = [" | ".join(rows[0])]
        lines.append(" | ".join(["---"] * len(rows[0])))
        for r in rows[1:]:
            lines.append(" | ".join(r))
        return "\n".join(lines)

    def chunk_docx(
        self,
        docx_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """解析 .docx 并按标题分块"""
        if metadata is None:
            metadata = {}
        doc = docx.Document(docx_path)
        elements: List[Dict[str, Any]] = []

        # 按文档顺序遍历段落与表格（document.paragraphs 会丢表格位置）
        body = doc.element.body
        for child in body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue
                style_name = (p.style.name or "") if p.style else ""
                if style_name.lower().startswith(("heading", "标题")):
                    m = HEADING_LVL_RE.search(style_name)
                    level = min(int(m.group(1)), 6) if m else 1
                    elements.append({
                        "type": "heading", "text": text,
                        "level": level, "page_num": 1,
                    })
                else:
                    elements.append({
                        "type": "text", "text": text, "page_num": 1,
                    })
            elif tag == "tbl":
                t = Table(child, doc)
                table_text = self._serialize_table(t)
                if table_text:
                    elements.append({
                        "type": "text", "text": table_text, "page_num": 1,
                    })

        return merge_elements_into_chunks(
            elements,
            self.chunk_size,
            metadata=metadata,
            source_name=metadata.get("source", "docx"),
        )
