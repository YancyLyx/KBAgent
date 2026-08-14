# -*- coding: utf-8 -*-
"""Markdown 文档分块模块

按标题层级（# / ## / ###）对 Markdown 进行语义化分块。
支持代码块、表格识别，图片直接按 alt 文本处理（不调视觉模型）。
"""

from typing import List, Dict, Any, Optional
import re
import yaml
from .chunk_utils import merge_elements_into_chunks


class MarkdownChunker:
    """Markdown 文档分块器

    按 # 标题层级分块，识别表格和代码块。
    chunk_size 默认 500，可在 rag_config.yaml 的 chunk_strategy 中修改。
    """

    # 匹配中英文标题
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        chunk_cfg = config.get("chunk_strategy", {})
        self.chunk_size = chunk_cfg.get("chunk_size", 500)

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def chunk_md(
        self,
        md_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """解析 Markdown 文件并按标题分块

        Args:
            md_path: .md 文件路径
            metadata: 附加元数据

        Returns:
            分块列表（格式与 PDFChunker 一致）
        """
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.chunk_text(text, metadata)

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """直接对 Markdown 文本字符串分块

        Args:
            text: Markdown 文本
            metadata: 附加元数据

        Returns:
            分块列表
        """
        if metadata is None:
            metadata = {}

        elements = self._parse_md(text)
        return merge_elements_into_chunks(
            elements,
            self.chunk_size,
            metadata=metadata,
            source_name=metadata.get("source", "md"),
        )

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    def _parse_md(self, text: str) -> List[Dict]:
        """将 Markdown 解析为结构化元素列表"""
        lines = text.split("\n")
        elements: List[Dict] = []

        i = 0
        in_code_block = False
        code_block_buffer: List[str] = []
        code_block_lang = ""
        _para_break = True  # 上次遇到的是段落断点

        while i < len(lines):
            line = lines[i]

            # ---- 代码块 ----
            if line.strip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_block_buffer = []
                    code_block_lang = line.strip()[3:].strip()
                    i += 1
                    continue
                else:
                    # 结束代码块
                    in_code_block = False
                    code_text = "\n".join(code_block_buffer)
                    if code_text.strip():
                        label = f"【代码块{ ' - ' + code_block_lang if code_block_lang else '' }】\n"
                        elements.append({
                            "type": "code_block",
                            "text": label + code_text,
                            "page_num": 1,
                            "level": 0,
                        })
                    i += 1
                    continue

            if in_code_block:
                code_block_buffer.append(line)
                i += 1
                continue

            trimmed = line.strip()

            # ---- 标题 ----
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                elements.append({
                    "type": "heading",
                    "text": heading_text,
                    "level": level,
                    "page_num": 1,
                })
                i += 1
                continue

            # ---- 表格 ----
            if "|" in line and i + 1 < len(lines) and re.match(r'^\|?[-:| ]+\|?$', lines[i + 1].strip()):
                table_lines = []
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                table_text = "\n".join(table_lines)
                elements.append({
                    "type": "table",
                    "text": "【表格】\n" + table_text,
                    "page_num": 1,
                    "level": 0,
                })
                continue

            # ---- 水平分割线 ----
            if re.match(r'^-{3,}$', trimmed) or re.match(r'^\*{3,}$', trimmed) or re.match(r'^_{3,}$', trimmed):
                # 分割线可以作为段落隔断，但不生成独立元素
                i += 1
                continue

            # ---- 正文（合并连续行到同一段落） ----
            if trimmed:
                if (elements
                    and elements[-1]["type"] == "text"
                    and not _para_break):
                    # 同一段落内的续行
                    elements[-1]["text"] += "\n" + line
                else:
                    elements.append({
                        "type": "text",
                        "text": line,
                        "page_num": 1,
                        "level": 0,
                    })
                _para_break = False
            else:
                # 空行 —— 段落结束标记
                _para_break = True

            i += 1

        return elements
