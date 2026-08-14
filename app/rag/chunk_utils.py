# -*- coding: utf-8 -*-
"""分块工具函数

PDFChunker 和 MarkdownChunker 共用的元素 → 分块合并逻辑。
"""

from typing import List, Dict, Any, Optional


def merge_elements_into_chunks(
    elements: List[Dict],
    chunk_size: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    source_name: str = "doc",
) -> List[Dict[str, Any]]:
    """将结构化元素按章节标题边界合并为 chunks

    Args:
        elements: 元素列表，每个元素含 type / text / page_num / level
        chunk_size: 每块最大字符数
        metadata: 附加元数据（如 source）
        source_name: 文档来源名，用于生成 chunk_id

    Returns:
        分块列表，含 content / chunk_id / chunk_index /
        section / page_range / content_types
    """
    if metadata is None:
        metadata = {}
    source_label = metadata.get("source", source_name)

    chunks: List[Dict[str, Any]] = []

    # 章节层级栈
    section_stack: List[str] = []

    buf_text: List[str] = []
    buf_size = 0
    buf_pages: set = set()
    buf_types: set = set()
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
