# -*- coding: utf-8 -*-
"""PDF 文档分块模块

从 PDF 中提取文字、表格、图片标题，按章节语义分块。
图片部分不依赖视觉模型/OCR，直接从 PDF 中提取图片附近的标题/描述文本。
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import yaml
import fitz  # PyMuPDF
import pdfplumber
from .chunk_utils import merge_elements_into_chunks
from collections import Counter


class PDFChunker:
    """PDF 文档分块器

    支持按章节层级、表格、图片标题进行语义化分块。
    表格用 pdfplumber 精确提取并序列化为文本；
    图片提取其标题文字（利用 PDF 中已有的 "图X-X ..." 说明文本）。
    """

    # 标题文本模式
    HEADING_PATTERNS = [
        r'^第[一二三四五六七八九十百千]+[章节条]',
        r'^\d+\.\d+(?:\s|$)',
        r'^\d+\.\s',
        r'^Chapter\s+\d+',
        r'^Section\s+\d+',
    ]

    # 图片 / 表格标题模式
    CAPTION_PATTERNS = {
        "image": [r'^图\s*\d+[\.\-]?\s*\d*', r'^Figure\s+\d+'],
        "table": [r'^表\s*\d+[\.\-]?\s*\d*', r'^Table\s+\d+'],
    }

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        chunk_cfg = config.get("chunk_strategy", {})
        self.chunk_size = chunk_cfg.get("chunk_size", 500)
        self.chunk_overlap = chunk_cfg.get("chunk_overlap", 100)

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def chunk_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """解析 PDF 并按章节语义分块

        Args:
            pdf_path: PDF 文件路径
            metadata: 附加元数据

        Returns:
            分块列表，每块含 content / chunk_id / chunk_index /
            section / page_range / content_types 等字段
        """
        if metadata is None:
            metadata = {}

        elements: List[Dict] = []

        with pdfplumber.open(pdf_path) as pdf_doc:
            fitz_doc = fitz.open(pdf_path)
            for page_num in range(len(fitz_doc)):
                page_elems = self._parse_page(
                    fitz_doc[page_num], page_num, pdf_doc.pages[page_num]
                )
                elements.extend(page_elems)
            fitz_doc.close()

        return merge_elements_into_chunks(elements, self.chunk_size, metadata=metadata, source_name=metadata.get("source", "pdf"))

    # ------------------------------------------------------------------
    # 单页解析
    # ------------------------------------------------------------------

    def _parse_page(
        self, page, page_num: int, plumber_page
    ) -> List[Dict]:
        """解析单个 PDF 页面，返回结构化元素列表"""
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b["type"] == 0]   # type 0 = text
        image_blocks = [b for b in blocks if b["type"] == 1]  # type 1 = image

        body_size = self._detect_body_font_size(text_blocks)

        # ---- 文本 / 标题 / 图片标题 ----
        text_items: List[Dict] = []
        for block in text_blocks:
            text = self._block_text(block)
            if not text:
                continue

            y0 = block["bbox"][1]
            is_heading, level = self._is_heading(block, body_size)

            if is_heading:
                text_items.append(self._make_elem("heading", text, page_num, y0, level=level))
            else:
                # 检查是否为图片/表格标题
                cap_type = self._detect_caption_type(text, image_blocks, block["bbox"])
                text_items.append(self._make_elem(cap_type, text, page_num, y0))

        # ---- 表格（用 pdfplumber 提取）----
        table_items: List[Dict] = []
        for table_obj in plumber_page.find_tables():
            bbox = table_obj.bbox
            rows = table_obj.extract()
            if not rows:
                continue
            # 清洗 None -> 空字符串
            clean_rows = [
                [str(c) if c is not None else "" for c in row]
                for row in rows
            ]
            serialized = "【表格】\n" + self._serialize_table(clean_rows)
            table_items.append(
                self._make_elem("table", serialized, page_num, bbox[1])
            )

        # ---- 按垂直位置排序合并 ----
        page_elems = text_items + table_items
        page_elems.sort(key=lambda e: e["_y0"])
        for e in page_elems:
            del e["_y0"]
        return page_elems

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _block_text(block: Dict) -> str:
        """拼接 PyMuPDF 文本块中的所有文本"""
        parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                parts.append(span.get("text", ""))
        return "".join(parts).strip()

    @staticmethod
    def _detect_body_font_size(text_blocks: List[Dict]) -> float:
        """检测正文默认字号（取众数）"""
        sizes = []
        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(round(span.get("size", 12), 1))
        if not sizes:
            return 12.0
        return Counter(sizes).most_common(1)[0][0]

    @staticmethod
    def _is_heading(block: Dict, body_size: float) -> Tuple[bool, int]:
        """判断文本块是否为标题，返回 (是/否, 等级)

        等级: 1=大标题（章级）, 2=节标题, 3=小标题
        """
        text = ""
        max_size = 0
        is_bold = False

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text += span.get("text", "")
                sz = span.get("size", 0)
                max_size = max(max_size, sz)
                if span.get("flags", 0) & 2:  # bold
                    is_bold = True

        text = text.strip()
        if not text or len(text) > 120:
            return False, 0

        # 1) 字体明显大于正文 → 标题
        ratio = max_size / body_size if body_size > 0 else 1
        if ratio > 1.3:
            return True, 1 if ratio > 1.8 else 2

        # 2) 模式匹配
        for pat in PDFChunker.HEADING_PATTERNS:
            if re.match(pat, text):
                return True, 2

        # 3) 加粗且较短的文本 → 子标题
        if is_bold and len(text) < 60:
            return True, 3

        return False, 0

    @staticmethod
    def _has_nearby_image(
        image_blocks: List[Dict], text_bbox: Tuple[float, ...]
    ) -> bool:
        """判断文本块附近（< 60pt）是否有图片"""
        tx0, ty0, tx1, ty1 = text_bbox
        for img in image_blocks:
            ix0, iy0, ix1, iy1 = img["bbox"]
            v_dist = min(abs(ty1 - iy0), abs(iy1 - ty0))
            h_overlap = min(tx1, ix1) - max(tx0, ix0)
            if v_dist < 60 and h_overlap > 0:
                return True
        return False

    def _detect_caption_type(
        self,
        text: str,
        image_blocks: List[Dict],
        text_bbox: Tuple[float, ...],
    ) -> str:
        """识别文本是否为图片/表格标题，返回元素类型"""
        # 检查图片标题
        for pat in self.CAPTION_PATTERNS["image"]:
            if re.match(pat, text):
                if self._has_nearby_image(image_blocks, text_bbox):
                    return "image_caption"
                return "text"
        # 检查表格标题
        for pat in self.CAPTION_PATTERNS["table"]:
            if re.match(pat, text):
                return "table_caption"
        return "text"

    @staticmethod
    def _make_elem(
        typ: str,
        text: str,
        page_num: int,
        y0: float,
        level: int = 0,
    ) -> Dict:
        return {
            "type": typ,
            "text": text,
            "page_num": page_num + 1,
            "level": level,
            "_y0": y0,
        }

    @staticmethod
    def _serialize_table(rows: List[List[str]]) -> str:
        """将表格数据序列化为 Markdown 风格文本"""
        if not rows:
            return ""
        lines = []
        lines.append(" | ".join(rows[0]))
        lines.append(" | ".join(["---"] * len(rows[0])))
        for row in rows[1:]:
            lines.append(" | ".join(row))
        return "\n".join(lines)

