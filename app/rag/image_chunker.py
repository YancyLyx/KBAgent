# -*- coding: utf-8 -*-
"""图片分块模块：用多模态 LLM（deepseek-v4-flash-vision-exp）描述图片内容。

定位：解析层文本化——把图片/图表/流程图转成结构化文本进现有 RAG pipeline，
检索架构零改动。诚实边界：描述由 VLM 生成，prompt 明确要求"不编造图中没有的内容"；
若 VLM 调用失败，返回可读的失败占位文本（不静默跳过，便于排查）。
"""

from typing import List, Dict, Any, Optional
import base64
import os
from pathlib import Path

from openai import OpenAI

from .chunk_utils import merge_elements_into_chunks

DEFAULT_VISION_MODEL = "deepseek-v4-flash-vision-exp"

VISION_PROMPT = (
    "请详细描述这张图片的内容，供知识库检索使用。要求：\n"
    "1. 先一句话概括图片主题；\n"
    "2. 若为图表（柱状图/折线图/饼图等），用【完整自然句子】描述每个数据点"
    "（如：2023年销售额为120万元，较上年增长40万元），再概述趋势；"
    "不要用'2022: 80'这类冒号列表；\n"
    "3. 若为流程图/结构图，用句子列出节点与关系；\n"
    "4. 若为表格截图，逐行转写成'列名是值'的句子；\n"
    "5. 若为纯文本/文档截图，转写文字内容。\n"
    "只描述图中确实存在的内容，不要编造或推测图中没有的信息。"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _vision_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


class ImageChunker:
    """图片分块器：VLM 描述 → 单个父块（短文本，父子分块后通常 1 个子块）"""

    def __init__(self, config_path: str = "config/rag_config.yaml"):
        # 配置仅用于与其它 chunker 保持构造签名一致；模型/密钥走 env
        self.config_path = config_path
        self.model = os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL)

    def _describe(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/{ext};base64,{b64}"
        client = _vision_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
            # vision-exp 是推理模型：token 先耗在 reasoning 上，
            # 800 会被截断导致 content 为空，放宽到 2000 让正文输出完整
            max_tokens=2000,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return f"[图片 {os.path.basename(image_path)} 未能生成描述]"
        return text

    def chunk_image(
        self,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if metadata is None:
            metadata = {}
        try:
            description = self._describe(image_path)
        except Exception as e:  # VLM 失败不静默：占位文本 + 可排查
            description = (
                f"[图片 {os.path.basename(image_path)} 描述失败：{type(e).__name__}: {e}]"
            )
        # 与其它 chunker 输出一致（父块格式，交给 ParentChildChunker）
        chunks = merge_elements_into_chunks(
            [{"type": "text", "text": description, "page_num": 1}],
            500,
            metadata=metadata,
            source_name=metadata.get("source", "img"),
        )
        # merge 会把 content_types 置为元素类型 text；图片描述块强制标注 image
        for c in chunks:
            c["content_types"] = ["image"]
            c["doc_type"] = c.get("doc_type", "image")
        return chunks
