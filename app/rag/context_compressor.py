# -*- coding: utf-8 -*-
"""上下文压缩（Advanced RAG 生成层优化）

检索/工具结果超预算时，用 LLM 压缩为保留关键信息的精简版，替代简单的硬截断。
简单截断会切断句子、丢失关键数字/条款；压缩保留与问题相关的事实与依据。
"""

from typing import Optional


async def compress_text(
    llm_client,
    llm_model: str,
    query: str,
    text: str,
    max_chars: int,
) -> Optional[str]:
    """把 text 压缩到 max_chars 以内，保留与 query 相关的关键信息。

    Returns:
        压缩后的文本；LLM 不可用或输出为空时返回 None（由调用方截断兜底）。
    """
    prompt = (
        "以下是从知识库检索到的内容，需要压缩后再交给回答者使用。\n"
        "要求：保留与问题相关的关键事实、数字、条款与来源信息；"
        "去掉重复与无关内容；不要编造原文没有的信息。\n\n"
        f"问题：{query}\n\n"
        f"原文：\n{text}\n\n"
        f"请压缩到 {max_chars} 字以内，直接输出压缩结果："
    )
    try:
        resp = await llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        compressed = (resp.choices[0].message.content or "").strip()
        if not compressed:
            return None
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars] + "...[压缩结果超长已截断]"
        return compressed
    except Exception:
        return None
