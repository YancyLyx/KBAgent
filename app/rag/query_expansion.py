# -*- coding: utf-8 -*-
"""查询增强（Advanced RAG 查询层优化）

- HyDE（Hypothetical Document Embeddings）：用 LLM 生成一段「假设答案」，
  用假设答案参与检索——答案文本通常与文档更相似，可提升语义召回。
- 多查询分解：把复杂问题拆成多个子查询，分别检索后合并去重，
  解决「一个问题跨多主题、单次检索只捞到一半」。
"""

import json
import re
from typing import List, Optional


async def generate_hypothetical_doc(
    llm_client,
    llm_model: str,
    query: str,
) -> Optional[str]:
    """HyDE：让 LLM 写一段假设答案（虚构但格式贴近文档），用于辅助检索"""
    prompt = (
        "请针对下面的问题，写一段假设性的知识库文档片段作为检索辅助。"
        "内容可以基于常识合理发挥，重点是与真实文档的表述风格一致，"
        "包含可能出现的术语、数字和条款描述。不要声称是官方内容。\n\n"
        f"问题：{query}\n\n假设文档片段："
    )
    try:
        resp = await llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


async def decompose_query(
    llm_client,
    llm_model: str,
    query: str,
) -> List[str]:
    """LLM 判断是否需要拆分，需要则拆成 2-3 个子查询；无需拆分返回 [原查询]。

    拆不拆是语义判断，交给 LLM 而非关键词规则：聚焦单主题的问题（"行权期
    是多久"）返回原查询；多主题问题（"差旅和餐补分别怎么规定"）拆成子查询。
    """
    prompt = (
        "判断下面的问题是否需要拆分成多个独立主题、分别检索知识库。\n"
        "规则：\n"
        "- 如果问题包含两个及以上需要分别检索的主题"
        "（如「差旅和餐补分别怎么规定」），拆成 2-3 个子问题，"
        "每个子问题聚焦一个主题，适合单独检索。\n"
        "- 如果问题本身聚焦单一主题（如「行权期是多久」），"
        "只输出原问题，不要拆分。\n"
        "只输出 JSON 数组，如 [\"子问题1\", \"子问题2\"]，不要其他文字。\n\n"
        f"问题：{query}"
    )
    try:
        resp = await llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return [query]
        data = json.loads(m.group(0))
        sub = [str(s).strip() for s in data if str(s).strip()]
        return sub[:3] if sub else [query]
    except Exception:
        return [query]
