# -*- coding: utf-8 -*-
"""LLM-as-Judge 评测器"""

import json
import os
import re
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .schemas import EvalScore


EVAL_PROMPT_TPL = """你是一个回答质量评测助手。请严格基于以下标准评分。

用户问题：{query}
AI回答：{answer}

评分维度（1-5分）：
1. 相关性（Relevance）：回答是否直接针对用户问题？
2. 完整性（Completeness）：回答是否覆盖了问题需要的所有关键信息？
3. 有用性（Usefulness）：回答是否切实解决了用户的问题？

输出格式（仅 JSON）：
{{"relevance": 1-5, "completeness": 1-5, "usefulness": 1-5, "explanation": "..."}}"""


class Evaluator:
    """LLM-as-Judge 评测器"""

    def __init__(self):
        load_dotenv()
        # 优先使用 EVAL_* 专用配置，回退到 DEEPSEEK 主配置
        api_key = os.getenv("EVAL_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("EVAL_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("EVAL_MODEL") or os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("EVAL_TIMEOUT", "30")),
            max_retries=int(os.getenv("EVAL_MAX_RETRIES", "2")),
        )

    async def evaluate(self, query: str, answer: str) -> Optional[EvalScore]:
        if not query.strip() or not answer.strip():
            return None
        prompt = EVAL_PROMPT_TPL.format(query=query, answer=answer)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            raw = response.choices[0].message.content or ""
            return self._parse(raw, query, answer)
        except Exception as e:
            print(f"[Eval] 评分失败: {e}")
            return None

    @staticmethod
    def _parse(raw: str, query: str, answer: str) -> Optional[EvalScore]:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if json_match:
            raw = json_match.group(1)
        try:
            data = json.loads(raw)
            return EvalScore(
                query=query,
                answer=answer[:200],
                relevance=int(data.get("relevance", 3)),
                completeness=int(data.get("completeness", 3)),
                usefulness=int(data.get("usefulness", 3)),
                explanation=data.get("explanation", ""),
                timestamp="",
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"[Eval] 解析失败: {raw[:100]}")
            return None
