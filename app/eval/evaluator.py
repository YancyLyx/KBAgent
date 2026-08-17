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
{contexts_section}

评分维度（1-5分）：
1. 相关性（Relevance）：回答是否直接针对用户问题？
2. 完整性（Completeness）：回答是否覆盖了问题需要的所有关键信息？
3. 有用性（Usefulness）：回答是否切实解决了用户的问题？
{faithfulness_section}

输出格式（仅 JSON）：
{{"relevance": 1-5, "completeness": 1-5, "usefulness": 1-5{faithfulness_field}, "explanation": "..."}}"""


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

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: Optional[str] = None,
    ) -> Optional[EvalScore]:
        """评测一条回答。

        contexts：本轮检索到的参考文档文本（可选）。提供时额外评估"忠实度"
        （答案是否基于参考文档、有无编造）——忠实度是幻觉最直接的量化信号。
        """
        if not query.strip() or not answer.strip():
            return None
        has_contexts = bool(contexts and contexts.strip())
        prompt = EVAL_PROMPT_TPL.format(
            query=query,
            answer=answer,
            contexts_section=(
                f"\n参考文档：\n{contexts[:2000]}" if has_contexts else ""
            ),
            faithfulness_section=(
                "\n4. 忠实度（Faithfulness）：回答是否严格基于参考文档，"
                "有没有编造参考文档中不存在的内容？"
                if has_contexts else ""
            ),
            faithfulness_field=', "faithfulness": 1-5' if has_contexts else "",
        )
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

    async def compare(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        contexts: Optional[str] = None,
    ) -> Optional[dict]:
        """Pairwise 比较：两个答案谁更好（相对判断比绝对打分更稳定）。

        Returns:
            {"winner": "a" | "b" | "tie", "reason": "..."}；失败返回 None。
        """
        if not query.strip() or not answer_a.strip() or not answer_b.strip():
            return None
        prompt = (
            "你是回答质量对比评测员。请比较两个回答哪个更好。\n\n"
            f"用户问题：{query}\n"
            + (f"参考文档：\n{contexts[:2000]}\n\n" if contexts and contexts.strip() else "\n")
            + f"回答 A：{answer_a}\n\n回答 B：{answer_b}\n\n"
            "比较维度：相关性（是否答对所问）、完整性（是否覆盖关键信息）、"
            "忠实度（是否基于参考文档、有无编造）、有用性（能否解决问题）。\n"
            '仅输出 JSON：{"winner": "a" 或 "b" 或 "tie", "reason": "简要理由（50 字内）"}'
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_compare(raw)
        except Exception as e:
            print(f"[Eval] 对比失败: {e}")
            return None

    @staticmethod
    def _parse_compare(raw: str) -> Optional[dict]:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if json_match:
            raw = json_match.group(1)
        try:
            data = json.loads(raw)
            winner = data.get("winner", "tie")
            if winner not in ("a", "b", "tie"):
                winner = "tie"
            return {"winner": winner, "reason": data.get("reason", "")}
        except (json.JSONDecodeError, ValueError, TypeError):
            # 宽松兜底：文本里出现 "回答 A/B 更好" 之类的判断
            if "回答 B 更好" in raw or "B 更好" in raw:
                return {"winner": "b", "reason": raw[:100]}
            if "回答 A 更好" in raw or "A 更好" in raw:
                return {"winner": "a", "reason": raw[:100]}
            return None

    @staticmethod
    def _parse(raw: str, query: str, answer: str) -> Optional[EvalScore]:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if json_match:
            raw = json_match.group(1)
        try:
            data = json.loads(raw)
            faithfulness = None
            if data.get("faithfulness") not in (None, ""):
                try:
                    faithfulness = int(data["faithfulness"])
                except (ValueError, TypeError):
                    # 忠实度字段异常不应让整条评分作废，回退 None
                    faithfulness = None
            return EvalScore(
                query=query,
                answer=answer[:200],
                relevance=int(data.get("relevance", 3)),
                completeness=int(data.get("completeness", 3)),
                usefulness=int(data.get("usefulness", 3)),
                faithfulness=faithfulness,
                explanation=data.get("explanation", ""),
                timestamp="",
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"[Eval] 解析失败: {raw[:100]}")
            return None
