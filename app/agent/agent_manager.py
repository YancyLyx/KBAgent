# -*- coding: utf-8 -*-
"""Agent 管理器模块

基于 ReAct（Reasoning + Acting）的 Agent 架构。

核心流程（两轮 LLM 调用）：
  第 1 轮：LLM 看到 Skill 概览 + 工具列表，自主决策是否/如何检索知识库
  第 2 轮：基于工具执行结果，组织最终回答

工具：
  - read_category_info(tag)      → 查看指定知识库的详细参考
  - search_knowledge_base(query, tag) → 检索知识库
"""

import json
import asyncio
import re
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path

from ..rag.rag_pipeline import RAGPipeline
from ..rag.vector_store import VectorStore
from ..memory.short_memory import ShortMemory
from ..memory.long_memory import LongMemory
from ..cache.query_cache import get_shared_cache
from .tool_router import ToolRouter
from .skill_manager import SkillManager
from ..memory.conversation_memory import ConversationMemory


class IntentType(Enum):
    """意图类型"""
    KNOWLEDGE_QUERY = "knowledge_query"  # 知识库查询
    GENERAL_CHAT = "general_chat"        # 普通对话


class AgentManager:
    """Agent 管理器

    基于 ReAct 模式的 Agent 架构：
    - 环境感知：Skill 概览注入 system prompt，LLM 了解当前可用的知识库
    - 自主决策：LLM 决定是否调用工具、调用哪个工具
    - 渐进披露：先 read_category_info 了解详情，再 search_knowledge_base 检索
    """

    SYSTEM_PROMPT_TPL = """你是一个企业级智能知识库问答助手。请用中文回答，保持简洁专业。

{skill_intro}

工作流程：
1. 如果用户问题需要检索知识库，请使用 read_category_info 了解各知识库内容范围
2. 明确知识库后使用 search_knowledge_base 检索相关文档
3. 基于检索结果组织回答
4. 如果用户问题不需要检索知识库，直接回答

回答原则（兜底边界）：
{fallback_rule}
- 严格基于知识库检索结果回答；引用具体数据/条款时尽量标注来源章节。
- 如果检索结果提示未找到相关内容，明确告知用户未找到，并引导换关键词或补充细节，不要编造答案。
- 通用常识问题（不属于任何知识库分类）可以基于通用知识回答，但需标注「以下为通用知识，并非企业知识库内容」。"""

    def __init__(
        self,
        user_id: Optional[str] = None,
        enable_memory: bool = True,
        enable_rag: bool = True,
        enable_tools: bool = True,
        enable_eval: bool = False,
    ):
        self.user_id = user_id
        self.enable_memory = enable_memory
        self.enable_rag = enable_rag
        self.enable_tools = enable_tools

        # 语义缓存（进程级共享：不同会话复用同一份，避免每个实例一个空缓存）
        self.query_cache = get_shared_cache()

        # 初始化各模块
        self.rag_pipeline = RAGPipeline() if enable_rag else None
        self.vector_store = VectorStore() if enable_rag else None
        self.skill_manager = SkillManager(self.vector_store) if enable_rag else None

        self.tool_router = ToolRouter() if enable_tools else None
        if self.tool_router and self.skill_manager:
            self.tool_router.skill_manager = self.skill_manager
        # 复用已有 RAGPipeline，避免工具检索时重复初始化 embedding/Chroma
        if self.tool_router and self.rag_pipeline:
            self.tool_router.skill_pipeline = self.rag_pipeline

        self.short_memory = ShortMemory() if enable_memory else None
        self.long_memory = LongMemory() if enable_memory else None
        self.enable_eval = enable_eval
        # 会话累计轮数（区别于 ShortMemory.get_turn_count：后者是窗口内剩余轮数，
        # 窗口填满后恒为 max_history_length，不能作为摘要触发依据）
        self._session_turn_count = 0
        # Agent 循环与上下文预算（Loop Engineering）
        # 轮数上限是"预算保护"不是"路径规划"；工具结果上限按上下文预算动态算
        self.loop_config = self._load_loop_config()
        _loop_cfg = self.loop_config.get("loop", {})
        _budget_cfg = self.loop_config.get("budget", {})
        self.max_loop_rounds = int(_loop_cfg.get("max_rounds", 5))
        self.no_progress_threshold = int(_loop_cfg.get("no_progress_threshold", 2))
        self.tool_result_max_chars = int(_budget_cfg.get("tool_result_max_chars", 3000))
        self.tool_result_min_chars = int(_budget_cfg.get("tool_result_min_chars", 500))
        self.summary_max_chars = int(_budget_cfg.get("summary_max_chars", 300))
        self.recall_max_chars = int(_budget_cfg.get("recall_max_chars", 200))
        self.budget_config = dict(_budget_cfg)
        self.llm_tokens = {
            "react_loop": int(self.loop_config.get("llm_tokens", {}).get("react_loop", 1000)),
            "summary": int(self.loop_config.get("llm_tokens", {}).get("summary", 400)),
            "fallback": int(self.loop_config.get("llm_tokens", {}).get("fallback", 500)),
            "self_rag_judge": int(
                self.loop_config.get("llm_tokens", {}).get("self_rag_judge", 150)
            ),
            "recall_rewrite": int(
                self.loop_config.get("llm_tokens", {}).get("recall_rewrite", 100)
            ),
        }
        self.self_rag_enabled = bool(
            self.loop_config.get("self_rag", {}).get("enabled", False)
        )
        self.self_rag_max_rounds = int(
            self.loop_config.get("self_rag", {}).get("max_rounds", 1)
        )
        # 兜底场景标签（第二篇文章：strict 拒绝编造 / lenient 允许通用知识兜底）
        _fallback = self.loop_config.get("fallback_policy", {})
        self.strict_tags = [
            str(t) for t in _fallback.get("strict_tags", [])
        ]
        self.lenient_tags = [
            str(t) for t in _fallback.get("lenient_tags", [])
        ]
        self.conversation_memory = (
            ConversationMemory(self.vector_store)
            if enable_memory and self.vector_store
            else None
        )

    @staticmethod
    def _load_loop_config(config_path: str = "config/loop_config.yaml") -> dict:
        """加载 Agent 循环/预算配置；缺失或损坏时回退默认值（不阻断启动）"""
        try:
            import yaml
            path = Path(__file__).resolve().parent.parent.parent / config_path
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _build_fallback_rule(self) -> str:
        """生成兜底场景规则（注入 system prompt）：
        strict 场景（制度/财务/合规）检索不到时明确拒绝编造；
        lenient 场景允许用通用知识兜底但必须标注来源。
        """
        lines = []
        if self.strict_tags:
            lines.append(
                "- 以下分类属于严格场景，检索不到时明确告知未找到，"
                "不得用模型内部知识编造："
                + "、".join(self.strict_tags)
            )
        if self.lenient_tags:
            lines.append(
                "- 以下分类属于宽松场景，检索不到时可基于通用知识回答"
                "但必须标注「以下为通用知识，并非企业知识库内容」："
                + "、".join(self.lenient_tags)
            )
        if not lines:
            lines.append(
                "- 默认按严格场景处理：检索不到时明确告知未找到，"
                "不得用模型内部知识编造企业制度/条款。"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 主对话接口
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """主对话接口（ReAct 循环：推理 → 执行 → 观察 → 再推理）

        循环直到 LLM 选择直接回答或达到最大轮次。
        每轮流程：
          1. 当前 messages → LLM（带工具描述）
          2. 如果 LLM 返回 tool_calls → 执行 → 结果追加到 messages → 继续循环
          3. 如果 LLM 返回文本 → 作为最终回答 → 跳出循环

        stream_callback：可选异步回调 async (token: str) -> None。
        传入时"最终回答"一轮用 stream=True 逐 token 产出并回调（工具调用轮
        仍为非流式，因为没有可输出的文本）；不传时行为与之前完全一致。
        """
        context = context or {}
        self._session_turn_count += 1

        # 0. 请求计时 + 语义缓存查询（精确/语义两级命中直接复用答案）
        _t0 = time.perf_counter()
        tool_calls_count = 0
        tool_failures_count = 0
        _query_emb = None
        _cache_ns = self.user_id or "default"
        try:
            if self.vector_store:
                # embedding 是 CPU 计算，丢线程池避免阻塞事件循环
                _query_emb = (await asyncio.to_thread(
                    self.vector_store.encode_text, [message]
                ))[0]
                _cached_answer = self.query_cache.get(
                    message, _query_emb, namespace=_cache_ns
                )
                if _cached_answer:
                    try:
                        from ..eval.metrics import record_request
                        record_request(
                            message,
                            (time.perf_counter() - _t0) * 1000,
                            cached=True,
                        )
                    except Exception:
                        pass
                    # 缓存命中也要更新短期记忆：否则用户追问「你刚才说啥」时
                    # 上下文里没有上一轮回答，对话会断裂
                    if self.short_memory:
                        self.short_memory.add_turn(message, _cached_answer)
                    if self.long_memory and self.user_id:
                        self.long_memory.update_memory(
                            self.user_id,
                            topic="knowledge_query",
                            last_question=message,
                        )
                    return {
                        "query": message,
                        "answer": _cached_answer,
                        "cached": True,
                        "success": True,
                        "intent": IntentType.KNOWLEDGE_QUERY.value,
                        "user_id": self.user_id,
                        "tool_used": None,
                        "context_count": None,
                    }
        except Exception:
            _query_emb = None

        # 1. 构建对话上下文
        history_context = ""
        if self.short_memory:
            history = self.short_memory.get_history()
            if history:
                history_context = self._format_history(history)

        # 2. 注入跨会话语义回忆（检索该用户历史对话中最相关的片段）
        recall_messages = ""
        if self.conversation_memory and self.user_id:
            try:
                # 指代查询先做改写（"那个方案"→具体实体），提升语义召回命中率；
                # 只有命中指代词才触发（低频低成本），失败回退原查询
                recall_query = message
                if self._may_contain_reference(message):
                    try:
                        recall_query = await self._rewrite_query_for_recall(message)
                    except Exception:
                        recall_query = message
                # Chroma 查询 + embedding 都是同步调用，丢线程池执行
                recall = await asyncio.to_thread(
                    self.conversation_memory.semantic_recall,
                    self.user_id, recall_query, 3,
                )
                if recall:
                    recall_lines = ["## 相关历史对话片段"]
                    for r in recall:
                        recall_lines.append(
                            f"- [{r['role']}] {r['content'][:self.recall_max_chars]}"
                        )
                    recall_lines.append("（以上历史片段供参考，不要直接引用为事实）")
                    recall_messages = "\n".join(recall_lines)
            except Exception:
                pass

        # 3. 构建 system prompt（注入 Skill 概览 + 语义回忆）
        skill_intro = self.skill_manager.get_intro() if self.skill_manager else ""
        system_prompt = self.SYSTEM_PROMPT_TPL.format(
            skill_intro=skill_intro,
            fallback_rule=self._build_fallback_rule(),
        )
        if recall_messages:
            system_prompt += "\n\n" + recall_messages
        # 注入用户画像和历史摘要
        user_context = ""
        if self.long_memory and self.user_id:
            try:
                # 注入路由：纯闲聊不需要用户画像/演进式摘要（省 token，
                # 也避免闲聊被记忆带偏）；知识/业务类问题才注入
                if not self._is_small_talk(message):
                    # 偏好时间线注入：当前生效偏好（完整历史经状态过滤），
                    # 交 LLM 综合判断矛盾，发现矛盾先澄清再回答
                    from ..memory.pref_store import get_active
                    prefs = get_active(self.user_id, limit=10)
                    if prefs:
                        pref_lines = [
                            f"- {p['updated_at'][:10]}: {p['text']}"
                            for p in prefs if p.get("text")
                        ]
                        if pref_lines:
                            user_context = "## 用户偏好历史（按时间）\n" + "\n".join(pref_lines)
                            user_context += (
                                "\n（若偏好历史存在矛盾或模糊，先向用户澄清再回答，"
                                "或说明按哪个偏好处理）"
                            )
                    running_summary = self.long_memory.get_running_summary(self.user_id)
                    if running_summary:
                        if user_context:
                            user_context += "\n\n"
                        user_context += (
                            "## 对话摘要\n" + running_summary[:self.summary_max_chars]
                        )
            except Exception:
                pass
        if user_context:
            system_prompt += "\n\n" + user_context

        # 4. 构建 messages
        messages = [{"role": "system", "content": system_prompt}]
        if history_context:
            messages.append({"role": "system", "content": f"对话历史：\n{history_context}"})
        messages.append({"role": "user", "content": message})

        # 5. 获取工具 schemas
        tools = []
        if self.tool_router and self.enable_tools and self.skill_manager:
            tools = self.tool_router.get_tool_schemas()

        # 6. ReAct 循环：推理 → 执行 → 观察 → 再推理，直到 LLM 直接回答
        response_data = {
            "query": message,
            "intent": IntentType.KNOWLEDGE_QUERY.value,
            "user_id": self.user_id,
        }
        last_tool_result = ""  # 兜底/统计复用，先初始化避免作用域问题

        try:
            if not self.rag_pipeline or not self.rag_pipeline.llm_client:
                raise RuntimeError("LLM 客户端未初始化（RAG 未启用）")
            llm_client = self.rag_pipeline.llm_client
            llm_model = self.rag_pipeline.llm_model

            react_messages = list(messages)
            last_tool_name = None
            last_tool_args = None
            # 死循环检测：同一工具 + 同一参数连续调用超过 2 次 → 中断
            tool_call_fingerprints: Dict[str, int] = {}
            # 无进展检测：换了参数但结果基本相同（检索命中集合重复）也算无进展
            last_result_signature = None
            no_progress_count = 0
            # 工具白名单：LLM 可能输出幻觉工具名，先校验再执行
            known_tool_names = (
                set(self.tool_router.get_tool_names())
                if self.tool_router else set()
            )

            for iteration in range(self.max_loop_rounds):
                if stream_callback is not None:
                    stream = await llm_client.chat.completions.create(
                        model=llm_model,
                        messages=react_messages,
                        tools=tools if tools else None,
                        tool_choice="auto" if tools else None,
                        temperature=0.7,
                        max_tokens=self.llm_tokens["react_loop"],
                        stream=True,
                    )
                    msg = await self._collect_stream_message(stream, stream_callback)
                else:
                    response = await llm_client.chat.completions.create(
                        model=llm_model,
                        messages=react_messages,
                        tools=tools if tools else None,
                        tool_choice="auto" if tools else None,
                        temperature=0.7,
                        max_tokens=self.llm_tokens["react_loop"],
                    )
                    msg = response.choices[0].message

                if msg.tool_calls and tools:
                    # 统一转成 OpenAI 消息 dict：流式聚合出的是 SimpleNamespace，
                    # 不能直接回填（SDK 序列化会报错）；非流式 SDK 对象也兼容此转换
                    react_messages.append(self._assistant_message_from(msg))
                    round_has_valid_call = False
                    for tc in msg.tool_calls:
                        func_name = tc.function.name

                        # 0) 工具名白名单校验（幻觉工具名 → 回填错误让 LLM 修正）
                        if known_tool_names and func_name not in known_tool_names:
                            round_has_valid_call = True
                            react_messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    f"未知工具 {func_name}，可用的工具为："
                                    f"{', '.join(sorted(known_tool_names))}。"
                                    "请重新选择工具。"
                                ),
                            })
                            continue

                        # 1) 容错解析工具参数（LLM 可能输出非法 JSON）
                        func_args, parse_ok = self._safe_parse_tool_args(tc.function.arguments)

                        # 2) 死循环检测：相同工具+参数指纹
                        try:
                            _fp = json.dumps(func_args, ensure_ascii=False, sort_keys=True)
                        except (TypeError, ValueError):
                            _fp = str(func_args)
                        _fingerprint = f"{func_name}:{_fp}"
                        tool_call_fingerprints[_fingerprint] = (
                            tool_call_fingerprints.get(_fingerprint, 0) + 1
                        )
                        if tool_call_fingerprints[_fingerprint] > 2:
                            round_has_valid_call = True
                            react_messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    "该工具已用相同参数连续调用多次，疑似循环。"
                                    "请停止重复调用，直接基于已有信息回答。"
                                ),
                            })
                            continue

                        # 3) 参数解析失败 → 回填错误，让 LLM 自行修正（不中断循环）
                        if not parse_ok:
                            round_has_valid_call = True
                            react_messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    f"工具 {func_name} 参数解析失败：模型返回的 arguments "
                                    f"不是合法 JSON（原始内容：{str(tc.function.arguments)[:200]}）。"
                                    "请检查参数格式后重新调用。"
                                ),
                            })
                            continue

                        # 4) 执行工具（异常隔离：单个工具抛错不中断整个对话）
                        try:
                            tool_calls_count += 1
                            if func_name == "search_knowledge_base":
                                # 复杂查询：先多查询分解 + HyDE 增强检索，失败走原路径
                                expanded = await self._search_with_expansion(func_args)
                                if expanded is not None:
                                    tool_result = expanded
                                elif self.self_rag_enabled:
                                    # Self-RAG：检索后 LLM 判断充分性，不足则改写再检
                                    self_rag_result = await self._search_with_self_rag(func_args)
                                    if self_rag_result is not None:
                                        tool_result = self_rag_result
                                    else:
                                        tool_result = await asyncio.to_thread(
                                            self.tool_router.call_skill_tool, func_name, func_args
                                        )
                                else:
                                    tool_result = await asyncio.to_thread(
                                        self.tool_router.call_skill_tool, func_name, func_args
                                    )
                            else:
                                # 工具内部包含检索+精排（同步 CPU/本地 IO），丢线程池执行
                                tool_result = await asyncio.to_thread(
                                    self.tool_router.call_skill_tool, func_name, func_args
                                )
                        except Exception as e:
                            tool_failures_count += 1
                            tool_result = f"工具 {func_name} 执行异常：{e}"

                        # 4.5) 检索空结果兜底（四层框架）：记录失败 query（系统层）、
                        # 低相关候选（交互层）、strict/lenient 场景话术（生成层）
                        if (
                            func_name == "search_knowledge_base"
                            and "未找到相关内容" in tool_result
                        ):
                            tool_result = await self._handle_missed_retrieval(
                                # 记录/抢救用用户原始问题：知识库盲区分析
                                # 面向的是"用户问什么"，而不是 LLM 改写后的工具参数
                                message,
                                str(func_args.get("tag", "")),
                                tool_result,
                            )

                        # 5) 上下文预算：按"窗口 − 当前上下文 − 输出预留 − 安全余量"
                        #    动态计算本轮工具结果上限，超过先 LLM 压缩，失败回退截断
                        _current_chars = sum(
                            len(m.get("content") or "")
                            + sum(
                                len(tc.get("function", {}).get("arguments", "") or "")
                                for tc in (m.get("tool_calls") or [])
                            )
                            for m in react_messages
                        )
                        budget_chars = self._compute_tool_result_budget(
                            _current_chars,
                            self.max_loop_rounds - iteration,
                            self.budget_config,
                        )
                        if len(tool_result) > budget_chars:
                            compressed = await self._compress_tool_result(
                                message, tool_result, max_chars=budget_chars
                            )
                            if compressed:
                                tool_result = compressed + "\n...[内容过长已压缩]"
                            else:
                                tool_result = tool_result[:budget_chars] + "\n...[内容过长已截断]"

                        # 6) 无进展检测：结果签名与上一轮相同 → 计数；连续达到阈值
                        #    则追加系统提示，让 LLM 停止重复检索（指纹防"同参数"，
                        #    无进展防"换参数但结果一样"，两层互补）
                        _sig = self._tool_result_signature(func_name, tool_result)
                        if _sig:
                            if _sig == last_result_signature:
                                no_progress_count += 1
                            else:
                                no_progress_count = 0
                            last_result_signature = _sig
                            if no_progress_count >= self.no_progress_threshold:
                                tool_result += (
                                    "\n\n[系统提示] 本轮检索/工具结果与上一轮基本相同，"
                                    "未获得新信息。请停止重复检索，直接基于已有信息回答，"
                                    "或换一个更具体/不同角度的查询词重新检索。"
                                )

                        round_has_valid_call = True
                        react_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        })
                        last_tool_name = func_name
                        last_tool_args = func_args
                        last_tool_result = tool_result
                    if round_has_valid_call:
                        continue

                answer = msg.content or ""
                break
            else:
                answer = "已达到最大推理轮次，请简化您的问题。"

            response_data.update({
                "answer": answer,
                "tool_used": last_tool_name,
                "tool_args": last_tool_args,
                "success": True,
                # 真实统计检索返回的上下文条数，而不是硬编码 3
                "context_count": (
                    len(re.findall(r"^\s*\[\d+\]", last_tool_result, re.M))
                    if last_tool_name == "search_knowledge_base" else None
                ),
            })

        except Exception as e:
            response_data.update({
                "answer": f"抱歉，处理请求时出现错误：{str(e)}",
                "error": str(e),
                "success": False,
            })

        # 9. 更新短期记忆
        if self.short_memory:
            self.short_memory.add_turn(message, response_data.get("answer", ""))

        # 10. 演进式摘要 + 自动提取用户画像
        #     触发时机 = 短期记忆窗口边界（max_history_length 轮），保证每一段
        #     从窗口滑出的对话都被「旧摘要 + 新增窗口」合并，不会漏掉前几轮
        if self.short_memory and self.long_memory and self.rag_pipeline:
            try:
                turn_count = self._session_turn_count
                summary_window = max(
                    int(self.short_memory.max_history_length), 1
                )
                if self.summary_due(turn_count, summary_window):
                    uid = self.user_id or "default"
                    old_summary = self.long_memory.get_running_summary(uid)
                    recent = self.short_memory.get_context_for_prompt(max_turns=summary_window)
                    prompt_parts = ["你正在维护一段对话的持续摘要，逐轮更新，保留完整上下文。"]
                    if old_summary:
                        prompt_parts.append(f"当前摘要：\n{old_summary}")
                    prompt_parts.append(f"新增对话：\n{recent}")
                    prompt_parts.append(
                        "请更新摘要，并提取用户画像。输出 JSON 格式：\n"
                        '{\n'
                        '  "summary": "更新后的摘要（不超过200字）",\n'
                        '  "user_profile": {\n'
                        '    "role": "用户角色",\n'
                        '    "preferences": ["偏好1"],\n'
                        '    "key_facts": ["事实1", "事实2"]\n'
                        '  }\n'
                        '}'
                    )
                    prompt = "\n\n".join(prompt_parts)
                    resp = await self.rag_pipeline.llm_client.chat.completions.create(
                        model=self.rag_pipeline.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=self.llm_tokens["summary"],
                    )
                    raw = resp.choices[0].message.content or ""
                    _m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
                    if _m:
                        raw = _m.group(1)
                    data = json.loads(raw)
                    if data.get("summary"):
                        self.long_memory.add_summary(uid, data["summary"])
                        # 演进式摘要同步写入向量记忆，便于跨会话按语义召回历史状态
                        if self.conversation_memory:
                            await asyncio.to_thread(
                                self.conversation_memory.add_turn,
                                uid, uid, "assistant", f"[演进式摘要] {data['summary']}",
                            )
                    profile = data.get("user_profile", {})
                    if profile.get("key_facts"):
                        self.long_memory.update_user_profile(
                            uid, preferences={"extracted_facts": profile["key_facts"]},
                            source="summary_extraction",
                        )
            except Exception:
                pass

        # 12. 实时 LLM-as-Judge 评测 + 低分告警 + 自动兜底
        if self.enable_eval:
            try:
                from ..eval.evaluator import Evaluator
                from ..eval.realtime import record as record_score
                from ..eval.alerts import check_and_alert as _alert
                eval_answer = response_data.get("answer", "")
                if eval_answer:
                    # 记录本轮注入的记忆上下文（回忆片段/画像/摘要）：
                    # badcase 回溯时能定位"是不是某条记忆污染带偏了回答"
                    memory_context = "\n".join(
                        part for part in (recall_messages, user_context) if part
                    )[:300]
                    # 传检索上下文：Judge 额外评估"忠实度"（答案是否基于检索资料）
                    score = await Evaluator().evaluate(
                        message, eval_answer, contexts=last_tool_result
                    )
                    if score:
                        record_score(message, eval_answer,
                                     score.relevance, score.completeness,
                                     score.usefulness, score.explanation,
                                     memory_context=memory_context,
                                     faithfulness=score.faithfulness)
                        if (
                            score.usefulness < 3
                            or score.relevance < 3
                            or (
                                score.faithfulness is not None
                                and score.faithfulness < 3
                            )
                        ):
                            _alert(message, eval_answer, score.relevance,
                                   score.completeness, score.usefulness,
                                   score.explanation,
                                   memory_context=memory_context,
                                   faithfulness=score.faithfulness)
                            try:
                                # 复用已有 pipeline：不再重复加载 embedding/rerank 模型
                                contexts = await asyncio.to_thread(
                                    self.rag_pipeline.retrieve,
                                    message, 5, True, None,
                                )
                                if contexts:
                                    ctx_text = "\n\n".join(c["content"] for c in contexts[:3])
                                    if last_tool_result:
                                        ctx_text += "\n\n【首轮工具检索结果】\n" + last_tool_result[:2000]
                                    prompt = (
                                        "请基于以下文档重新回答用户问题。"
                                        f"\n\n文档：\n{ctx_text}"
                                        f"\n\n问题：{message}"
                                    )
                                    resp = await self.rag_pipeline.llm_client.chat.completions.create(
                                        model=self.rag_pipeline.llm_model,
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0.7, max_tokens=self.llm_tokens["fallback"],
                                    )
                                    fallback = resp.choices[0].message.content or ""
                                    if fallback:
                                        response_data["answer"] = fallback
                                        response_data["fallback"] = True
                            except Exception as e:
                                print(f"[EvalFallback] {e}")
            except Exception as e:
                print(f"[Eval] 评分失败: {e}")

        # 11. 更新长程记忆（ChromaDB 语义记忆）
        if self.conversation_memory:
            uid = self.user_id or "default"
            answer = response_data.get("answer", "")
            try:
                await asyncio.to_thread(
                    self.conversation_memory.add_turn, uid, uid, "user", message
                )
                await asyncio.to_thread(
                    self.conversation_memory.add_turn, uid, uid, "assistant", answer
                )
            except Exception as e:
                print(f"[Memory] 写入长程记忆失败: {e}")

        # 11. 更新长期记忆
        if self.long_memory and self.user_id:
            self.long_memory.update_memory(
                self.user_id,
                topic="knowledge_query",
                last_question=message,
            )

        # 12. 写入语义缓存 + 记录监控埋点
        try:
            _answer = response_data.get("answer", "")
            if (
                _answer
                and _query_emb is not None
                and not response_data.get("error")
                and not response_data.get("fallback")
                and response_data.get("success", True)
            ):
                self.query_cache.put(
                    message, _answer, _query_emb, namespace=_cache_ns
                )

            from ..eval.metrics import record_request
            record_request(
                message,
                (time.perf_counter() - _t0) * 1000,
                cached=False,
                error=response_data.get("error"),
                input_chars=sum(len(m.get("content", "")) for m in messages),
                output_chars=len(_answer),
                alert=bool(response_data.get("fallback")),
                tool_calls=tool_calls_count,
                tool_failures=tool_failures_count,
            )
        except Exception:
            pass

        return response_data

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    async def _collect_stream_message(stream, stream_callback) -> SimpleNamespace:
        """聚合流式输出：逐 token 回调，同时累积 content 与 tool_calls。

        流式模式下模型可能走工具调用（delta.tool_calls 增量返回），
        需要按 index 聚合 id/name/arguments，构造出与普通响应一致的
        message 对象，让上层工具分支零改动。
        """
        content_parts = []
        tool_calls_agg: Dict[int, Dict[str, str]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                await stream_callback(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    entry = tool_calls_agg.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

        if tool_calls_agg:
            calls = [
                SimpleNamespace(
                    id=agg["id"],
                    function=SimpleNamespace(name=agg["name"], arguments=agg["arguments"]),
                )
                for _, agg in sorted(tool_calls_agg.items())
            ]
            return SimpleNamespace(content=None, tool_calls=calls)
        return SimpleNamespace(content="".join(content_parts), tool_calls=None)

    @staticmethod
    def _assistant_message_from(msg) -> dict:
        """把 LLM 返回（流式聚合的 SimpleNamespace 或 SDK 对象）转成 OpenAI
        assistant 消息 dict，供下一轮请求回填。"""
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        return {"role": "assistant", "content": getattr(msg, "content", "") or ""}

    # 工具结果压缩上限默认值（实际值以 config/loop_config.yaml 的 budget 段为准；
    # 保留类常量是为了向后兼容与单测构造，运行期一律用 self.tool_result_max_chars）
    MAX_TOOL_RESULT_CHARS = 3000

    # 指代/模糊表述标记：命中才触发查询改写，避免每次召回都多一次 LLM 调用
    REFERENCE_MARKERS = (
        "上次", "之前", "刚才", "那个", "这个",
        "它", "后来", "说过", "提到", "记得",
    )

    # 纯闲聊标记：命中则跳过画像/摘要注入（轻量路由，不做 LLM 判断）
    SMALL_TALK_PHRASES = {
        "你好", "您好", "你好啊", "您好啊", "嗨", "哈喽", "hello", "hi",
        "谢谢", "感谢", "再见", "拜拜", "在吗", "你是谁", "介绍一下你自己",
        "你能做什么", "你会什么", "谢谢你的回答",
    }

    @staticmethod
    def _is_small_talk(query: str) -> bool:
        """轻量判断是否为纯闲聊（去标点后匹配常见问候/寒暄）"""
        normalized = re.sub(r"[\s，。！？!?,.、]+", "", query or "").strip().lower()
        if normalized in AgentManager.SMALL_TALK_PHRASES:
            return True
        # 短问候开头（如"你好，在吗"）也视为闲聊
        return (
            len(normalized) <= 6
            and any(normalized.startswith(p) for p in ("你好", "您好", "嗨", "hi", "hello"))
        )

    @staticmethod
    def _may_contain_reference(query: str) -> bool:
        """轻量检测查询是否可能含指代（零成本，命中才触发 LLM 改写）"""
        return any(m in query for m in AgentManager.REFERENCE_MARKERS)

    async def _compress_tool_result(
        self, query: str, tool_result: str, max_chars: Optional[int] = None
    ) -> Optional[str]:
        """LLM 压缩超长工具结果（Advanced RAG 生成层优化）；失败返回 None 由截断兜底。

        max_chars 是本次压缩的目标长度（上下文预算动态算出的本轮上限），
        不传时回退到配置上限 self.tool_result_max_chars。
        """
        try:
            if not self.rag_pipeline or not self.rag_pipeline.llm_client:
                return None
            from ..rag.context_compressor import compress_text
            return await compress_text(
                self.rag_pipeline.llm_client,
                self.rag_pipeline.llm_model,
                query,
                tool_result,
                max_chars if max_chars is not None else self.tool_result_max_chars,
            )
        except Exception:
            return None

    @staticmethod
    def _compute_tool_result_budget(
        current_chars: int,
        remaining_rounds: int,
        budget_config: Optional[dict] = None,
    ) -> int:
        """按上下文预算公式计算本轮工具结果上限（字符数）。

        公式：可用预算 = 模型窗口 − 当前上下文（system/历史/已有轮次）估算
                     − 输出预留 − 安全余量
        再按剩余轮数均摊（本轮最多用一半，给后续轮留余地），
        最后 clamp 到 [tool_result_min_chars, tool_result_max_chars]。

        current_chars：当前 messages 的字符数估算；remaining_rounds：剩余轮数（≥1）。
        纯函数，便于单测：窗口很小/上下文很高时回退到下限，正常时在上限内。
        """
        cfg = budget_config or {}
        window = int(cfg.get("model_window_tokens", 64000))
        max_chars = int(cfg.get("tool_result_max_chars", 3000))
        min_chars = int(cfg.get("tool_result_min_chars", 500))
        output_reserve = int(cfg.get("output_reserve_tokens", 2000))
        safety = int(cfg.get("safety_margin_tokens", 4000))
        chars_per_token = float(cfg.get("chars_per_token", 1.0))
        remaining = max(int(remaining_rounds), 1)

        current_tokens = int(current_chars * chars_per_token)
        available_tokens = window - current_tokens - output_reserve - safety
        per_round_chars = int(
            available_tokens / remaining * 0.5 * (1.0 / chars_per_token)
        )
        return max(min_chars, min(max_chars, per_round_chars))

    @staticmethod
    def _tool_result_signature(func_name: str, result: str) -> str:
        """工具结果签名：用于无进展检测。

        search_knowledge_base 用「来源章节集合 + 命中段数」作签名——两个不同查询
        若命中的文档集合相同，视为无新信息；其他工具用规范化文本前缀兜底。
        返回空串表示无法签名（空结果等），不参与无进展计数。
        """
        if not result:
            return ""
        if func_name == "search_knowledge_base":
            sections = sorted(
                set(re.findall(r"（来源章节[:：]?\s*(.+)）", result))
            )
            if sections:
                return f"kb:{len(sections)}:{'|'.join(sections)}"
        norm = re.sub(r"\[\d+\]", "", result)
        norm = re.sub(r"\s+", "", norm)[:200]
        return f"text:{norm}"

    async def _handle_missed_retrieval(
        self, query: str, tag: str, tool_result: str
    ) -> str:
        """检索空结果的兜底链（第二篇文章四层框架）：

        1. 系统层：记录失败 query（missed_queries.jsonl），供高频未命中聚合
           反哺知识库（"失败 query 是金矿"闭环）；
        2. 交互层：放大召回（不精排）拿低相关候选，附"仅供参考"提示；
        3. 生成层：按场景标签回填话术——strict 明确拒绝编造，
           lenient 允许通用知识兜底但必须标注来源。
        """
        # 1) 失败 query 日志（系统层闭环）
        try:
            from ..eval.missed_queries import record_missed
            await asyncio.to_thread(record_missed, query, tag, reason="empty")
        except Exception:
            pass

        # 2) 低相关候选（交互层：检索不到精确匹配时退而求其次）
        candidate_hint = ""
        try:
            pipeline = getattr(self.tool_router, "skill_pipeline", None)
            if pipeline is None:
                pipeline = self.rag_pipeline
            if pipeline is not None:
                # 不精排 + 放宽 top_k：找回被相关性阈值过滤掉但可能仍相关的候选
                low_rel = await asyncio.to_thread(
                    pipeline.retrieve, query, 3, False, tag
                )
                low_rel = [c for c in low_rel if c.get("content")]
                if low_rel:
                    parts = [
                        "\n\n以下资料相关度较低，仅供参考"
                        "（未达到精确匹配阈值）："
                    ]
                    for c in low_rel[:2]:
                        sec = (
                            c.get("section", "")
                            or c.get("parent_section", "")
                        )
                        line = f"- {c['content'][:120]}"
                        if sec:
                            line += f"（来源：{sec}）"
                        parts.append(line)
                    candidate_hint = "\n".join(parts)
        except Exception:
            pass

        # 3) 场景话术（生成层：strict / lenient 边界）
        if tag in self.lenient_tags:
            guidance = (
                "\n\n[系统提示] 知识库中未找到直接匹配内容（宽松场景）。"
                "如果用户问题属于通用常识，可以基于通用知识回答，"
                "但必须在开头标注「以下为通用知识，并非企业知识库内容」；"
                "如果是企业专属问题，请引导用户换关键词或补充细节。"
            )
        else:
            guidance = (
                "\n\n[系统提示] 知识库中未找到相关内容（严格场景）。"
                "请明确告知用户未找到，并引导其换关键词或补充细节，"
                "不要用模型内部知识编造企业制度/条款。"
            )
        return tool_result + candidate_hint + guidance

    async def _search_with_self_rag(self, args: dict) -> Optional[str]:
        """Self-RAG 单步反思：检索后 LLM 判断片段是否足够回答，不足则改写查询补检。

        与多查询分解互补：后者在检索前拆分复杂问题，本方法在检索后做充分性判断
        （Agent 驱动迭代检索的反思环节）。返回与 search_knowledge_base 一致的
        格式化结果；失败/未启用时返回 None 走原路径。
        """
        try:
            if not self.rag_pipeline or not self.rag_pipeline.llm_client:
                return None
            pipeline = getattr(self.tool_router, "skill_pipeline", None)
            if pipeline is None:
                return None
            query = str(args.get("query", "")).strip()
            tag = str(args.get("tag", "")).strip()
            if not query or not tag:
                return None

            # 第 1 次检索（复用已加载 pipeline，不重复初始化模型）
            contexts = await asyncio.to_thread(pipeline.retrieve, query, 3, True, tag)
            if not contexts:
                return None
            first = self._format_search_result(query, tag, contexts)

            # 充分性判断：片段能否回答用户问题
            prompt = (
                "你是检索质量评估器。判断下面的检索片段是否足以回答用户问题。\n\n"
                f"用户问题：{query}\n\n检索片段：\n{first}\n\n"
                '仅输出 JSON：{"sufficient": true/false, '
                '"rewritten_query": "不足时改写后的检索词（足够时填空）"}'
            )
            resp = await self.rag_pipeline.llm_client.chat.completions.create(
                model=self.rag_pipeline.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.llm_tokens["self_rag_judge"],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = self._parse_json_loose(raw)
            if data.get("sufficient") is True:
                return first

            rewritten = str(data.get("rewritten_query") or "").strip()
            if not rewritten or rewritten == query:
                return first

            # 补检：最多 self_rag_max_rounds 次，每次用改写查询再检并按内容去重合并
            seen = {c.get("content", "")[:80] for c in contexts}
            merged = list(contexts)
            for _ in range(max(self.self_rag_max_rounds, 1)):
                contexts2 = await asyncio.to_thread(
                    pipeline.retrieve, rewritten, 3, True, tag
                )
                if not contexts2:
                    break
                added = False
                for c in contexts2:
                    key = c.get("content", "")[:80]
                    if key and key not in seen:
                        merged.append(c)
                        seen.add(key)
                        added = True
                if not added:
                    # 补检未带来新信息：即使配置允许多次，也不做无意义重复
                    break
            return self._format_search_result(query, tag, merged[:3])
        except Exception:
            return None

    @staticmethod
    def _format_search_result(query: str, tag: str, contexts: list) -> str:
        """把检索结果格式化成与 search_knowledge_base 工具一致的文本"""
        if not contexts:
            return f"知识库「{tag}」中未找到相关内容。"
        result = f"检索到以下相关文档（来源：{tag}）：\n\n"
        for i, ctx in enumerate(contexts):
            result += f"[{i+1}] {ctx.get('content', '')}\n"
            sec = ctx.get("section", "") or ctx.get("parent_section", "")
            if sec:
                result += f"  （来源章节：{sec}）\n"
            result += "\n"
        return result

    @staticmethod
    def _parse_json_loose(raw: str) -> dict:
        """宽松解析 LLM 输出的 JSON（容忍代码块包裹/前后说明文字）"""
        if not raw:
            return {}
        _m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if _m:
            raw = _m.group(1)
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(raw[brace_start:brace_end + 1])
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    async def _search_with_expansion(self, args: dict) -> Optional[str]:
        """复杂查询增强检索：分解子查询 + HyDE 假设文档，多路检索合并去重。

        Returns:
            合并后的格式化结果；无需增强/失败返回 None（走原单次检索路径）。
        """
        try:
            if not self.rag_pipeline or not self.rag_pipeline.llm_client:
                return None
            query = str(args.get("query", "")).strip()
            tag = str(args.get("tag", "")).strip()
            if not query or not tag:
                return None
            expand_chars = int(
                self.loop_config.get("query_expansion", {}).get(
                    "max_chars_trigger", 25
                )
            )
            if not self._should_expand_query(query, expand_chars):
                return None

            from ..rag.query_expansion import decompose_query, generate_hypothetical_doc

            sub_queries = await decompose_query(
                self.rag_pipeline.llm_client, self.rag_pipeline.llm_model, query
            )
            candidates = [query] + [q for q in sub_queries if q != query]
            hyde = await generate_hypothetical_doc(
                self.rag_pipeline.llm_client, self.rag_pipeline.llm_model, query
            )
            if hyde:
                candidates.append(hyde)

            merged: Dict[str, Dict] = {}
            for q in candidates:
                ctxs = await asyncio.to_thread(
                    self.rag_pipeline.retrieve, q, 3, True, tag
                )
                for ctx in ctxs:
                    key = ctx.get("content", "")
                    if key and key not in merged:
                        merged[key] = ctx
            results = list(merged.values())[:3]
            if not results:
                return None

            lines = ["检索到以下相关文档：\n"]
            for i, ctx in enumerate(results):
                lines.append(f"[{i+1}] {ctx.get('content', '')}")
                sec = ctx.get("section", "") or ctx.get("parent_section", "")
                if sec:
                    lines.append(f"  来源：{sec}")
                lines.append("")
            lines.append("（多路查询合并结果）")
            return "\n".join(lines)
        except Exception:
            return None

    @staticmethod
    def _should_expand_query(query: str, max_chars_trigger: int = 25) -> bool:
        """规则触发多查询增强：问题含多主题词或较长（长度阈值可配置）"""
        markers = ("分别", "对比", "以及", "哪些", "所有", "不同", "各")
        return any(m in query for m in markers) or len(query) > max_chars_trigger

    async def _rewrite_query_for_recall(self, query: str) -> str:
        """查询改写/指代补全：把"那个方案"补全成具体实体。

        向量检索对"无实体词的纯指代"会失效（三条记忆相似度几乎相同），
        先用 LLM 把指代补全成明确实体再检索；失败/无 LLM 时回退原查询。
        """
        if not self.rag_pipeline or not self.rag_pipeline.llm_client:
            return query
        prompt = (
            "用户的问题里可能包含指代（如「那个方案」「之前说的」），"
            "请把它补全成一句明确的、可用于检索历史对话的查询。"
            "只输出补全后的查询，不要任何解释或多余标点。\n\n"
            f"原问题：{query}\n\n补全后："
        )
        try:
            resp = await self.rag_pipeline.llm_client.chat.completions.create(
                model=self.rag_pipeline.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.llm_tokens["recall_rewrite"],
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            return rewritten or query
        except Exception:
            # LLM 不可用/改写失败 → 回退原查询，不让回忆链路挂掉
            return query

    @staticmethod
    def summary_due(turn_count: int, summary_window: int) -> bool:
        """判断是否到达演进式摘要触发点：短期记忆窗口边界（第 N、2N、3N 轮）。

        N = max_history_length：第 N 轮时摘要「空摘要 + 第 1-N 轮」，第 2N 轮时
        摘要「旧摘要 + 第 N+1-2N 轮」，保证滑出窗口的对话不会丢失。
        """
        return turn_count > 0 and turn_count % max(int(summary_window), 1) == 0

    @staticmethod
    def _format_facts_for_prompt(
        facts,
        max_facts: int = 5,
        max_chars: int = 80,
    ) -> list:
        """画像注入前格式化：兼容旧版 list[str] 与新版 list[dict]（带时间戳）。

        按 updated_at 倒序取最新，控制条数与每条长度，避免撑爆 system prompt；
        时间戳附在条目后（如"（2026-08-12）"），供 LLM 判断时效性。
        """
        normalized = []
        for f in facts or []:
            if isinstance(f, str):
                normalized.append({"text": f, "updated_at": ""})
            elif isinstance(f, dict):
                normalized.append({
                    "text": str(f.get("text", "")),
                    "updated_at": str(f.get("updated_at", "")),
                })
        normalized.sort(key=lambda x: x["updated_at"], reverse=True)

        lines = []
        for f in normalized[:max_facts]:
            text = f["text"].strip()[:max_chars]
            if not text:
                continue
            ts = f["updated_at"][:10] if f["updated_at"] else ""
            lines.append(f"- {text}" + (f"（{ts}）" if ts else ""))
        return lines

    @staticmethod
    def _safe_parse_tool_args(raw_args: Optional[str]) -> tuple:
        """容错解析工具参数 JSON。

        LLM 输出的 arguments 可能是非法 JSON（含注释、单引号、多余文字等）。
        依次尝试：标准 JSON → 提取花括号子串 → ast 宽松解析。

        Returns:
            (args_dict, is_ok)：is_ok=False 表示解析失败，args_dict 为空 dict
        """
        if not raw_args or not str(raw_args).strip():
            return {}, False
        raw = str(raw_args).strip()

        # 1) 标准解析
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data, True
            return {}, False
        except (json.JSONDecodeError, TypeError):
            pass

        # 2) 提取最外层花括号内容再解析（模型可能混入注释/说明文字）
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(raw[brace_start:brace_end + 1])
                if isinstance(data, dict):
                    return data, True
            except (json.JSONDecodeError, TypeError):
                pass

        # 3) ast 宽松解析（容忍单引号、True/None 等 Python 风格字面量）
        try:
            import ast
            data = ast.literal_eval(raw)
            if isinstance(data, dict):
                return data, True
        except (ValueError, SyntaxError):
            pass

        return {}, False

    @staticmethod
    def _format_history(history: list) -> str:
        formatted = []
        for turn in history:
            formatted.append(f"用户: {turn['user']}")
            formatted.append(f"助手: {turn['assistant']}")
        return "\n".join(formatted)

    def get_conversation_history(self) -> list:
        if self.short_memory:
            return self.short_memory.get_history()
        return []

    def clear_memory(self) -> None:
        if self.short_memory:
            self.short_memory.clear()

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "memory_enabled": self.enable_memory,
            "rag_enabled": self.enable_rag,
            "tools_enabled": self.enable_tools,
        }
        if self.rag_pipeline:
            stats["rag_stats"] = self.rag_pipeline.get_stats()
        if self.short_memory:
            stats["history_length"] = len(self.short_memory.get_history())
        return stats
