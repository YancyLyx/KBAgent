# -*- coding: utf-8 -*-
"""工具路由器模块

统一管理 Agent 可调用工具：
- Skill 管理器动态注册的知识库检索工具（read_category_info / search_knowledge_base）
- 文件操作工具（read_file / write_file / create_file / list_files）

工具注册表与 OpenAI Function Calling schema 保持一致：Agent 只能调用
schema 里暴露的工具，避免「注册了但模型看不到」的历史遗留死工具。
"""

import json
from typing import Dict, Any


class ToolRouter:
    """工具路由器

    管理所有可用工具，根据参数路由到具体工具执行。
    """

    def __init__(self):
        """初始化工具路由器"""
        # Skill 管理器（外部设置，由 AgentManager 在初始化时注入）
        self.skill_manager = None

    def get_tool_schemas(self) -> list:
        """返回所有工具的 OpenAI Function Calling schema

        包含内置工具和 SkillManager 注册的知识库工具。
        """
        schemas = []
        if self.skill_manager:
            schemas.extend(self.skill_manager.get_tool_schemas())
        return schemas

    def get_tool_names(self) -> list:
        """返回当前可调用工具名列表（白名单校验用）"""
        return [
            s["function"]["name"]
            for s in self.get_tool_schemas()
            if s.get("function", {}).get("name")
        ]

    def call_skill_tool(self, name: str, args: dict) -> str:
        """调用 SkillManager 的工具（read_category_info / search_knowledge_base）"""
        try:
            # 参数防御：兼容 str / dict 两种入参
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if not isinstance(args, dict):
                args = {}

            if not self.skill_manager:
                return "Skill 管理器未初始化"

            if name == "read_category_info":
                tag = str(args.get("tag", ""))
                if not tag.strip():
                    return (
                        "参数缺失：read_category_info 需要必填参数 tag "
                        "（知识库标签，如「文献」）。请补充后重新调用。"
                    )
                return self.skill_manager.get_category_reference(tag)

            if name == "search_knowledge_base":
                query = str(args.get("query", ""))
                tag = str(args.get("tag", ""))
                if not query.strip() or not tag.strip():
                    return (
                        "参数缺失：search_knowledge_base 需要必填参数 "
                        "query（检索问题）和 tag（知识库标签）。请补充后重新调用。"
                    )
                # 调用 RAGPipeline 检索（优先重用已有 pipeline）
                pipeline = getattr(self, "skill_pipeline", None)
                if pipeline is None:
                    from ..rag.rag_pipeline import RAGPipeline
                    pipeline = RAGPipeline()
                contexts = pipeline.retrieve(query=query, top_k=3, use_rerank=True, tag=tag)
                if not contexts:
                    return "该知识库中未找到相关内容"
                result = "检索到以下相关文档：\n\n"
                for i, ctx in enumerate(contexts):
                    result += f"[{i+1}] {ctx.get('content', '')}\n"
                    sec = ctx.get("section", "") or ctx.get("parent_section", "")
                    if sec:
                        result += f"  来源：{sec}\n"
                    result += "\n"
                return result

            return f"未知的 Skill 工具: {name}"
        except Exception as e:
            return f"工具 {name} 执行异常：{e}"
