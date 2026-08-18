# -*- coding: utf-8 -*-
"""Skill 管理器

管理技能目录中的多个技能，每个技能包含 skill.md 概览和一组工具。
当前技能：
  - knowledge_retrieval：知识库检索，支持按分类检索文档
  - file_tools：文件操作，支持读写文件和目录浏览
  - web_search：互联网搜索（预留）

参考（reference）文件存储在 skills/knowledge_retrieval/references/ 下，
是每个分类的详细说明。新标签上传后首次访问时会自动生成。
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional


SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


class SkillManager:
    """技能管理器

    管理 skills/ 目录下的多个技能，聚合工具描述和概览。
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.skills_dir = SKILLS_DIR

    # ------------------------------------------------------------------
    # 技能概览 —— 注入 system prompt
    # ------------------------------------------------------------------

    def get_intro(self) -> str:
        """返回所有技能的概览"""
        parts = []
        # 知识库检索技能（从 skill.md 读取，已包含动态标签列表）
        parts.append(self._load_md("knowledge_retrieval", "skill.md"))
        # 文件操作技能
        parts.append(self._load_md("file_tools", "skill.md"))
        web = self._load_md("web_search", "skill.md")
        parts.append(f"（此外还有 {web.split(chr(10))[0]}）")
        return "\n\n---\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # 工具注册
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict]:
        """返回所有技能的工具 schema"""
        schemas = []
        schemas.extend(self._knowledge_tool_schemas())
        schemas.extend(self._file_tool_schemas())
        return schemas

    def _knowledge_tool_schemas(self) -> List[Dict]:
        """知识库检索技能的工具"""
        tags = self._get_tags()
        if not tags:
            return []
        schema_ref = self._tool_schema(
            "read_category_info",
            "查看指定知识库的详细参考，包括内容范围、典型问题和使用示例。调用此工具后再决定是否使用 search_knowledge_base。",
            {"tag": {"type": "string", "enum": tags, "description": "知识库标签"}},
            ["tag"],
        )
        schema_search = self._tool_schema(
            "search_knowledge_base",
            "在指定知识库中检索与问题相关的文档片段。先使用 read_category_info 了解各知识库内容后使用本工具。",
            {"query": {"type": "string", "description": "搜索关键词或问题"},
             "tag": {"type": "string", "enum": tags, "description": "知识库标签"}},
            ["query", "tag"],
        )
        return [schema_ref, schema_search]

    def _file_tool_schemas(self) -> List[Dict]:
        """文件操作技能的工具"""
        return [
            self._tool_schema(
                "read_file",
                "读取指定文件的内容。path 为相对于项目根目录的路径。",
                {"path": {"type": "string", "description": "文件路径（相对项目根目录）"}},
                ["path"],
            ),
            self._tool_schema(
                "write_file",
                "将内容写入指定文件（覆盖已有内容）。path 为相对于项目根目录的路径。",
                {"path": {"type": "string", "description": "文件路径"},
                 "content": {"type": "string", "description": "写入的内容"}},
                ["path", "content"],
            ),
            self._tool_schema(
                "create_file",
                "创建一个新文件并写入内容。如果文件已存在则返回错误。",
                {"path": {"type": "string", "description": "文件路径"},
                 "content": {"type": "string", "description": "文件内容"}},
                ["path", "content"],
            ),
            self._tool_schema(
                "list_files",
                "列出指定目录下的文件和子目录。",
                {"path": {"type": "string", "description": "目录路径（默认为项目根目录）"}},
                ["path"],
            ),
        ]

    @staticmethod
    def _tool_schema(name: str, desc: str, properties: dict, required: list) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    def execute_tool(self, name: str, args: dict) -> str:
        """执行指定工具"""
        handlers = {
            "read_category_info": self._read_category_info,
            "search_knowledge_base": self._search_knowledge_base,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "create_file": self._create_file,
            "list_files": self._list_files,
        }
        handler = handlers.get(name)
        if not handler:
            return f"错误：未知的工具 {name}"
        return handler(**args)

    def _read_category_info(self, tag: str) -> str:
        """读取某个分类的参考文件"""
        ref_path = self.skills_dir / "knowledge_retrieval" / "references" / f"{tag}.md"
        # 如果参考文件不存在，从模板生成
        if not ref_path.exists():
            self._generate_reference(tag, ref_path)
        return self._read_md(ref_path)

    def get_category_reference(self, tag: str) -> str:
        """查看指定分类的详细参考（ToolRouter 调用入口）。

        真实环境验证发现的 bug：ToolRouter 一直调用 get_category_reference，
        但此前只有私有方法 _read_category_info，read_category_info 工具
        每次都失败被吞（mock 测试里 MockSkill 有该方法所以没抓到）。
        """
        return self._read_category_info(tag)

    def _search_knowledge_base(self, query: str, tag: str) -> str:
        """检索知识库"""
        if not self.vector_store:
            return "错误：知识库未初始化"
        from ..rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        contexts = pipeline.retrieve(query=query, top_k=3, use_rerank=True, tag=tag)
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

    def _read_file(self, path: str) -> str:
        """读取文件"""
        try:
            full = self._resolve_path(path)
        except ValueError as e:
            return f"错误：{e}"
        if not full.exists():
            return f"错误：文件不存在 {path}"
        if not full.is_file():
            return f"错误：{path} 不是文件"
        if self._is_sensitive_path(full):
            return f"错误：{path} 是敏感配置文件，禁止读取"
        try:
            content = full.read_text(encoding="utf-8")
            return f"文件 {path} 内容：\n```\n{content}\n```"
        except Exception as e:
            return f"读取文件失败：{e}"

    def _write_file(self, path: str, content: str) -> str:
        """写入文件"""
        try:
            full = self._resolve_path(path)
        except ValueError as e:
            return f"错误：{e}"
        if not full.exists():
            return f"错误：文件不存在 {path}，请使用 create_file 创建新文件"
        if self._is_sensitive_path(full):
            return f"错误：{path} 是敏感配置文件，禁止修改"
        try:
            full.write_text(content, encoding="utf-8")
            return f"文件 {path} 已更新。"
        except Exception as e:
            return f"写入文件失败：{e}"

    def _create_file(self, path: str, content: str) -> str:
        """创建新文件"""
        try:
            full = self._resolve_path(path)
        except ValueError as e:
            return f"错误：{e}"
        if full.exists():
            return f"错误：文件已存在 {path}"
        if self._is_sensitive_path(full):
            return f"错误：{path} 是敏感配置文件，禁止创建"
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return f"文件 {path} 已创建。"
        except Exception as e:
            return f"创建文件失败：{e}"

    def _list_files(self, path: str = ".") -> str:
        """列出目录"""
        try:
            full = self._resolve_path(path)
        except ValueError as e:
            return f"错误：{e}"
        if not full.exists():
            return f"错误：目录不存在 {path}"
        if not full.is_dir():
            return f"错误：{path} 不是目录"
        try:
            items = sorted(full.iterdir())
            dirs = [p.name + "/" for p in items if p.is_dir()]
            files = [p.name for p in items if p.is_file()]
            result = f"目录 {path} 内容：\n"
            for d in dirs:
                result += f"  📁 {d}\n"
            for f in files:
                result += f"  📄 {f}\n"
            return result
        except Exception as e:
            return f"列出目录失败：{e}"

    # ------------------------------------------------------------------
    # 参考文件管理
    # ------------------------------------------------------------------

    def _generate_reference(self, tag: str, target_path: Path):
        """为新标签生成参考文件（使用 file_tools 创建）"""
        template = self.skills_dir / "knowledge_retrieval" / "references" / "_template.md"
        if template.exists():
            text = template.read_text(encoding="utf-8")
            text = text.replace("{{tag}}", tag)
        else:
            text = f"# {tag}\n\n## 内容范围\n此知识库包含与 {tag} 相关的文档。\n"
        rel_path = str(target_path.relative_to(self.PROJECT_ROOT))
        self._create_file(rel_path, text)
        print(f"[SkillManager] 已为标签「{tag}」生成参考文件")

    def update_skill_overview(self):
        """动态更新 skill.md：追加当前标签列表到末尾"""
        md_path = self.skills_dir / "knowledge_retrieval" / "skill.md"
        if not md_path.exists():
            return
        tags = self._get_tags()
        tag_line = f"\n\n---\n\n当前知识库分类：{' · '.join(tags) if tags else '（暂无分类）'}"
        # 读取原始内容（去掉上次的动态追加部分）
        base = md_path.read_text(encoding="utf-8").split("\n\n---\n\n当前知识库分类")[0].strip()
        new_content = base + tag_line
        rel_path = "skills/knowledge_retrieval/skill.md"
        self._write_file(rel_path, new_content)

    def sync_references(self):
        """同步参考文件：检查所有标签是否有对应的参考文件，缺失则生成"""
        if not self.vector_store:
            return
        tags = self._get_tags()
        ref_dir = self.skills_dir / "knowledge_retrieval" / "references"
        for tag in tags:
            ref_path = ref_dir / f"{tag}.md"
            if not ref_path.exists():
                self._generate_reference(tag, ref_path)
        # 更新 skill 总览
        self.update_skill_overview()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_tags(self) -> List[str]:
        if not self.vector_store:
            return []
        return self.vector_store.get_available_tags()

    def _load_md(self, *parts: str) -> str:
        """从 skills/ 目录加载 markdown 文件"""
        path = self.skills_dir.joinpath(*parts)
        return self._read_md(path) if path.exists() else ""

    @staticmethod
    def _read_md(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def _resolve_path(self, path: str) -> Path:
        """解析路径，限制在项目根目录内"""
        p = Path(path)
        if not p.is_absolute():
            p = self.PROJECT_ROOT / p
        p = p.resolve()
        # 安全检查：不允许跳出项目根目录
        if not str(p).startswith(str(self.PROJECT_ROOT)):
            raise ValueError(
                f"路径越界，禁止访问项目根目录以外的文件: {path}"
            )
        return p

    @staticmethod
    def _is_sensitive_path(p: Path) -> bool:
        """敏感文件识别：.env*（可能含 API Key）与 .git*（可能含凭据/配置）"""
        for part in p.parts:
            if part.startswith(".env") or part.startswith(".git"):
                return True
        return False
