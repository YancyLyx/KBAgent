# -*- coding: utf-8 -*-
"""RAG 流水线模块

整合分块、检索、重排序等功能，提供完整的 RAG 流程。
"""

from typing import List, Dict, Any, Optional
import os
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .chunker import Chunker
from .pdf_chunker import PDFChunker
from .markdown_chunker import MarkdownChunker
from .parent_child_chunker import ParentChildChunker
from .vector_store import VectorStore
from .retriever import Retriever
from .reranker import Reranker


class RAGPipeline:
    """RAG 流水线"""

    def __init__(
        self,
        config_path: str = "config/rag_config.yaml",
        collection_name: str = "knowledge_base"
    ):
        """
        初始化 RAG 流水线

        Args:
            config_path: 配置文件路径
            collection_name: 向量集合名称
        """
        # 加载环境变量
        load_dotenv()
        self.config_path = config_path
        
        # 初始化 LLM 客户端（超时 + SDK 内置重试；异步客户端：网络等待不占线程）
        self.llm_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-v4-pro")
        
        # 初始化 RAG 组件
        self.chunker = Chunker(config_path)
        self.vector_store = VectorStore(collection_name, config_path)
        self.retriever = Retriever(self.vector_store, config_path)
        self.reranker = Reranker(config_path)
        # 重排层增强配置（默认全开，retrieve 参数为 None 时读这里；
        # 显式传值可覆盖）：
        # - score_threshold → 阈值过滤（min_rerank_score 默认值）
        # - autocut / autocut_drop_ratio → 分数悬崖动态截断
        # - diversity_rerank → MMR 去冗余
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _cfg = yaml.safe_load(f) or {}
            _retrieval = _cfg.get("retrieval_strategy", {})
            self.autocut_enabled = bool(_retrieval.get("autocut", False))
            self.autocut_drop_ratio = float(_retrieval.get("autocut_drop_ratio", 0.3))
            self.min_rerank_score_default = float(
                _retrieval.get("score_threshold", 0.5)
            )
            self.diversity_rerank_enabled = bool(
                _retrieval.get("diversity_rerank", False)
            )
        except Exception:
            self.autocut_enabled = False
            self.autocut_drop_ratio = 0.3
            self.min_rerank_score_default = 0.5
            self.diversity_rerank_enabled = False

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每个文档包含 content 和 metadata
        """
        # 分块
        chunks = self.chunker.split_documents(documents)
        print(f"文档分块完成，共 {len(chunks)} 个块")
        
        # 添加到向量库
        self.vector_store.add_documents(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        use_rerank: bool = True,
        tag: Optional[str] = None,
        diversity_rerank: Optional[bool] = None,
        min_rerank_score: Optional[float] = None,
        autocut: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档

        使用父子分块策略：检索子块（混合检索）→ 按 parent_id 去重 →
        返回父块 → 可选重排序。支持按标签（知识库分类）过滤。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_rerank: 是否使用重排序
            tag: 知识库标签，只检索该分类下的文档。None 时不限分类
            autocut: 是否启用分数悬崖动态截断（None 读配置，默认关）

        Returns:
            检索结果列表
        """
        # 父子检索（内部执行 hybrid: vector + BM25）
        retrieval_top_k = top_k * 3 if use_rerank else top_k
        filters = {"tag": tag} if tag else None
        results = self.retriever.parent_child_search(
            query,
            top_k=retrieval_top_k,
            filters=filters,
        )
        
        if not use_rerank or not results:
            return results[:top_k]
        
        # 重排序（在父块级别）：阈值过滤 + Autocut 在精排内部（精排后），
        # MMR 在精排之后、返回前。三个增强默认从配置读（全开），可显式覆盖
        threshold = (
            min_rerank_score
            if min_rerank_score is not None
            else getattr(self, "min_rerank_score_default", None)
        )
        use_mmr = (
            diversity_rerank
            if diversity_rerank is not None
            else getattr(self, "diversity_rerank_enabled", False)
        )
        reranked_results = self.reranker.rerank(
            query,
            results,
            top_k=top_k,
            threshold=threshold,
            autocut=(
                getattr(self, "autocut_enabled", False)
                if autocut is None else autocut
            ),
            drop_ratio=getattr(self, "autocut_drop_ratio", 0.3),
        )
        if use_mmr and len(reranked_results) > 1:
            # MMR 去冗余：相关性 + 多样性平衡，避免 top-k 语义重复
            reranked_results = self._apply_mmr(query, reranked_results, top_k=top_k)
        return reranked_results

    def _apply_mmr(
        self,
        query: str,
        reranked: List[Dict[str, Any]],
        top_k: int = 3,
        lambda_: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """MMR 去冗余：对重排后文档编码，按相关性与多样性重新选择顺序"""
        texts = [d.get("content", "") for d in reranked]
        embs = self.vector_store.encode_text(texts)
        scores = [float(d.get("rerank_score", 0)) for d in reranked]
        from .mmr import mmr_rerank
        order = mmr_rerank(embs, scores, lambda_=lambda_, top_k=top_k)
        return [reranked[i] for i in order]

    async def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> str:
        """
        基于上下文生成回答

        Args:
            query: 用户查询
            contexts: 检索到的上下文
            temperature: 生成温度

        Returns:
            生成的回答
        """
        # 构建上下文文本
        context_text = "\n\n".join([
            f"[文档 {i+1}] {ctx.get('content', '')}"
            for i, ctx in enumerate(contexts)
        ])
        
        # 构建提示词
        system_prompt = """你是一个专业的企业智能客服助手。请基于提供的参考文档回答用户问题。
如果参考文档中没有相关信息，请明确告知用户无法回答。
请用中文回答，保持简洁专业。"""
        
        user_prompt = f"""参考文档：
{context_text}

用户问题：{query}

请基于以上参考文档回答问题。"""
        
        # 调用 LLM
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    async def query(
        self,
        query: str,
        top_k: int = 3,
        use_rerank: bool = True,
        temperature: float = 0.7,
        tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        完整的 RAG 查询流程

        Args:
            query: 用户查询
            top_k: 检索结果数量
            use_rerank: 是否使用重排序
            temperature: 生成温度
            tag: 知识库标签

        Returns:
            包含回答和检索结果的字典
        """
        # 检索相关文档
        contexts = self.retrieve(query, top_k=top_k, use_rerank=use_rerank, tag=tag)
        
        # 生成回答
        answer = await self.generate(query, contexts, temperature)
        
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "context_count": len(contexts)
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_store.get_collection_stats()

    def add_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加文件到知识库（自动识别文件类型并进行父子分块）

        支持 PDF / Markdown / 纯文本。
        流程：语义级分块（按章节）→ 父子分块（子块索引 → 父块返回给 LLM）。

        Args:
            file_path: 文件路径
            metadata: 附加元数据
        """
        meta = dict(metadata or {})
        fname = file_path.split("/")[-1] if "/" in file_path else file_path
        meta.setdefault("source", fname)
        meta.setdefault("doc_path", file_path)
        meta.setdefault("doc_type", "file")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            chunker = PDFChunker(self.config_path)
            parents = chunker.chunk_pdf(file_path, metadata=meta)
        elif ext == ".md":
            chunker = MarkdownChunker(self.config_path)
            parents = chunker.chunk_md(file_path, metadata=meta)
        else:
            # txt / 无后缀 → 走原本文本分块
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            parents = self.chunker.split_text(
                content,
                metadata=meta,
            )

        # 父子分块
        pc_chunker = ParentChildChunker(self.config_path)
        children = pc_chunker.chunk(parents)

        print(f"文件解析完成，父块 {len(parents)} 个，子块 {len(children)} 个")
        self.vector_store.add_documents(children)

    def add_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加 PDF 文档到知识库（委托给 add_file）"""
        meta = dict(metadata or {})
        meta.setdefault("doc_type", "pdf")
        self.add_file(pdf_path, metadata=meta)

    def delete_documents_by_source(self, source: str) -> int:
        """按文档源文件名删除全部向量分块（文档删除/重索引时调用）。

        分块入库时 metadata.source 即文件名；删除后同步重建 BM25 索引，
        避免「文件已删但检索仍能召回」的脏数据问题。

        Returns:
            删除的分块数
        """
        return self.vector_store.delete_by_metadata({"source": source})
