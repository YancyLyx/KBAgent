# -*- coding: utf-8 -*-
"""管理后台API模块

提供企业端管理后台所需的API接口：
- 文档管理（上传/列表/删除/预览）
- 配置管理（RAG/记忆策略CRUD）
- 统计API（对话统计/查询分析）
"""

import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..rag.chunker import Chunker
from ..rag.vector_store import VectorStore
from ..db.document_store import DocumentStore
from .admin_auth import (
    get_admin_username,
    issue_admin_token,
    verify_admin_token as _verify_admin_sig,
    verify_password as _verify_admin_password,
)

# 创建路由
router = APIRouter(prefix="/api/admin", tags=["admin"])


def _deep_merge(base: dict, patch: dict) -> dict:
    """深度合并配置：patch 只覆盖 base 中存在的键，未提交的配置段保持原样。

    配置更新接口是「局部修改」语义，不能整文件覆盖——否则前端只提交
    chunk_strategy 时，embedding/parent_child/rerank 等段落会被清空。
    """
    result = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ============== 数据模型 ==============

class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    original_name: str
    file_type: str
    size: int
    tag: str = ""
    status: str  # pending, processing, indexed, error
    chunk_count: Optional[int] = None
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    items: List[DocumentInfo]
    page: int
    page_size: int


class RAGConfig(BaseModel):
    """RAG配置"""
    chunk_strategy: dict = Field(default_factory=dict)
    parent_child: dict = Field(default_factory=dict)
    retrieval_strategy: dict = Field(default_factory=dict)
    embedding: dict = Field(default_factory=dict)
    rerank: dict = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """记忆配置"""
    short_term_memory: dict = Field(default_factory=dict)
    long_term_memory: dict = Field(default_factory=dict)
    memory_topics: List[str] = Field(default_factory=list)


class OverviewStats(BaseModel):
    """概览统计"""
    total_conversations: int
    total_users: int
    total_documents: int
    today_conversations: int
    active_users_7d: int
    query_types: dict


class ConversationStats(BaseModel):
    """对话统计"""
    date: str
    count: int
    avg_response_time: Optional[float] = None


class DeleteMemoryFactRequest(BaseModel):
    """删除单条偏好条目的请求"""
    text: str = Field(..., description="要删除的偏好条目原文")


class AdminLoginRequest(BaseModel):
    """管理员登录请求"""
    username: str = Field(..., description="管理员用户名")
    password: str = Field(..., description="管理员密码")


# ============== 全局状态 ==============

# 文档存储路径
DOCS_STORAGE_PATH = Path("./data/uploaded_docs")
DOCS_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# 对话日志（用于统计）- 仍使用内存，因为对话日志量大且不需要持久化
_conversation_logs: list = []


# ============== 依赖注入 ==============

async def verify_admin_token(
    authorization: Optional[str] = Header(None),
):
    """验证管理员 token（Authorization: Bearer <token>），无效返回 401"""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    username = _verify_admin_sig(token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权：请先通过 /api/admin/login 获取 admin token",
        )
    return {"admin": True, "username": username}


@router.post("/login")
async def admin_login(body: AdminLoginRequest):
    """管理员登录：校验用户名密码，签发带过期的 admin token"""
    if (
        body.username != get_admin_username()
        or not _verify_admin_password(body.password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    ttl = int(os.getenv("ADMIN_TOKEN_TTL_SECONDS", str(24 * 3600)))
    token = issue_admin_token(body.username, ttl)
    return {
        "token": token,
        "username": body.username,
        "expires_in": ttl,
    }


# ============== 文档管理API ==============

@router.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    tag: str = Form(""),
    admin: dict = Depends(verify_admin_token)
):
    """
    上传文档到知识库

    支持格式: .txt, .md, .pdf, .docx
    """
    # 验证文件类型
    allowed_extensions = {'.txt', '.md', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file_ext}"
        )

    # 生成文档ID
    doc_id = str(uuid.uuid4())[:8]

    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_name = f"{doc_id}_{timestamp}{file_ext}"
    file_path = DOCS_STORAGE_PATH / storage_name

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 获取文件大小
        file_size = os.path.getsize(file_path)

        # 创建文档记录
        doc_info = {
            "id": doc_id,
            "filename": storage_name,
            "original_name": file.filename,
            "file_type": file_ext,
            "size": file_size,
            "tag": tag,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # 保存到数据库
        DocumentStore.create(doc_info)

        # 自动触发索引（异步处理）
        import asyncio
        asyncio.create_task(_process_document_indexing(doc_id, file_path, file_ext, tag))

        return DocumentInfo(**doc_info)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件上传失败: {str(e)}"
        )


async def _process_document_indexing(doc_id: str, file_path: Path, file_ext: str, tag: str = ""):
    """后台处理文档索引（使用新父子分块管线）"""
    try:
        # 更新状态为处理中
        DocumentStore.update(doc_id, {
            "status": "processing",
            "updated_at": datetime.now().isoformat()
        })

        from ..rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.add_file(
            str(file_path),
            metadata={"doc_id": doc_id, "source": file_path.name, "tag": tag},
        )

        # 统计块数
        stats = pipeline.get_stats()

        # 更新状态为已完成
        DocumentStore.update(doc_id, {
            "status": "indexed",
            "chunk_count": stats.get("document_count", 0),
            "updated_at": datetime.now().isoformat()
        })
        print(f"文档 {doc_id} 索引成功")

        # 索引内容已变化，主动失效语义缓存
        try:
            from ..cache.query_cache import get_shared_cache
            get_shared_cache().invalidate()
        except Exception as e:
            print(f"索引后失效缓存失败: {e}")

        # 同步 Skill 参考文件（新标签自动生成）
        try:
            from ..rag.vector_store import create_vector_store
            from ..agent.skill_manager import SkillManager
            sm = SkillManager(create_vector_store())
            sm.sync_references()
        except Exception:
            pass

    except Exception as e:
        import traceback
        # 更新状态为错误
        DocumentStore.update(doc_id, {
            "status": "error",
            "updated_at": datetime.now().isoformat()
        })
        print(f"文档 {doc_id} 索引失败: {e}")
        print(traceback.format_exc())


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    admin: dict = Depends(verify_admin_token)
):
    """
    获取文档列表（支持分页和筛选）
    """
    items = DocumentStore.list_all(status=status, search=search)

    # 分页
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_items = items[start:end]

    return DocumentListResponse(
        total=total,
        items=[DocumentInfo(**item) for item in paginated_items],
        page=page,
        page_size=page_size
    )


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    admin: dict = Depends(verify_admin_token)
):
    """获取文档详情"""
    doc = DocumentStore.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    return DocumentInfo(**doc)


@router.get("/documents/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    admin: dict = Depends(verify_admin_token)
):
    """获取文档内容（预览）"""
    doc = DocumentStore.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    file_path = DOCS_STORAGE_PATH / doc["filename"]

    try:
        # 只支持文本文件预览
        if doc["file_type"] not in ['.txt', '.md']:
            return {"content": "[该文件类型不支持预览]", "type": doc["file_type"]}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {"content": content, "type": doc["file_type"]}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"读取文件失败: {str(e)}"
        )


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    admin: dict = Depends(verify_admin_token)
):
    """获取文档分块详情"""
    doc = DocumentStore.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    file_path = DOCS_STORAGE_PATH / doc["filename"]

    try:
        # 读取并分块
        if doc["file_type"] not in ['.txt', '.md']:
            return {"chunks": [], "message": "该文件类型不支持分块预览"}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunker = Chunker()
        chunks = chunker.split_text(content, metadata={"source": doc["original_name"]})

        return {
            "chunks": [
                {
                    "id": chunk["chunk_id"],
                    "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "length": len(chunk["content"])
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分块处理失败: {str(e)}"
        )


@router.post("/documents/{doc_id}/reindex")
async def reindex_document(
    doc_id: str,
    admin: dict = Depends(verify_admin_token)
):
    """重新索引文档"""
    doc = DocumentStore.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    file_path = DOCS_STORAGE_PATH / doc["filename"]
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档文件不存在"
        )

    # 更新状态为处理中
    DocumentStore.update(doc_id, {
        "status": "processing",
        "updated_at": datetime.now().isoformat()
    })

    # 触发后台索引任务
    import asyncio
    asyncio.create_task(_process_document_indexing(doc_id, file_path, doc["file_type"]))

    # 重索引前先清理该文档旧的向量分块，避免重复入库
    try:
        from ..rag.rag_pipeline import RAGPipeline
        removed = RAGPipeline().delete_documents_by_source(file_path.name)
        if removed:
            print(f"文档 {doc_id} 重索引：已清理旧向量分块 {removed} 条")
    except Exception as e:
        print(f"重索引清理旧向量失败: {e}")

    return {"message": "重新索引任务已启动", "doc_id": doc_id}


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    admin: dict = Depends(verify_admin_token)
):
    """删除文档"""
    doc = DocumentStore.get(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    file_path = DOCS_STORAGE_PATH / doc["filename"]

    try:
        # 删除文件
        if file_path.exists():
            os.remove(file_path)

        # 删除向量库中的分块（避免文件已删但检索仍能召回）
        try:
            from ..rag.rag_pipeline import RAGPipeline
            removed = RAGPipeline().delete_documents_by_source(file_path.name)
            if removed:
                print(f"文档 {doc_id} 删除：已清理向量分块 {removed} 条")
        except Exception as e:
            print(f"删除向量数据失败: {e}")

        # 主动失效语义缓存：文档内容已变化，旧答案不应继续命中
        try:
            from ..cache.query_cache import get_shared_cache
            get_shared_cache().invalidate()
        except Exception as e:
            print(f"失效缓存失败: {e}")

        # 删除数据库记录
        DocumentStore.delete(doc_id)

        return {"message": "文档已删除", "doc_id": doc_id}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除失败: {str(e)}"
        )


# ============== 配置管理API ==============

@router.get("/config/rag", response_model=RAGConfig)
async def get_rag_config(admin: dict = Depends(verify_admin_token)):
    """获取RAG配置"""
    import yaml

    config_path = Path("config/rag_config.yaml")
    if not config_path.exists():
        return RAGConfig()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return RAGConfig(**config)
    except Exception:
        return RAGConfig()


@router.put("/config/rag")
async def update_rag_config(
    config: RAGConfig,
    admin: dict = Depends(verify_admin_token)
):
    """更新RAG配置（含回归门禁：更新前保存基线）"""
    import yaml

    global _eval_baseline
    config_path = Path("config/rag_config.yaml")

    try:
        # 更新前保存评测基线
        from ..eval.realtime import load as _load_rt
        _eval_baseline = {
            "baseline_scores": _load_rt()[-50:],
            "config_before": yaml.safe_load(config_path.open(encoding="utf-8")) if config_path.exists() else {},
            "updated_at": datetime.now().isoformat(),
        }

        existing = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
        merged = _deep_merge(existing, config.dict(exclude_none=True))
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(merged, f, allow_unicode=True, default_flow_style=False)

        return {"message": "RAG配置已更新，已保存回归基线"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置保存失败: {str(e)}"
        )


@router.get("/config/memory", response_model=MemoryConfig)
async def get_memory_config(admin: dict = Depends(verify_admin_token)):
    """获取记忆配置"""
    import yaml

    config_path = Path("config/memory_config.yaml")
    if not config_path.exists():
        return MemoryConfig()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return MemoryConfig(**config)
    except Exception:
        return MemoryConfig()


@router.put("/config/memory")
async def update_memory_config(
    config: MemoryConfig,
    admin: dict = Depends(verify_admin_token)
):
    """更新记忆配置"""
    import yaml

    config_path = Path("config/memory_config.yaml")

    try:
        existing = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
        merged = _deep_merge(existing, config.dict(exclude_none=True))
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(merged, f, allow_unicode=True, default_flow_style=False)

        return {"message": "记忆配置已更新"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置保存失败: {str(e)}"
        )


# ============== 统计API ==============

# ============== 记忆管理API（人工纠错入口） ==============

@router.get("/memory/{user_id}")
async def get_user_memory_detail(
    user_id: str,
    admin: dict = Depends(verify_admin_token),
):
    """查看用户记忆：当前画像 + 最新摘要 + 人可读审计日志"""
    from ..memory.long_memory import LongMemory, PROFILES_AUDIT_DIR

    mem = LongMemory()
    profile = mem.get_user_memory(user_id)
    summary = mem.get_running_summary(user_id)
    audit_path = Path(PROFILES_AUDIT_DIR) / f"{user_id}.md"
    audit_md = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    from ..memory import pref_store
    return {
        "user_id": user_id,
        "profile": profile,
        "running_summary": summary,
        "audit_md": audit_md,
        "prefs_timeline": pref_store.get_timeline(user_id),
        "prefs_active": pref_store.get_active(user_id),
    }


@router.delete("/memory/{user_id}/facts")
async def delete_user_memory_fact(
    user_id: str,
    body: DeleteMemoryFactRequest,
    admin: dict = Depends(verify_admin_token),
):
    """删除用户画像中的一条偏好（人工纠错）"""
    from ..memory.long_memory import LongMemory

    ok = LongMemory().delete_profile_fact(user_id, body.text)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到该偏好条目或删除失败",
        )
    return {"message": f"已删除偏好条目：{body.text}", "user_id": user_id}


@router.delete("/memory/{user_id}")
async def clear_user_memory_api(
    user_id: str,
    admin: dict = Depends(verify_admin_token),
):
    """清空用户记忆（话题计数 + 画像 + 摘要；审计日志保留供追溯）"""
    from ..memory.long_memory import LongMemory

    ok = LongMemory().clear_user_memory(user_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清空用户记忆失败",
        )
    return {"message": f"用户 {user_id} 的记忆已清空", "user_id": user_id}

@router.get("/stats/overview", response_model=OverviewStats)
async def get_overview_stats(admin: dict = Depends(verify_admin_token)):
    """获取概览统计"""

    # 计算今日对话数
    today = datetime.now().date().isoformat()
    today_conversations = len([
        log for log in _conversation_logs
        if log["timestamp"].startswith(today)
    ])

    # 计算7天活跃用户数
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    active_users_7d = len(set([
        log["user_id"] for log in _conversation_logs
        if log["timestamp"] >= week_ago
    ]))

    # 查询类型分布（从日志统计）
    query_types = {}
    for log in _conversation_logs:
        intent = log.get("intent", "unknown")
        query_types[intent] = query_types.get(intent, 0) + 1

    # 从数据库获取文档数量
    total_documents = DocumentStore.count_by_status()

    return OverviewStats(
        total_conversations=len(_conversation_logs),
        total_users=len(set(log["user_id"] for log in _conversation_logs)),
        total_documents=total_documents,
        today_conversations=today_conversations,
        active_users_7d=active_users_7d,
        query_types=query_types
    )


@router.get("/stats/conversations")
async def get_conversation_stats(
    days: int = Query(7, ge=1, le=90),
    admin: dict = Depends(verify_admin_token)
):
    """获取对话统计（按日期）"""

    # 生成日期范围
    end_date = datetime.now().date()
    dates = [(end_date - timedelta(days=i)).isoformat() for i in range(days)]
    dates.reverse()

    # 统计每日对话数
    stats = []
    for date in dates:
        count = len([
            log for log in _conversation_logs
            if log["timestamp"].startswith(date)
        ])
        stats.append({
            "date": date,
            "count": count
        })

    return {"stats": stats}


@router.get("/stats/queries")
async def get_query_stats(admin: dict = Depends(verify_admin_token)):
    """获取查询类型分布"""

    query_types = {}
    for log in _conversation_logs:
        intent = log.get("intent", "unknown")
        query_types[intent] = query_types.get(intent, 0) + 1

    # 转换为图表数据格式
    chart_data = [
        {"type": k, "count": v}
        for k, v in sorted(query_types.items(), key=lambda x: x[1], reverse=True)
    ]

    return {"distribution": chart_data}


@router.get("/stats/top-questions")
async def get_top_questions(
    limit: int = Query(10, ge=1, le=50),
    admin: dict = Depends(verify_admin_token)
):
    """获取热门问题"""

    from collections import Counter

    questions = [log["message"] for log in _conversation_logs if "message" in log]
    top_questions = Counter(questions).most_common(limit)

    return {
        "top_questions": [
            {"question": q, "count": c}
            for q, c in top_questions
        ]
    }


# ============== 日志API ==============

@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    admin: dict = Depends(verify_admin_token)
):
    """获取系统日志（简化版）"""

    # 返回最近的对话日志
    logs = _conversation_logs[-limit:]

    return {"logs": logs}


# ============== 工具函数 ==============

def log_conversation(user_id: str, message: str, intent: str = "unknown", answer: str = ""):
    """记录对话日志（供chat_api调用）"""
    _conversation_logs.append({
        "user_id": user_id,
        "message": message,
        "intent": intent,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })

    # 限制日志数量
    if len(_conversation_logs) > 10000:
        _conversation_logs[:] = _conversation_logs[-5000:]


# ============== 标签 API ==============


@router.get("/tags")
async def get_available_tags(admin: dict = Depends(verify_admin_token)):
    """获取当前知识库中所有已使用的标签"""
    from ..rag.vector_store import create_vector_store
    store = create_vector_store()
    return {"tags": store.get_available_tags()}


# ============== 评测 API ==============

_eval_runner: Optional["EvalRunner"] = None


def _get_eval_runner():
    global _eval_runner
    if _eval_runner is None:
        from ..eval.runner import EvalRunner
        _eval_runner = EvalRunner()
    return _eval_runner


@router.post("/eval/run")
async def run_evaluation(
    admin: dict = Depends(verify_admin_token),
    sample_size: int = Query(50, ge=1, le=500),
):
    """对最近 N 条对话运行 LLM-as-Judge 评测"""
    conversations = _conversation_logs[-sample_size:]
    runner = _get_eval_runner()
    report = await runner.run(conversations)
    return {
        "message": f"评测完成，共 {report.total_samples} 条",
        "report": report.model_dump() if hasattr(report, 'model_dump') else report.dict(),
    }


@router.get("/eval/report")
async def get_eval_report(admin: dict = Depends(verify_admin_token)):
    """获取最新评测报告"""
    runner = _get_eval_runner()
    report = runner.get_latest_report()
    if not report:
        return {"message": "暂无评测数据"}
    return report.model_dump() if hasattr(report, 'model_dump') else report.dict()


@router.get("/eval/alerts")
async def get_eval_alerts(admin: dict = Depends(verify_admin_token)):
    """获取低分告警列表"""
    from ..eval.alerts import load_alerts
    return {"alerts": load_alerts(), "total": len(load_alerts())}


@router.get("/eval/summary")
async def get_eval_summary(admin: dict = Depends(verify_admin_token)):
    """获取评测摘要（平均分 + 告警统计 + 回归基线对比）"""
    from ..eval.alerts import get_summary
    from ..eval.realtime import load as load_realtime
    scores = load_realtime()
    total = len(scores)
    if total == 0:
        return {"total_scores": 0, "alerts": get_summary()}
    recent = scores[-50:]
    avg_r = sum(s["relevance"] for s in recent) / len(recent)
    avg_c = sum(s["completeness"] for s in recent) / len(recent)
    avg_u = sum(s["usefulness"] for s in recent) / len(recent)
    faith_scores = [
        s["faithfulness"] for s in recent
        if s.get("faithfulness") is not None
    ]
    avg_f = round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else None
    return {
        "total_scores": total,
        "recent_samples": len(recent),
        "avg_relevance": round(avg_r, 2),
        "avg_completeness": round(avg_c, 2),
        "avg_usefulness": round(avg_u, 2),
        "avg_faithfulness": avg_f,
        "alerts": get_summary(),
    }


# 评测基线（用于回归对比）
_eval_baseline: dict = {}


@router.get("/eval/baseline")
async def get_eval_baseline(admin: dict = Depends(verify_admin_token)):
    return _eval_baseline or {"message": "暂无基线"}


@router.get("/eval/realtime")
async def get_realtime_eval(admin: dict = Depends(verify_admin_token)):
    from ..eval.realtime import load
    scores = load()
    return {"scores": scores, "total": len(scores)}


@router.get("/eval/missed")
async def get_missed_queries(
    admin: dict = Depends(verify_admin_token),
    top_n: int = Query(20, ge=1, le=100),
):
    """检索失败 query 聚合：高频未命中即知识库盲区，反哺知识库迭代"""
    from ..eval.missed_queries import get_missed_summary, load_missed
    return {
        "summary": get_missed_summary(top_n),
        "total_missed": len(load_missed()),
    }


@router.get("/eval/history")
async def get_eval_history(admin: dict = Depends(verify_admin_token)):
    """获取评测历史（回归对比）"""
    runner = _get_eval_runner()
    return {"history": runner.get_history()}
