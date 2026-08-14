# -*- coding: utf-8 -*-
"""Chat API 模块

提供 FastAPI 接口，对外暴露智能客服对话能力。
"""

import asyncio
import json
import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agent.agent_manager import AgentManager
from .admin_api import router as admin_router, log_conversation
from ..db.session_store import SessionStore
from .anon_auth import (
    issue_anonymous_user_id,
    issue_token,
    verify_token,
)


# 创建 FastAPI 应用
app = FastAPI(
    title="KBAgent - Smart Knowledge Base Q&A System",
    description="企业级智能客服 Agent API",
    version="1.0.0"
)

# 注册管理后台路由
app.include_router(admin_router)

# 添加 CORS 中间件（生产环境通过 CORS_ORIGINS 指定白名单；
# 通配符模式下不允许携带 credentials，避免浏览器直接拒绝）
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储活跃的 Agent 会话
_active_sessions: dict = {}

# 会话生命周期：每个会话持有 RAGPipeline（embedding + rerank 模型），
# 不做清理会直接内存泄漏。空闲超时 + 上限双保险。
SESSION_IDLE_TTL_SECONDS = float(os.getenv("SESSION_IDLE_TTL_SECONDS", "1800"))
MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", "200"))


# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: Optional[str] = Field(
        None,
        description="已废弃：身份改由服务端签发 token 决定（Authorization: Bearer），此字段不再作为身份依据",
    )
    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID（用于保持上下文）")
    context: Optional[dict] = Field(None, description="额外上下文信息")


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool = Field(..., description="是否成功")
    reply: str = Field(..., description="助手回复")
    intent: Optional[str] = Field(None, description="识别到的意图")
    tool_used: Optional[str] = Field(None, description="使用的工具")
    context_count: Optional[int] = Field(None, description="检索到的上下文数量")
    session_id: str = Field(..., description="会话ID")
    token: Optional[str] = Field(None, description="服务端签发的匿名身份 token（首次调用返回，客户端需保存并在后续请求的 Authorization 头携带）")
    timestamp: str = Field(..., description="响应时间")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    user_id: str
    history_length: int
    created_at: str
    last_active: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str


# ==================== API 端点 ====================

def _resolve_user_id(authorization: Optional[str], issue_if_missing: bool = False):
    """身份解析：优先 Authorization Bearer token（验签），无有效 token 则签发新的。

    返回 (user_id, new_token)：new_token 为 None 表示用的是已有身份；
    issue_if_missing=True 时无有效 token 会签发新身份（仅 /chat 使用，
    因为只有它能把新 token 返回给客户端）；其他端点传 False，
    未认证时返回 (None, None)，避免每次 GET 都漂移出新身份。
    """
    if authorization and authorization.lower().startswith("bearer "):
        uid = verify_token(authorization[7:].strip())
        if uid:
            return uid, None
    if not issue_if_missing:
        return None, None
    new_uid = issue_anonymous_user_id()
    return new_uid, issue_token(new_uid)


def _resolve_session(request: "ChatRequest", authorization: Optional[str]):
    """会话解析公共逻辑：身份签发/校验 + 会话创建/归属校验。

    /chat 与 /chat/stream 共用，保证两条路径的会话行为一致。
    """
    user_id, new_token = _resolve_user_id(authorization, issue_if_missing=True)
    _evict_stale_sessions()

    session_id = request.session_id or f"sess_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if session_id in _active_sessions:
        session_info = _active_sessions[session_id]
        if session_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="该会话不属于当前用户")
        agent = session_info["agent"]
    else:
        agent = AgentManager(
            user_id=user_id,
            enable_memory=True,
            enable_rag=True,
            enable_tools=True,
            enable_eval=True,
        )
        _active_sessions[session_id] = {
            "agent": agent,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
        }
        SessionStore.create_session(session_id, user_id)

    _active_sessions[session_id]["last_active"] = datetime.now().isoformat()
    return session_id, user_id, agent, new_token

def _session_idle_seconds(info: dict, now: datetime) -> float:
    """计算会话空闲秒数"""
    try:
        last = datetime.fromisoformat(info.get("last_active", ""))
        return (now - last).total_seconds()
    except ValueError:
        return SESSION_IDLE_TTL_SECONDS + 1  # 解析失败按过期处理


def _evict_stale_sessions(now: Optional[datetime] = None) -> int:
    """清理空闲超时/超量会话，防止内存无限增长。返回清理数量。"""
    now = now or datetime.now()
    stale = [
        sid for sid, info in _active_sessions.items()
        if _session_idle_seconds(info, now) > SESSION_IDLE_TTL_SECONDS
    ]
    for sid in stale:
        _active_sessions.pop(sid, None)

    if len(_active_sessions) > MAX_ACTIVE_SESSIONS:
        ordered = sorted(
            _active_sessions.items(),
            key=lambda kv: kv[1].get("last_active", ""),
        )
        overflow = len(ordered) - MAX_ACTIVE_SESSIONS
        for sid, _ in ordered[:overflow]:
            _active_sessions.pop(sid, None)
    return len(stale)

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 服务信息"""
    return HealthResponse(
        status="running",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    """
    主对话接口

    处理用户消息，根据意图路由到 RAG/Tool/LLM，返回答复。
    """
    try:
        # 会话解析（身份签发 + 归属校验）与 /chat/stream 共用
        session_id, user_id, agent, new_token = _resolve_session(request, authorization)

        # 持久化用户消息
        SessionStore.add_message(session_id, "user", request.message)

        # Agent 已整体异步化：LLM 用 AsyncOpenAI（网络等待不占线程），
        # embedding/rerank 在内部丢线程池，这里直接 await 即可
        result = await agent.chat(
            request.message,
            request.context or {},
        )

        # 持久化助手回复
        reply = result.get("answer", "")
        SessionStore.add_message(session_id, "assistant", reply)

        # 记录对话日志（用于统计 + 评测）
        log_conversation(
            user_id=user_id,
            message=request.message,
            intent=result.get("intent", "unknown"),
            answer=reply,
        )

        # 构建响应
        return ChatResponse(
            success=result.get("success", True),
            reply=reply,
            intent=result.get("intent"),
            tool_used=result.get("tool_used"),
            context_count=result.get("context_count"),
            session_id=session_id,
            token=new_token,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    """SSE 流式对话接口。

    与 /chat 同一条 Agent 链路，区别是"最终回答"阶段逐 token 推送：
    data: {"type":"token","content":"..."}
    data: {"type":"done","reply":"...","session_id":"...","token":"..."}
    data: {"type":"error","message":"..."}
    工具调用/检索阶段仍为非流式（没有可输出的文本），前端保持 loading 即可。
    """
    try:
        session_id, user_id, agent, new_token = _resolve_session(request, authorization)
        SessionStore.add_message(session_id, "user", request.message)
        queue: "asyncio.Queue" = asyncio.Queue()

        async def on_token(token: str) -> None:
            await queue.put(("token", token))

        async def run_agent() -> None:
            """后台跑完整 Agent 链路（含记忆/评测/缓存等后置），结束后推 done/error"""
            try:
                result = await agent.chat(
                    request.message,
                    request.context or {},
                    stream_callback=on_token,
                )
                reply = result.get("answer", "")
                SessionStore.add_message(session_id, "assistant", reply)
                log_conversation(
                    user_id=user_id,
                    message=request.message,
                    intent=result.get("intent", "unknown"),
                    answer=reply,
                )
                await queue.put(("done", result))
            except Exception as e:
                await queue.put(("error", str(e)))

        async def sse_gen():
            task = asyncio.create_task(run_agent())
            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': payload}, ensure_ascii=False)}\n\n"
                elif kind == "done":
                    r = payload
                    done_payload = {
                        "type": "done",
                        "success": r.get("success", True),
                        "reply": r.get("answer", ""),
                        "session_id": session_id,
                        "token": new_token,
                        "tool_used": r.get("tool_used"),
                        "context_count": r.get("context_count"),
                    }
                    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': payload}, ensure_ascii=False)}\n\n"
                    break
            await task  # 确保 run_agent 的异常不被静默吞掉

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


@app.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    获取会话历史

    Args:
        session_id: 会话ID

    Returns:
        对话历史列表
    """
    # 会话归属校验：只能读自己的会话
    user_id, _ = _resolve_user_id(authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="未认证：请先通过 /chat 获取身份 token")
    session = SessionStore.get_session(session_id)
    if session and session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="该会话不属于当前用户")

    # 优先从数据库加载历史
    db_history = SessionStore.get_history(session_id)
    if db_history:
        # 转换为前端期望的格式
        formatted_history = []
        user_msg = None
        for msg in db_history:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and user_msg:
                formatted_history.append({
                    "user": user_msg,
                    "assistant": msg["content"]
                })
                user_msg = None

        return {
            "session_id": session_id,
            "history": formatted_history,
            "turn_count": len(formatted_history)
        }

    # 如果内存中有会话，也从内存获取（Agent可能有更完整的历史）
    if session_id in _active_sessions:
        agent = _active_sessions[session_id]["agent"]
        history = agent.get_conversation_history()
        return {
            "session_id": session_id,
            "history": history,
            "turn_count": len(history)
        }

    raise HTTPException(status_code=404, detail="会话不存在")


@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    清除会话（删除历史记录）

    Args:
        session_id: 会话ID

    Returns:
        操作结果
    """
    # 会话归属校验
    user_id, _ = _resolve_user_id(authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="未认证：请先通过 /chat 获取身份 token")
    session = SessionStore.get_session(session_id)
    if session and session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="该会话不属于当前用户")

    # 从数据库删除
    deleted = SessionStore.delete_session(session_id)

    # 如果内存中有，也从内存中移除
    if session_id in _active_sessions:
        agent = _active_sessions[session_id]["agent"]
        agent.clear_memory()
        del _active_sessions[session_id]

    if not deleted and session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {
        "success": True,
        "message": f"会话 {session_id} 已清除"
    }


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """
    列出活跃会话

    Args:
        user_id: 可选的用户ID过滤

    Returns:
        会话列表
    """
    # 身份优先取 token（防伪造）；query 参数 user_id 仅作无 token 时的兼容
    resolved_uid, _ = _resolve_user_id(authorization)
    if resolved_uid is None:
        return []  # 未认证：看不到任何会话（多用户隔离）
    user_id = resolved_uid  # 身份以 token 为准（防伪造）

    # 从数据库加载会话列表
    db_sessions = SessionStore.list_sessions(user_id)

    sessions = []
    for session in db_sessions:
        # 检查内存中是否有该会话
        if session["session_id"] in _active_sessions:
            info = _active_sessions[session["session_id"]]
            agent = info["agent"]
            history_length = len(agent.get_conversation_history())
        else:
            # 从数据库计算历史长度
            history = SessionStore.get_history(session["session_id"])
            # 每两条消息（user+assistant）算作一轮
            history_length = len([h for h in history if h["role"] == "user"])

        sessions.append(SessionInfo(
            session_id=session["session_id"],
            user_id=session["user_id"],
            history_length=history_length,
            created_at=session.get("created_at", ""),
            last_active=session.get("last_active", "")
        ))

    return sessions


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    total_sessions = len(_active_sessions)

    # 统计各用户会话数
    user_sessions = {}
    for info in _active_sessions.values():
        user_id = info["user_id"]
        user_sessions[user_id] = user_sessions.get(user_id, 0) + 1

    return {
        "total_active_sessions": total_sessions,
        "unique_users": len(user_sessions),
        "user_session_counts": user_sessions,
        "timestamp": datetime.now().isoformat()
    }


# ==================== 启动和清理 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("=" * 50)
    print("KBAgent - Smart Knowledge Base Q&A System 服务启动")
    print("=" * 50)
    print(f"文档地址: http://127.0.0.1:8000/docs")
    print(f"健康检查: http://127.0.0.1:8000/health")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    print("正在关闭 KBAgent - Smart Knowledge Base Q&A System 服务...")
    # 清理资源
    _active_sessions.clear()


# 如果是直接运行此文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
