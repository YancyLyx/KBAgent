# 系统架构

## 整体概述

KBAgent 是一个基于 ReAct 架构的智能知识库问答系统。通过 Function Calling 让 LLM 自主决策工具调用，实现知识库检索与文件操作的智能编排。

## 架构层次

```
┌──────────────────────────────────────────────────┐
│                    用户界面                        │
│         用户门户 (React) · 管理门户 (React)       │
├──────────────────────────────────────────────────┤
│                    API 层                         │
│     FastAPI · Chat API · Admin API · Eval API     │
├──────────────────────────────────────────────────┤
│                   Agent 层                        │
│  AgentManager (ReAct) · ToolRouter · SkillManager │
├──────────────────────────────────────────────────┤
│                   RAG 管线                        │
│  Chunker · Retriever · Reranker · VectorStore      │
├──────────────────────────────────────────────────┤
│                存储与检索                          │
│  ChromaDB · BM25 · Sentence-BERT · SQLite · JSON   │
├──────────────────────────────────────────────────┤
│              横切能力（可观测性）                  │
│  QueryCache (语义缓存) · Metrics (埋点) · Eval    │
└──────────────────────────────────────────────────┘
```

## 请求流程（ReAct 循环）

```
用户提问 → Chat API → AgentManager
  │
  ├── 循环（最多 5 轮）：
  │   1. LLM 推理：当前 context + 工具列表 → 决策
  │   2. 如果 LLM 调用工具 → 执行 → 观察结果 → 追加到 context → 返回步骤 1
  │   3. 如果 LLM 直接回答 → 跳出循环 → 返回回答
  │
  │   例：用户问"文献和项目中都提到的方案是什么"
  │   第 1 轮：LLM 调用 search_knowledge_base("方案", tag="文献") → 结果
  │   第 2 轮：LLM 调用 search_knowledge_base("方案", tag="项目") → 结果
  │   第 3 轮：LLM 基于两次结果生成对比回答 → 跳出
  │
└── 回复后：实时评测 → 存储分数 → 检查告警

区别于硬编码路由，LLM 自主决策每轮要调用的工具，无需预设流程图。
```

## ReAct 循环的异常处理策略

生产环境 LLM 输出不可控，循环内的每一环都做了防御：

| 环节 | 策略 | 说明 |
|------|------|------|
| 工具名校验 | 白名单前置校验 | LLM 可能输出幻觉工具名，先比对 schema 内工具列表，未知工具回填错误让 LLM 修正，不进入执行 |
| 参数解析 | `_safe_parse_tool_args` 三级容错 | 标准 JSON → 提取花括号子串 → ast 宽松解析；失败时回填错误让 LLM 修正，不中断循环 |
| 必填参数校验 | 缺失即回填纠错信息 | `read_category_info` 缺 tag、`search_knowledge_base` 缺 query/tag 时直接提示补充，避免生成 `None.md` 之类的脏数据 |
| 工具执行 | try/except 异常隔离 | 单个工具抛错只回填错误信息，不影响整轮对话与其他工具 |
| 死循环检测 | 工具+参数指纹计数 | 同一指纹累计超过 2 次 → 提示 LLM 停止重复调用并直接回答 |
| 上下文保护 | 工具结果截断（3000 字符） | 防止超长工具返回撑爆上下文窗口 |
| 资源复用 | ToolRouter 注入共享 RAGPipeline | 避免每次工具检索重复初始化 embedding/Chroma |
| LLM 稳定性 | 超时 + SDK 内置重试 | `LLM_TIMEOUT`/`LLM_MAX_RETRIES` 可配；网络抖动不再直接导致整轮失败，失败时响应带 `success=false` + `error` 字段，不再用 200 掩盖 |

#### 参数解析的三级容错（示例）

模型返回的 `arguments` 是字符串，本应是合法 JSON，但实际经常不是。三级容错是
**依次尝试的降级链**，成功即停：

```text
第 1 级：json.loads(原始字符串)
  处理"完全合法"的 JSON：{"query": "报销流程", "tag": "财务"}

第 2 级：截取第一个 { 到最后一个 } 的部分再 json.loads
  处理"JSON 前后混了说明文字"：
  "我准备查询财务库。{"query": "报销流程", "tag": "财务"} 请稍等"
  → 整体不是合法 JSON，截取花括号部分后解析成功

第 3 级：ast.literal_eval(原始字符串)
  处理"Python 风格字面量"（单引号、True/None 不是合法 JSON）：
  {'query': '报销流程', 'tag': None}
  → literal_eval 只解析字面量、不执行代码，是安全的
```

三级都失败 → 返回空 dict + 失败标记，AgentManager 把错误回填给 LLM 让其修正，
不中断循环。

#### 必填参数校验（示例）

`read_category_info` 缺 tag、`search_knowledge_base` 缺 query/tag 时，直接回填：

```text
LLM 调用：read_category_info(tag="")     ← 漏传/传空
系统回填：参数缺失：read_category_info 需要必填参数 tag
          （知识库标签，如「文献」）。请补充后重新调用。
LLM 下一轮：read_category_info(tag="财务")   ← 自己修正
```

如果不校验，空 tag 会拼出 `references/None.md` 这样的脏文件。

## 记忆分层存储设计

记忆系统按「状态」与「回忆」分层存储，各司其职：

| 层 | 存储 | 内容 | 为什么 |
|----|------|------|--------|
| 短期记忆 | 内存 deque（maxlen=N 轮，当前配置 10） | 当前会话最近对话 | 会话内高频读写，无需持久化（deque=双端队列，这里用作固定长度滑动窗口，满了自动挤掉最旧的） |
| 长期状态 | SQLite | 演进式摘要（最新版）、用户画像、话题计数 | 按 user_id 精确读取 + 覆盖更新，SQLite 保证原子性与事务；本质是「当前状态」而非语料 |
| 历史回忆 | ChromaDB 向量库 | 对话原文 + 演进摘要快照 | 需要跨会话语义检索（`semantic_recall`），向量库按相关性召回 |

**为什么演进式摘要/用户画像不用文档存储**：
摘要是「覆盖式状态」（读多写少、按用户精确查询、需要原子更新），SQLite 是正确选择；
真正需要「文档/向量」的是历史对话的语义回忆，这部分已由 ChromaDB 承担。
每次演进式摘要更新时，同时写入向量库一份快照，使历史摘要也可被语义召回。

**演进式摘要的触发时机**：摘要触发点对齐短期记忆窗口边界（第 N、2N、3N 轮，
N=max_history_length），用「旧摘要 + 窗口内 N 轮」合并。早期实现按「每 10 轮」
触发，但 `ShortMemory.get_turn_count()` 返回的是窗口内剩余轮数（恒 ≤N），
触发条件要么永不成立、要么窗口填满后每轮都触发——这是上线前很难发现、
面试官一定会追问的经典缺陷。现在由 AgentManager 维护会话累计轮数。

## 系统设计（生产落地）

### 语义缓存（QueryCache）
- **两级命中**：精确匹配（O(1)）+ 语义匹配（同 namespace 内 query embedding 余弦相似度 ≥ 阈值）
- **namespace 隔离**：缓存键 = (user_id, query)，不同用户问同一问题不串答案；语义检索只在同 namespace 内进行
- **TTL + 主动失效**：默认 1 小时过期（TTL=Time-To-Live，到期自动失效）；文档删除/重索引/上传成功后调用 `invalidate()` 清空，避免知识库更新后旧答案继续命中
- **进程级共享**：全局单例，不同会话复用同一份缓存（早期是每个 Agent 实例一个空缓存，命中率趋近于 0）
- **容量受限**：LRU 淘汰（默认 256 条，Least Recently Used：满时淘汰最久未访问的条目），线程安全
- **命中路径一致性**：缓存命中也会写入短期记忆，用户追问"你刚才说啥"时上下文不断裂
- **写入质量门禁**：报错、低分兜底（fallback）的答案不写缓存
- **可替换**：内存态实现，生产可平滑换 Redis（key 前缀 + ZSET 过期扫描，保持 get/put/invalidate 接口）

**为什么用内存缓存而不是 Redis**：当前是单机部署、单进程，内存 LRU 完全够用且
零运维成本；真正决定缓存可用性的是「键设计（namespace）+ 失效语义（TTL/invalidate）」
这两件事，与存储介质无关。接口层面已经把 get/put/invalidate 抽象出来，
换 Redis 时不需要动 Agent 逻辑。什么时候必须换：多实例部署（每实例一份缓存会
重复计算且失效不互通）、缓存容量超过单机内存、需要持久化防重启丢失。

**缓存键长什么样（示例）**：

```text
用户 u1 问 "报销流程是什么" → 键 = ("user_8969a39e1546", "报销流程是什么")
用户 u2 问同样的问题       → 键 = ("user_xxxx", "报销流程是什么")，namespace 不同，不会命中
同一个用户换个说法问       → 语义命中（仅同 namespace 内余弦 ≥ 0.92 才复用）
```

缓存内容：key 是 (namespace, query)，value 是 **LLM 的最终回复文本** +
query 的归一化向量 + 写入时间/TTL（详细格式见 interview_prep.md §3.4⑤）。

### 监控埋点（Metrics）
- 每次请求记录：耗时、是否缓存命中、错误、输入/输出 token 估算
- 聚合指标：P50/P95 延迟、缓存命中率、错误率、token 消耗
- 本地 JSONL 追加写入（JSON Lines：一行一个 JSON 对象，适合追加日志），生产可替换 Prometheus/SkyWalking

### 瓶颈与扩展方向
- LLM 调用已异步非阻塞（AsyncOpenAI），embedding/rerank 已线程池化（毫秒级）；
  高并发下瓶颈转移到 LLM API 吞吐、每会话模型实例内存与 SQLite 写入
- 已支持：SSE 流式输出（`/chat/stream`，用户先看到首个 token，前端逐字渲染）
- 扩展方向：精排降级开关
  （评测显示精排对回复质量提升有限但延迟 15 倍，线上可默认关闭）、
  模型自托管（vLLM，OpenAI 兼容协议无缝切换）

### 会话生命周期（内存管理）
- 每个会话持有独立 AgentManager（含 embedding + rerank 模型），内存开销大，**必须回收**
- 空闲超时：`SESSION_IDLE_TTL_SECONDS`（默认 30 分钟）未活跃即从内存移除
- 容量上限：`MAX_ACTIVE_SESSIONS`（默认 200），超量按最后活跃时间淘汰最旧会话
- 会话归属校验：客户端传 session_id 时校验 user_id 一致，防止串用户记忆

> 注意：**会话上限（200）不是线程池大小**。会话上限管"内存里同时存在的
> Agent 实例数"（每个持有模型和用户状态，超限淘汰最旧会话）；线程池
> （`asyncio.to_thread` 默认约 min(32, CPU核数+4)）管"同时能并行的阻塞任务数"。
> 前者是状态/内存维度，后者是并发执行维度，两者独立。

### 身份与鉴权（阶段 0：匿名身份 token）
- **现状**：用户端从"客户端自由填写 user_id"升级为**服务端签发 HMAC 签名 token**
  （`Authorization: Bearer {user_id}.{hmac}`），客户端不知道签名密钥，
  无法伪造任意 user_id
- **管理端**：`POST /api/admin/login`（用户名+密码，密码支持 pbkdf2 哈希）
  签发带过期的 admin token（`admin.{exp}.{hmac}`）；所有管理接口
  `verify_admin_token` 真实验签，未带/无效 token 返回 401
- **签发时机**：`/chat` 首次无有效 token 时签发并随响应返回，前端存入
  localStorage 后后续请求自动携带；会话/历史/列表端点也按 token 解析身份并
  做归属校验（未认证看不到任何会话）
- **边界**：这是"登录鉴权（阶段 1）"之前的过渡：匿名身份不等于账号体系，
  换设备/清缓存会丢失身份；管理端目前是单管理员账号（环境变量配置），
  生产必须设置 `ANON_TOKEN_SECRET` / `ADMIN_TOKEN_SECRET`，
  多管理员/角色权限是下一步

> 概念详解（token 是什么、密码为什么存哈希、HMAC 签名怎么防伪造、
> 两套身份体系的完整流程示例）见 [auth_guide.md](auth_guide.md)。

**token 长什么样（示例）**：

```text
首次 /chat 响应：
  { ..., "token": "user_8969a39e1546.b97997ea2ee6a85f8e1a78d22cb183e4...", ... }
  （前半段是服务端签发的 user_id，后半段是 HMAC-SHA256 签名）

客户端后续请求：
  Authorization: Bearer user_8969a39e1546.b97997ea2ee6a85f8e1a78d22cb183e4...

伪造尝试：把前半段改成 user_attacker → 验签失败 → 服务端当作新访客重新签发
```

### 异步执行模型
- **全链路异步**：`/chat` 为 async 端点，`agent.chat()` 是 `async def`；
  LLM 网络调用使用 `AsyncOpenAI`（真正的非阻塞，网络等待不占线程），
  ReAct 循环、实时评测、低分兜底全部 `await`
- **CPU 计算线程池化**：embedding、rerank、Chroma 检索、工具执行是同步
  CPU/本地 IO（无法非阻塞），统一包在 `asyncio.to_thread` 里，占用毫秒级
- **演进路径**：早期实现直接在 async 函数里同步调用 LLM（10 并发就能卡死
  事件循环）→ 先用 `asyncio.to_thread` 兜住 → 升级为 `AsyncOpenAI` 全链路异步 →
  SSE 流式输出（`/chat/stream` 逐 token 推送，前端逐字渲染）

### 工具安全边界
- 文件类工具（read/write/create/list）路径解析限制在项目根目录内，越界路径直接报错
- 敏感文件拦截：`.env*`（可能含 API Key）、`.git*`（可能含凭据）禁止读写
- 早期 `_resolve_path` 越界时静默返回项目根目录，工具读错文件且无任何报错，属于安全盲区

## 核心设计原则

区别于简单的 RAG demo，本系统的核心差异化在于 ReAct 架构：
LLM 自主做出工具决策，而非硬编码路由。
这使得系统可扩展：新增工具无需改变路由逻辑。

**为什么是 ReAct/Agent 而不是 Workflow（硬编码流程）**：本场景里用户问题类型
开放——可能只查一个知识库，也可能需要跨两个库对比、还可能不需要检索直接聊天。
Workflow 需要预先枚举所有分支，新增一个知识库分类就要改流程图；Agent 把「下一
步调什么工具」交给 LLM 决策，新增分类只需要注册工具 + 更新枚举，路由逻辑零改动。
代价是 LLM 输出不可控，所以循环外必须包一层 Harness（工具白名单、参数校验、
死循环检测、结果截断），用工程手段把不确定性关进笼子里。
