# KBAgent - Smart Knowledge Base Q&A System

<p align="center">
  <strong>智能知识库问答系统，支持多知识库检索，是一款通用、简易、综合的RAG系统</strong>
</p>


<p align="center">
  <a href="#-核心功能">核心功能</a> •
  <a href="#-技术栈">技术栈</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-项目结构">项目结构</a> •
  <a href="doc/">详细文档</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18+-61dafb.svg" alt="React">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

基于 ReAct 架构的智能知识库问答系统。支持多格式文档（PDF/MD/TXT/DOCX/图片 VLM 描述）上传与管理，让 LLM 自主决策工具调用，实现知识库检索与文件操作的智能编排。

## 🎯 核心功能

| 功能 | 说明 |
|------|------|
| **多格式文档接入** | PDF 章节+表格、Markdown 标题分块、DOCX 段落+表格、图片 VLM 描述（deepseek-vision-exp） |
| **父子分块策略** | 子块精确匹配，父块完整上下文，兼顾精度与完整性 |
| **混合检索（RRF）** | 向量检索 + BM25 关键词，Reciprocal Rank Fusion 合并，CrossEncoder 精排 |
| **渐进式知识库路由** | LLM 三层渐进发现可用知识库，按需选择分类检索 |
| **ReAct Agent（queryLoop）** | 结构化退出 reason（completed/no_progress/max_turns/输出超长/上下文超长/缓存命中/异常），输出截断续写 + 被动压缩 + 六层工具防御 |
| **多层记忆系统** | 短期 deque + 长程向量语义召回 + 演进式摘要 + 用户画像（含时间线/冲突澄清） |
| **LLM-as-Judge** | 实时三维度评分，低分告警 + 自动兜底 + 回归基线 + 可视化仪表盘 |
| **用户附图问答** | 聊天框上传图片（截图/图表/照片），后端 VLM 描述并入本轮上下文后回答 |
| **管理后台** | 文档标签管理、RAG 策略配置、评测面板（告警/失败检索/评分明细可回溯）、对话统计 |

## 📊 评测结果

### 检索效果（test-corpus 企业行政文档 67 条 QA）
| 策略 | HR@1 | HR@3 | HR@5 | HR@10 | MRR | P50(ms) |
|---|---|---|---|---|---|---|
| 纯向量检索 | 82.09% | 94.03% | 97.01% | 97.01% | 0.8799 | 6.8ms |
| 混合检索 (BM25+向量+RRF) | 85.07% | 94.03% | 95.52% | 97.01% | 0.8986 | 17.2ms |
| 混合+RRF+Cross-Encoder精排 | 86.57% | 92.54% | 92.54% | 92.54% | 0.8955 | 265.6ms |

### 回复质量（LLM-as-Judge 三维评测）
| 策略 | 样本数 | 相关性 | 完整性 | 有用性 | 总分 |
|---|---|---|---|---|---|
| 纯向量检索 (top-3) | 67 | 4.94 | 4.67 | 4.78 | 4.80 |
| 混合检索 (BM25+向量+RRF) | 67 | 4.99 | 4.61 | 4.79 | 4.80 |
| 混合+RRF+Cross-Encoder精排 | 67 | 4.90 | 4.60 | 4.79 | 4.76 |

### 公开基准 CRUD-RAG（8.6 万文档 / 28.3 万 chunks，Milvus 规模化）

评测范式：CRUD-RAG 测试样本的源文档不在发布语料中（探查实证），故采用**已知项注入**——把抽样样本的源文档注入检索库作为"已知相关项"，测问题能否在 28.3 万干扰 chunks 中召回。450 条 = 单文档/双文档/三文档各 150（seed=42，样本 ID 落盘可复现）。**口径与上方 67 条 QA 不同，不可直接对比**（那是领域内 golden 命中）。

**规模化基础设施（实测）**

| 项 | 值 |
|---|---|
| 语料 / 索引 | 86,834 篇中文新闻 → 283,110 子块（Milvus `crud_rag`） |
| 导入 | ~27 chunks/s，全量约 2.9h；0 入库报错、count 精确 |
| 磁盘 | 17.5 KB/chunk → 4.73GB（growing）/ ~6.6GB（建索引后） |
| 检索延迟 | 混合检索 ~20ms/query（450 条实测均值，含 query embedding） |

**Task 2 检索指标（450 条，已知项注入，top-10 文档级）**

| 子集 | HR@1 | HR@3 | HR@5 | MRR | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| 单文档（hybrid） | 0.733 | 0.940 | 0.960 | 0.831 | 0.980 | 0.870 |
| 双文档（hybrid） | 0.360 | 0.740 | 0.807 | 0.552 | 0.740 | 0.516 |
| 三文档（hybrid） | 0.280 | 0.707 | 0.813 | 0.502 | 0.631 | 0.422 |
| 单文档（dense-only） | 0.687 | 0.907 | 0.940 | 0.797 | 0.973 | 0.842 |
| 双文档（dense-only） | 0.400 | 0.733 | 0.773 | 0.567 | 0.730 | 0.522 |
| 三文档（dense-only） | 0.313 | 0.667 | 0.820 | 0.506 | 0.596 | 0.411 |

发现：单文档 hybrid 更优（BM25 精确互补）；**双/三文档 dense 反超 hybrid**——跨文档聚合问题上 query 高频词被 BM25 带入同主题干扰文档。

**Task 3 端到端（450 条，LLM-as-Judge 0-5）**

| 任务 | n | 结果 | 说明 |
|---|---|---|---|
| Read（单文档问答） | 150 | **4.51** | 对照"无检索基线"（模型直接答不上来）提升显著 |
| Create（新闻续写） | 150 | **2.99** | 背景增强续写（有背景 3.26 / 无背景 2.69，70 条无召回如实） |
| Update（幻觉识别+修正） | 150 | 识别率 **0.28** / 修正 **0.80** | 定位 LLM 判断力上限，低分如实 |

token 成本 ~98 万（DeepSeek）。Delete 任务官方数据缺失，跳过；Read 端到端仅覆盖单文档（双/三文档的跨文档推理只在检索指标中覆盖）。

**增强三档对比（同一 450 条，top-10 文档级）**

| 子集 | 档位 | HR@1 | MRR | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|
| 单文档 | T1 裸检索 | 0.733 | 0.832 | 0.980 | 0.869 |
| 单文档 | T2 管线（精排/阈值/Autocut/MMR） | **0.753** | 0.837 | 0.993 | 0.875 |
| 单文档 | T3 完整（+多查询/HyDE） | 0.747 | 0.812 | 0.900 | 0.835 |
| 双文档 | T1 裸检索 | **0.327** | 0.527 | 0.740 | 0.528 |
| 双文档 | T2 管线 | 0.320 | 0.492 | 0.710 | 0.517 |
| 双文档 | T3 完整 | 0.320 | 0.504 | 0.637 | 0.495 |
| 三文档 | T1 裸检索 | 0.293 | 0.511 | 0.631 | 0.462 |
| 三文档 | T2 管线 | **0.333** | 0.530 | 0.702 | 0.518 |
| 三文档 | T3 完整 | 0.327 | 0.506 | 0.560 | 0.443 |

结论：T2（精排+阈值+Autocut+MMR）在单/三文档有实证收益（HR@1 +2~4pp）；T3（多查询分解+HyDE）三档 R@10 全面下降 → 主路径默认关、**HyDE 仅检索空结果时抢救**（数据驱动，详见 doc/rag_retrieval.md）。

**chunk 粒度实验（同 2 万语料池，child 子块 vs parent 整篇）**

| 子集 | child HR@1 | parent HR@1 | 全量 86k 池 parent vs child 基线 |
|---|---|---|---|
| 单文档 | 0.807 | **0.853** | 0.733 → 0.680（父块回退 5.3pp，整篇词面宽、BM25 更易撞干扰） |
| 双文档 | 0.560 | **0.647** | 0.327 → 0.340（R@10 0.740→0.790） |
| 三文档 | 0.580 | **0.593** | 0.293 → **0.367**（HR@1 +7.4pp，R@10 +8pp） |

结论：chunk 粒度**分场景最优**——多文档聚合用父块（整篇事实参与匹配）、单文档精确用子块；混合粒度是演进方向。

**数据产物**：`data/crud_rag/eval/`（task2/task3/三档对比/实验 JSON）、`data/crud_rag/repo/`（语料，本地不入库 git）、脚本 `scripts/milvus/`。全部数字真实跑批可复现。



## 🛠 技术栈

| 层级 | 选型 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| LLM | DeepSeek（OpenAI SDK） |
| 向量存储 | Milvus（默认，内置 BM25 中文分词）/ ChromaDB（可切换） |
| Embedding | BAAI/bge-small-zh（Sentence-BERT） |
| 重排序 | BAAI/bge-reranker-base（CrossEncoder） |
| 关键词检索 | BM25（Milvus 内置 chinese analyzer；Chroma 兼容路径 rank-bm25/bigram） |
| 文档解析 | PDF：PyMuPDF+pdfplumber；DOCX：python-docx；图片：DeepSeek vision-exp（VLM 描述） |
| 前端 | React 18/19 + TypeScript + Ant Design 5/6 + Vite |
| 数据库 | SQLite + ChromaDB |

## 🚀 快速开始

```bash
# 1. 克隆并配置
git clone <your-repo-url> && cd KBAgent
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 2. Python 环境
pip install -r requirements.txt

# 3. 启动后端（首次自动下载 embedding 模型 ~100MB）
python3 -m uvicorn app.api.chat_api:app --port 8000

# 4. 前端（新终端）
cd web/admin-portal && npm install && npm run dev
cd web/user-portal && npm install && npm run dev
```

## 📁 项目结构

```
KBAgent/
├── app/                  # Python 后端
│   ├── agent/            # Agent 管理层（AgentManager, ToolRouter, SkillManager）
│   ├── api/              # FastAPI 接口（chat, admin, eval）
│   ├── db/               # 文档存储（SQLite）
│   ├── eval/             # LLM-as-Judge 评测管线
│   ├── memory/           # 多层记忆系统
│   ├── rag/              # RAG 管线（分块/检索/重排序/向量库）
│   └── tools/            # 工具示例
├── config/               # 策略配置
├── doc/                  # 技术文档（中文）
├── skills/               # Skill 系统文件
├── web/                  # 前端
│   ├── admin-portal/     # 管理门户（React + Ant Design）
│   └── user-portal/      # 用户门户（React + Ant Design）
└── tests/                # 测试
```

## 📖 详细文档

| 文档 | 内容 |
|------|------|
| [doc/architecture.md](doc/architecture.md) | 系统架构与请求流程 |
| [doc/auth_guide.md](doc/auth_guide.md) | 登录与鉴权详解（token/密码哈希/HMAC/完整流程示例） |
| [doc/v2_vision.md](doc/v2_vision.md) | V2 愿景：知识库资源化（个人私库 + 共享市场规划） |
| [doc/skill_system.md](doc/skill_system.md) | 渐进式多知识库路由 |
| [doc/hybrid_retrieval.md](doc/hybrid_retrieval.md) | RRF 混合检索 |
| [doc/memory_system.md](doc/memory_system.md) | 多层记忆系统 |
| [doc/evaluation.md](doc/evaluation.md) | LLM-as-Judge 评测 |
| [doc/tech_stack.md](doc/tech_stack.md) | 技术选型 |
| [doc/features.md](doc/features.md) | 功能列表 |

## 📄 License

MIT
