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

基于 ReAct 架构的智能知识库问答系统。支持多格式文档（PDF/MD/TXT）上传与管理，让 LLM 自主决策工具调用，实现知识库检索与文件操作的智能编排。

## 🎯 核心功能

| 功能 | 说明 |
|------|------|
| **多格式文档接入** | PDF 章节检测 + 表格提取，Markdown 标题分块，纯文本兜底 |
| **父子分块策略** | 子块精确匹配，父块完整上下文，兼顾精度与完整性 |
| **混合检索（RRF）** | 向量检索 + BM25 关键词，Reciprocal Rank Fusion 合并，CrossEncoder 精排 |
| **渐进式知识库路由** | LLM 三层渐进发现可用知识库，按需选择分类检索 |
| **ReAct Agent** | 两轮 LLM 调用：自主决策工具 → 执行 → 生成回答。内置 6 个工具 |
| **多层记忆系统** | 短期 deque + 长程向量语义召回 + 演进式摘要 + 用户画像 |
| **LLM-as-Judge** | 实时三维度评分，低分告警 + 自动兜底 + 回归基线 + 可视化仪表盘 |
| **管理后台** | 文档标签管理、RAG 策略配置、评测面板、对话统计 |

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


## 🛠 技术栈

| 层级 | 选型 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| LLM | DeepSeek（OpenAI SDK） |
| 向量存储 | ChromaDB（PersistentClient） |
| Embedding | BAAI/bge-small-zh（Sentence-BERT） |
| 重排序 | BAAI/bge-reranker-base（CrossEncoder） |
| 关键词检索 | rank-bm25（BM25Okapi） |
| PDF 解析 | PyMuPDF + pdfplumber |
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
