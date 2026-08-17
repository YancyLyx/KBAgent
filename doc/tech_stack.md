# 技术选型

## 后端

| 组件 | 技术 | 用途 |
|------|------|------|
| 框架 | FastAPI + Uvicorn | REST API 服务 |
| LLM | DeepSeek（OpenAI SDK） | 文本生成与工具调用 |
| 向量库 | ChromaDB（PersistentClient） | 向量存储与相似度检索 |
| Embedding | BAAI/bge-small-zh（Sentence-BERT） | 文本向量化 |
| 重排序 | BAAI/bge-reranker-base（CrossEncoder） | 二阶段相关性排序 |
| 关键词 | rank-bm25（BM25Okapi） | 精确匹配检索 |
| PDF 解析 | PyMuPDF + pdfplumber | 版式/表格提取 |
| 文本切割 | LangChain RecursiveCharacterTextSplitter + 自研章节分块（PDF 字号 / Markdown 标题） | 两级分块：父块按章节语义，子块按 RecursiveCharacterTextSplitter |
| 配置 | PyYAML + python-dotenv | 配置项与密钥管理 |
| 数据库 | SQLite（会话/文档/摘要/话题）+ JSONL 文件（用户偏好） | 会话/文档/摘要用 SQLite；偏好用文件只追加保留时间线 |
| 语义缓存 | 进程内 LRU + TTL（接口可换 Redis） | 精确 + 语义两级命中 |
| 评测 | LLM-as-Judge（OpenAI 兼容接口）+ 本地检索指标脚本 | 实时三维评分 + 离线 HR/MRR 评测 |

## 前端

| 组件 | 技术 | 用途 |
|------|------|------|
| 用户门户 | React 19 + Ant Design 6 + Vite | 用户对话界面 |
| 管理门户 | React 18 + Ant Design 5 + Vite + Recharts | 管理仪表盘 |

## 环境要求

- Python 3.12+
- Node.js 18+
- 8GB+ 内存（用于 Embedding 模型）
- 互联网连接（首次下载模型 + LLM API 调用）

## 备注

- `requirements.txt` 中的 `sqlite-utils` 为历史遗留依赖，代码实际使用标准库
  `sqlite3`，可安全移除
