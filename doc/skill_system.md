# 渐进式多知识库路由（Skill 系统）

## 设计目标

常见的 RAG 做法是将所有文档混入一个向量池，检索时不知道用户想要什么类型的信息。这在多知识源场景下有两个问题：

1. **检索噪声**：混合池中学术论文、产品手册、项目文档混在一起，语义相似的无关结果会被误召回。
2. **无路由机制**：LLM 没有途径"选择"查哪个知识库，只能盲目调用检索工具。

解决方案：让 LLM 先了解有哪些知识可用，再决定查哪个。这模仿了人类查找信息的自然方式——先选择去哪个图书馆，再检索。

## 三层渐进式披露

```
第 1 层：Skill 总览（注入 system prompt）
  "可用知识库：文献、项目、产品手册"
  LLM 看到分类名称和一句话简介。

第 2 层：read_category_info(tag) → 返回分类详细参考
  "文献库：包含 2023-2024 AI 论文，方向 NLP/CV/多模态。
   典型问题：Transformer 原理、最新语义分割方法
   检索工具：search_knowledge_base(query, tag='文献')"

第 3 层：search_knowledge_base(query, tag) → 执行检索
  调用 RAG 管线，仅在该分类内检索，返回文档片段。
```

LLM 逐层展开：不确定时才查看详细参考，确定分类后才检索。

**完整对话示例（渐进披露如何运转）**：

```text
用户：员工持股计划的行权期是多久？

第 1 轮：LLM 看到 system prompt 里的 Skill 总览
  （可用知识库：文献、项目、股权、人力……）
  → 不确定"员工持股计划"属于哪个库，先调用 read_category_info(tag="股权")

第 2 轮：拿到股权库参考（包含行权期、锁定期等关键词说明）
  → 确定该查股权库，调用 search_knowledge_base(query="行权期", tag="股权")

第 3 轮：拿到检索片段，基于片段直接回答 → 跳出循环
```

对比"没有渐进披露"的做法：所有文档混在一个向量池，LLM 只调一次
`search_knowledge_base`，很可能在"人力库"里找到"员工持股"的只言片语，
把不同库的内容混在一起回答。渐进披露用一次 `read_category_info` 的廉价调用
（不检索、只读参考文件），换来检索范围收敛，避免跨库串答案。

## 标签式组织

文档上传时通过自由文本标签标注分类（非预定义枚举）。系统自动：

1. 将标签存入 ChromaDB metadata
2. 在 `skills/knowledge_retrieval/references/{tag}.md` 生成参考文件
3. 更新 skill 总览（`skill.md`）追加当前标签列表
4. 将标签注册到检索工具的 enum 参数（枚举值：在工具 schema 里限定 tag 只能取
   当前知识库分类，模型就不会传出不存在的标签）

新增分类只需上传文档时输入新标签，无需改代码或重启服务。

## 标签过滤的一致性

`search_knowledge_base(query, tag)` 会把 tag 同时传给向量检索（ChromaDB `where`）
和 BM25 检索（metadata 逐条过滤），两路都只返回该分类结果，再从源头消除跨分类
噪声。注意：BM25 与向量库是两套独立索引，曾出现过 BM25 忽略 filters 导致
keyword-only 命中混入其他知识库的问题，修复细节见
[hybrid_retrieval.md](hybrid_retrieval.md)。

## 工具安全边界

文件类工具（read_file / write_file / create_file / list_files）虽然暴露给 LLM，
但有明确安全约束：

- 路径解析限制在项目根目录内，`../` 或绝对路径越界直接报错（早期实现越界时
  静默返回项目根目录，属于安全盲区）
- `.env*`（可能含 API Key）与 `.git*`（可能含凭据）禁止读写
- 敏感配置类文件一律拒绝，防止 LLM 被 prompt injection 诱导读取密钥

面试常问"工具权限边界怎么设计"，答案核心是：**给 LLM 最小权限 + 路径白名单 +
敏感文件黑名单 + 越界即报错，而不是依赖 LLM 自觉**。

## 文件结构

```
skills/
├── knowledge_retrieval/
│   ├── skill.md           ← 技能总览（自动包含当前标签列表）
│   └── references/        ← 各分类详细参考
│       ├── _template.md   ← 自动生成时的模板
│       ├── 文献.md         ← 上传"文献"标签时自动生成
│       └── 项目.md         ← 上传"项目"标签时自动生成
├── file_tools/
│   └── skill.md           ← 文件操作技能
└── web_search/
    └── skill.md           ← 预留
```
