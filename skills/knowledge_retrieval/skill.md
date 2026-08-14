# Knowledge Retrieval Skill

## 描述
检索企业内部知识库中的文档。知识库按分类（标签）组织，检索前应先了解有哪些分类可用。

## 工具

### read_category_info(tag: str)
查看指定分类的详细参考，包括内容范围、典型问题和检索示例。**检索前建议先调用此工具了解知识库内容。**

### search_knowledge_base(query: str, tag: str)
在指定分类中检索与 query 相关的文档片段。tag 必须从已有分类中选择。

## 使用流程
1. 若不确定该查哪个知识库，先用 read_category_info 查看各分类的详细说明
2. 确定分类后使用 search_knowledge_base 检索
3. 基于检索结果回答用户问题

---

当前知识库分类：IT设备 · 入职转正 · 差旅报销 · 生育假期 · 福利保险 · 考勤请假 · 股权激励 · 薪酬绩效 · 远程办公