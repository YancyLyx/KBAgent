# -*- coding: utf-8 -*-
"""Phase 0 验证脚本：Milvus 2.5.x 内置 BM25(chinese/jieba) + RRFRanker 混合检索。

验证目标（对应任务 Phase 0 验收门）：
1. 中文全文检索（analyzer type=chinese/jieba）能返回正确结果
2. BM25 function + RRFRanker 混合检索能跑通
3. 标量过滤（tag）+ partition 隔离可用

说明：
- dense 向量用随机值占位（本脚本只验证 BM25/RRF/filter/partition 管道，
  真实 embedding 在 Phase 2 入库时接入 bge-small-zh-v1.5 dim=512）。
- 一致性级别 Bounded（入库快，不阻塞）。
"""
import os
import random
import time
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker

URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "phase0_verify"
PARTITION_A = "kb_policy"
PARTITION_B = "kb_news"

client = MilvusClient(uri=URI, consistency_level="Bounded")

# 清理旧集合
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)
    print(f"[cleanup] dropped {COLLECTION}")

# ----------------------------------------------------------------------
# 1. Schema：显式字段 + BM25 function
#    主键=chunk_id(VARCHAR)，与现有 Chroma 路径的 chunk_id 语义一致
#    元数据显式字段：tag/source/parent_id/parent_section/parent_content/
#                   chunk_index/title/page_range/content_types/doc_type
#    多余元数据走动态字段（enable_dynamic_field=True），不进 schema
# ----------------------------------------------------------------------
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
# 中文 analyzer：type=chinese 内部使用 jieba 分词
schema.add_field(
    "text", DataType.VARCHAR, max_length=4096,
    enable_analyzer=True, analyzer_params={"type": "chinese"},
)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR, description="bm25")
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=512)
# 显式标量字段（参考 What-to-eat-today 的显式建模，方便走标量索引过滤）
schema.add_field("tag", DataType.VARCHAR, max_length=64)
schema.add_field("source", DataType.VARCHAR, max_length=256)
schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
schema.add_field("parent_section", DataType.VARCHAR, max_length=256)
schema.add_field("parent_content", DataType.VARCHAR, max_length=8192)
schema.add_field("chunk_index", DataType.INT64)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("page_range", DataType.VARCHAR, max_length=64)
schema.add_field("content_types", DataType.VARCHAR, max_length=256)
schema.add_field("doc_type", DataType.VARCHAR, max_length=64)

# BM25 function: text -> sparse
schema.add_function(Function(
    name="bm25_text",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse"],
))

# ----------------------------------------------------------------------
# 2. Index
# ----------------------------------------------------------------------
index_params = client.prepare_index_params()
index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(
    field_name="sparse", index_type="SPARSE_WAND", metric_type="BM25",
)

client.create_collection(
    collection_name=COLLECTION, schema=schema, index_params=index_params,
)
print(f"[create] collection={COLLECTION} partition-aware")

# 创建 partition（粗粒度 ASCII 命名：按知识库/语料隔离，tag 走标量 filter）
# 策略说明：Milvus partition 名仅 ASCII，中文 tag 不直接作 partition 名。
# 采用"粗粒度 partition + tag 标量过滤"：每个知识库/语料一个 partition
# （如 kb_policy=制度库、kb_news=CRUD-RAG 新闻语料），drop 即可整体重导；
# 库内细分（制度/文献/新闻）由 tag 标量过滤实现。
for p in (PARTITION_A, PARTITION_B):
    if not client.has_partition(COLLECTION, p):
        client.create_partition(COLLECTION, p)
        print(f"[partition] created {p}")

# ----------------------------------------------------------------------
# 3. 插入数据（Bounded 一致性）
# ----------------------------------------------------------------------
docs = [
    # partition=kb_policy 制度类
    {"chunk_id": "c1", "text": "差旅报销标准为每晚400元，出差期间餐饮补贴每日100元。", "tag": "制度", "source": "差旅管理办法.pdf", "parent_id": "doc1", "parent_section": "第三章 报销标准", "parent_content": "差旅报销标准为每晚400元...", "chunk_index": 0, "title": "差旅管理办法", "page_range": "1-2", "content_types": "text", "doc_type": "pdf", "_partition": PARTITION_A},
    {"chunk_id": "c2", "text": "员工行权期内可按授予价格购买公司股票，行权期一般为四年。", "tag": "制度", "source": "股权激励制度.pdf", "parent_id": "doc2", "parent_section": "第二章 行权", "parent_content": "员工行权期内可按授予价格...", "chunk_index": 0, "title": "股权激励制度", "page_range": "3-4", "content_types": "text", "doc_type": "pdf", "_partition": PARTITION_A},
    {"chunk_id": "c3", "text": "年假按工龄累计：满1年5天，满5年10天，满10年15天。", "tag": "制度", "source": "考勤制度.pdf", "parent_id": "doc3", "parent_section": "年假", "parent_content": "年假按工龄累计...", "chunk_index": 0, "title": "考勤制度", "page_range": "5", "content_types": "text", "doc_type": "pdf", "_partition": PARTITION_A},
    # partition=kb_news 新闻类（用于验证 partition 隔离 + tag 过滤）
    {"chunk_id": "c4", "text": "某科技公司发布新一代大模型，支持百万token上下文。", "tag": "新闻", "source": "tech_news.txt", "parent_id": "doc4", "parent_section": "", "parent_content": "", "chunk_index": 0, "title": "科技新闻", "page_range": "", "content_types": "text", "doc_type": "txt", "_partition": PARTITION_B},
    {"chunk_id": "c5", "text": "差旅行业复苏，今年机票预订量同比增长40%。", "tag": "新闻", "source": "travel_news.txt", "parent_id": "doc5", "parent_section": "", "parent_content": "", "chunk_index": 0, "title": "差旅新闻", "page_range": "", "content_types": "text", "doc_type": "txt", "_partition": PARTITION_B},
]
random.seed(42)
rows = []
for d in docs:
    p = d.pop("_partition")
    row = {k: v for k, v in d.items() if k != "_partition"}
    row["dense"] = [random.random() for _ in range(512)]
    rows.append((p, row))

for p, r in rows:
    client.insert(COLLECTION, r, partition_name=p)
print(f"[insert] {len(rows)} rows into 2 partitions")
# Bounded 一致性下 growing segment 首次可检索有偶发延迟，等待一下
time.sleep(2)

# ----------------------------------------------------------------------
# 4. 验证 A：中文全文检索（BM25 sparse）—— "行权期" 领域词
# ----------------------------------------------------------------------
print("\n=== 验证A: BM25 中文全文检索 (query='行权期 购买股票') ===")
# 注：drop_ratio_search 在小语料+短查询上会丢掉关键 token，Phase 0 不启用；
# Phase 2 大语料入库后再按需调参。
res = client.search(
    collection_name=COLLECTION,
    data=["行权期 购买股票"],          # BM25 字段直接传文本，function 自动算 sparse 查询向量
    anns_field="sparse",
    limit=3,
    output_fields=["text", "tag", "source", "parent_id", "parent_section", "chunk_id"],
)
for hit in res[0]:
    print(f"  chunk_id={hit['chunk_id']} score={hit['distance']:.4f} tag={hit['entity']['tag']} section={hit['entity']['parent_section']} text={hit['entity']['text'][:40]}")
assert any("行权期" in h["entity"]["text"] for h in res[0]), "BM25 未命中 行权期 文档"
# 关键修正验证：parent_section 必须能取回（retriever/agent 层依赖它）
assert any(h["entity"]["parent_section"] == "第二章 行权" for h in res[0]), "parent_section 字段丢失"
print("  [OK] BM25 命中 行权期 文档（jieba 分词生效），parent_section 正确取回")

# ----------------------------------------------------------------------
# 5. 验证 B：标量过滤（tag=制度）+ partition 隔离
# ----------------------------------------------------------------------
print("\n=== 验证B: BM25 + 标量过滤 tag='制度' (query='差旅报销') ===")
res = client.search(
    collection_name=COLLECTION,
    data=["差旅报销"],
    anns_field="sparse",
    limit=5,
    filter='tag == "制度"',
    output_fields=["text", "tag", "source"],
)
for hit in res[0]:
    print(f"  id={hit['chunk_id']} score={hit['distance']:.4f} tag={hit['entity']['tag']} text={hit['entity']['text'][:40]}")
assert all(h["entity"]["tag"] == "制度" for h in res[0]), "tag 过滤失效"
print("  [OK] tag 过滤全部命中 制度")

print("\n=== 验证B2: partition 隔离 (只在 kb_news 检索 '差旅') ===")
res = client.search(
    collection_name=COLLECTION,
    data=["差旅"],
    anns_field="sparse",
    limit=5,
    partition_names=[PARTITION_B],
    output_fields=["text", "tag", "source"],
)
for hit in res[0]:
    print(f"  id={hit['chunk_id']} score={hit['distance']:.4f} tag={hit['entity']['tag']} source={hit['entity']['source']}")
assert all(h["entity"]["source"].endswith("news.txt") for h in res[0]), "partition 隔离失效"
print("  [OK] partition 隔离只返回 kb_news")

# ----------------------------------------------------------------------
# 6. 验证 C：RRFRanker 混合检索（dense + sparse）
# ----------------------------------------------------------------------
print("\n=== 验证C: RRFRanker 混合检索 (query='差旅报销标准') ===")
query_text = "差旅报销标准"
query_dense = [random.random() for _ in range(512)]  # 占位，真实场景用 bge 编码

req_dense = AnnSearchRequest(
    data=[query_dense], anns_field="dense", param={"metric_type": "COSINE"}, limit=5,
)
req_sparse = AnnSearchRequest(
    data=[query_text], anns_field="sparse", param={"metric_type": "BM25"}, limit=5,
)
res = client.hybrid_search(
    collection_name=COLLECTION,
    reqs=[req_dense, req_sparse],
    ranker=RRFRanker(100),       # 任务锚点：RRF k=100
    limit=3,
    output_fields=["text", "tag", "source"],
)
for hit in res[0]:
    print(f"  id={hit['chunk_id']} score={hit['distance']:.4f} tag={hit['entity']['tag']} text={hit['entity']['text'][:40]}")
print("  [OK] hybrid_search + RRFRanker 跑通")

# ----------------------------------------------------------------------
# 7. 验证 D：delete by filter（按 parent_id 删除某文档所有块）
# ----------------------------------------------------------------------
print("\n=== 验证D: delete by filter parent_id=='doc1' ===")
client.delete(COLLECTION, filter='parent_id == "doc1"')
# Bounded 一致性下立即查可能仍可见，等一下再查
time.sleep(0.5)
res = client.search(
    collection_name=COLLECTION, data=["差旅报销标准为每晚400元"],
    anns_field="sparse", limit=5, output_fields=["text", "parent_id"],
)
hit_texts = [h["entity"]["text"][:30] for h in res[0]]
print(f"  删除后检索结果: {hit_texts}")
assert not any("doc1" == h["entity"]["parent_id"] for h in res[0]), "delete 失效"
print("  [OK] parent_id=doc1 已删除")

# ----------------------------------------------------------------------
# 统计（注：growing segment 的 row_count 可能为 0，属已知延迟，以可检索为准）
# ----------------------------------------------------------------------
stats = client.get_collection_stats(COLLECTION)
print(f"\n[stats] {stats}（row_count 为 growing segment 延迟统计，可检索即数据存在）")
print("\n========== Phase 0 验证全部通过 ==========")
