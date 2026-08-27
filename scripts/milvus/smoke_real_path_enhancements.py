# -*- coding: utf-8 -*-
"""Step3 冒烟：验证产品工具路径的真实增强执行（非摆设）。

用产品默认配置（Chroma knowledge_base + 制度文档）构造真实 AgentManager，
直接调用工具执行路径的两个方法：
1. _search_with_expansion：多主题问题应触发 LLM 拆分 → 返回带"多路查询合并结果"
2. _search_with_self_rag：检索不足时 LLM 判断 + 改写补检

输出为真实行为记录；LLM 判断有随机性，不硬性断言触发与否，但标记必须出现
在"确实走了增强"的结果里。
"""
import asyncio, sys, os
from dotenv import load_dotenv
sys_path = "/Users/yanxinluo/Documents/PycharmProjects/KBAgent"
load_dotenv(f"{sys_path}/.env")
sys.path.insert(0, sys_path)
from app.agent.agent_manager import AgentManager


async def main():
    agent = AgentManager(
        user_id="smoke_real_path",
        enable_memory=False,
        enable_rag=True,
        enable_tools=True,
        enable_eval=False,
    )
    print("query_expansion_enabled:", agent.query_expansion_enabled)
    print("self_rag_enabled:", agent.self_rag_enabled)

    # 1) 多主题问题 → 期望触发多查询分解 + HyDE
    q1 = {"query": "差旅报销和餐补分别怎么规定", "tag": "差旅报销"}
    r1 = await agent._search_with_expansion(q1)
    print("\n[expansion] query:", q1["query"])
    if r1 is None:
        print("[expansion] 未触发（LLM 判断无需拆分）→ 走原单次检索路径")
    else:
        print(f"[expansion] 触发 ✓ 长度={len(r1)} 含标记={'多路查询合并结果' in r1}")
        print("  前 200 字:", r1[:200].replace(chr(10), " "))

    # 2) 检索可能不足的问题 → 走 Self-RAG 充分性判断
    q2 = {"query": "员工持股计划的锁定期和考核指标是什么", "tag": "股权激励"}
    r2 = await agent._search_with_self_rag(q2)
    print("\n[self_rag] query:", q2["query"])
    if r2 is None:
        print("[self_rag] 返回 None（走原路径/失败兜底）")
    else:
        print(f"[self_rag] 执行 ✓ 长度={len(r2)}")
        print("  前 200 字:", r2[:200].replace(chr(10), " "))

    # 3) 兜底验证：配置关闭时 expansion 应直接走原路径
    agent.query_expansion_enabled = False
    r3 = await agent._search_with_expansion(q1)
    print("\n[expansion-disabled] 应返回 None:", r3 is None)


asyncio.run(main())
