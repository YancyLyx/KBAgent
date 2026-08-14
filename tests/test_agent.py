# -*- coding: utf-8 -*-
"""Agent 功能测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.agent_manager import AgentManager


def test_agent_init():
    """测试 Agent 初始化"""
    agent = AgentManager(
        user_id="test_user",
        enable_memory=False,
        enable_rag=False,
        enable_tools=False,
    )
    assert agent is not None
    assert agent.user_id == "test_user"
    print("✓ test_agent_init")


def test_agent_light():
    """测试 Agent 无 RAG 模式"""
    agent = AgentManager(
        enable_memory=True,
        enable_rag=False,
        enable_tools=True,
    )
    assert agent.tool_router is not None
    assert agent.short_memory is not None
    print("✓ test_agent_light")


def test_small_talk_detection():
    """注入路由：纯闲聊跳过画像/摘要注入，知识问题正常注入"""
    from app.agent.agent_manager import AgentManager

    # 纯闲聊 → 跳过注入
    assert AgentManager._is_small_talk("你好")
    assert AgentManager._is_small_talk("你是谁")
    assert AgentManager._is_small_talk("你好，在吗")
    assert AgentManager._is_small_talk("谢谢")

    # 知识/业务问题 → 正常注入（即使带问候前缀）
    assert not AgentManager._is_small_talk("报销流程是什么")
    assert not AgentManager._is_small_talk("你好，请问报销流程是什么")
    assert not AgentManager._is_small_talk("员工持股计划的行权期")
    print("✓ test_small_talk_detection")


if __name__ == "__main__":
    print("=== Agent Tests ===")
    test_agent_init()
    test_agent_light()
    test_small_talk_detection()
    print("\n✅ All agent tests passed")
