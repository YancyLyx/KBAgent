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


if __name__ == "__main__":
    print("=== Agent Tests ===")
    test_agent_init()
    test_agent_light()
    print("\n✅ All agent tests passed")
