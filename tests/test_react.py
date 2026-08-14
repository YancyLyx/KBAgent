# -*- coding: utf-8 -*-
"""测试 Phase 2：ReAct Agent + ToolRouter"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.tool_router import ToolRouter


def test_tool_router_schemas():
    """ToolRouter 动态生成 tool schema"""
    router = ToolRouter()
    assert router.get_tool_schemas() == []

    class MockSkill:
        def get_tool_schemas(self):
            return [{"type": "function", "function": {"name": "read_category_info"}}]
    router.skill_manager = MockSkill()
    schemas = router.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "read_category_info"
    print("✓ test_tool_router_schemas")


def test_tool_router_call_skill():
    """call_skill_tool 路由"""
    router = ToolRouter()
    router.skill_pipeline = None

    class MockSkill:
        def get_category_reference(self, tag):
            return f"## {tag} 参考文档\n\n调用 search_knowledge_base 工具检索"
    router.skill_manager = MockSkill()

    result = router.call_skill_tool("read_category_info", {"tag": "文献"})
    assert "search_knowledge_base" in result
    assert "文献" in result

    result = router.call_skill_tool("unknown_tool", {})
    assert "未知" in result
    print("✓ test_tool_router_call_skill")


def test_agent_manager_init_light():
    """AgentManager 初始化（不加载 model）"""
    from app.agent.agent_manager import AgentManager
    agent = AgentManager(
        user_id="test_user",
        enable_memory=True,
        enable_rag=False,
        enable_tools=True,
    )
    assert agent.tool_router is not None
    assert agent.short_memory is not None
    print("✓ test_agent_manager_init_light")


if __name__ == "__main__":
    print("=== Phase 2 Tests ===")
    test_tool_router_schemas()
    test_tool_router_call_skill()
    test_agent_manager_init_light()
    print("\n✅ Phase 2 全部通过")
