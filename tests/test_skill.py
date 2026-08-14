# -*- coding: utf-8 -*-
"""测试 Phase 1：Skill 系统 + 动态标签"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.skill_manager import SkillManager


def _clean_refs():
    """清理测试产生的参考文件"""
    import os, glob
    ref_dir = Path(__file__).parent.parent / "skills" / "knowledge_retrieval" / "references"
    for f in glob.glob(str(ref_dir / "test_*.md")):
        os.remove(f)


def test_skill_manager_empty():
    """无标签时 SkillManager 行为"""
    class MockStore:
        def get_available_tags(self):
            return []
    sm = SkillManager(MockStore())
    intro = sm.get_intro()
    assert len(intro) > 0  # 仍返回 skills 目录下的静态概览
    assert len(sm.get_tool_schemas()) >= 4  # file_tools 至少有 4 个
    names = [s["function"]["name"] for s in sm.get_tool_schemas()]
    assert "read_file" in names
    print("✓ test_skill_manager_empty")


def test_skill_manager_with_tags():
    """有标签时工具 schema 包含正确枚举"""
    class MockStore:
        def get_available_tags(self):
            return ["文献", "项目"]
    sm = SkillManager(MockStore())
    schemas = sm.get_tool_schemas()
    # 验证知识库工具包含标签枚举
    for s in schemas:
        fn = s["function"]
        if fn["name"] == "read_category_info":
            tag_param = fn["parameters"]["properties"]["tag"]
            assert "文献" in tag_param["enum"]
            assert "项目" in tag_param["enum"]
            break
    else:
        assert False, "read_category_info not found"

    # 验证参考文件自动生成
    ref = sm._read_category_info(tag="文献")
    assert "文献" in ref
    assert "search_knowledge_base" in ref
    _clean_refs()
    print("✓ test_skill_manager_with_tags")


if __name__ == "__main__":
    print("=== Skill Tests ===")
    test_skill_manager_empty()
    test_skill_manager_with_tags()
    print("\n✅ All skill tests passed")
