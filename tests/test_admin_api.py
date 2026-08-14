# -*- coding: utf-8 -*-
"""管理后台API测试脚本

测试 KBAgent - Smart Knowledge Base Q&A System 管理后台API功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import json
import pytest

# API基础URL
BASE_URL = "http://127.0.0.1:8000"

# 管理端鉴权：登录后统一携带 admin token
_admin_session = requests.Session()


def _ensure_admin_token():
    """登录管理接口并设置 Authorization 头（幂等，重复调用刷新 token）"""
    resp = _admin_session.post(
        f"{BASE_URL}/api/admin/login",
        json={"username": "admin", "password": "admin123"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"admin login failed: {resp.status_code} {resp.text}")
    _admin_session.headers["Authorization"] = f"Bearer {resp.json()['token']}"


@pytest.fixture(scope="module", autouse=True)
def _authed_admin():
    _ensure_admin_token()
    yield


# ============== 公共 fixture ==============

def _upload_test_document() -> str:
    """上传一个测试文档，返回 doc_id"""
    test_content = """# 测试文档

这是一个用于测试的文档。

## 功能介绍

KBAgent - Smart Knowledge Base Q&A System 是一个企业级智能客服系统。

主要功能：
1. RAG 知识库问答
2. 文档检索
3. 混合检索
"""
    test_file = Path("test_upload.md")
    test_file.write_text(test_content, encoding='utf-8')
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_upload.md', f, 'text/markdown')}
            response = _admin_session.post(f"{BASE_URL}/api/admin/documents/upload", files=files)
        assert response.status_code == 200, f"上传失败: {response.status_code} {response.text}"
        result = response.json()
        assert result.get('id') is not None
        return result['id']
    finally:
        test_file.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def doc_id():
    """模块级 fixture：上传测试文档，测试结束后清理"""
    doc_id_value = _upload_test_document()
    yield doc_id_value
    # teardown：删除文档（若 test_document_delete 已删，这里 404 无影响）
    try:
        _admin_session.delete(f"{BASE_URL}/api/admin/documents/{doc_id_value}")
    except Exception:
        pass


def test_health():
    """测试健康检查"""
    print("\n" + "=" * 50)
    print("测试1: 健康检查")
    print("=" * 50)

    response = _admin_session.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("健康检查通过!")


def test_document_upload(doc_id):
    """测试文档上传（fixture 已完成上传，这里验证文档存在）"""
    print("\n" + "=" * 50)
    print("测试2: 文档上传")
    print("=" * 50)

    response = _admin_session.get(f"{BASE_URL}/api/admin/documents/{doc_id}")
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"文档ID: {result.get('id')}")
    print(f"文件名: {result.get('original_name')}")
    print(f"状态: {result.get('status')}")

    assert response.status_code == 200
    assert result.get('id') == doc_id
    print("文档上传测试通过!")


def test_document_list():
    """测试文档列表"""
    print("\n" + "=" * 50)
    print("测试3: 文档列表")
    print("=" * 50)

    response = _admin_session.get(f"{BASE_URL}/api/admin/documents")
    print(f"状态码: {response.status_code}")

    result = response.json()
    print(f"总数: {result.get('total')}")
    print(f"当前页: {result.get('page')}")
    print(f"文档数量: {len(result.get('items', []))}")

    for item in result.get('items', []):
        print(f"  - {item['original_name']} ({item['status']})")

    assert response.status_code == 200
    print("文档列表测试通过!")


def test_document_detail(doc_id):
    """测试文档详情"""
    print("\n" + "=" * 50)
    print("测试4: 文档详情")
    print("=" * 50)

    response = _admin_session.get(f"{BASE_URL}/api/admin/documents/{doc_id}")
    print(f"状态码: {response.status_code}")

    result = response.json()
    print(f"文档ID: {result.get('id')}")
    print(f"文件名: {result.get('original_name')}")
    print(f"大小: {result.get('size')} bytes")

    assert response.status_code == 200
    print("文档详情测试通过!")


def test_document_content(doc_id):
    """测试文档内容预览"""
    print("\n" + "=" * 50)
    print("测试5: 文档内容预览")
    print("=" * 50)

    response = _admin_session.get(f"{BASE_URL}/api/admin/documents/{doc_id}/content")
    print(f"状态码: {response.status_code}")

    result = response.json()
    content_preview = result.get('content', '')[:100]
    print(f"内容预览: {content_preview}...")

    assert response.status_code == 200
    print("文档内容预览测试通过!")


def test_document_chunks(doc_id):
    """测试文档分块"""
    print("\n" + "=" * 50)
    print("测试6: 文档分块详情")
    print("=" * 50)

    response = _admin_session.get(f"{BASE_URL}/api/admin/documents/{doc_id}/chunks")
    print(f"状态码: {response.status_code}")

    result = response.json()
    print(f"总分块数: {result.get('total_chunks')}")

    for chunk in result.get('chunks', [])[:3]:
        print(f"  块 {chunk['id']}: {chunk['length']} 字符")

    assert response.status_code == 200
    print("文档分块测试通过!")


def test_rag_config():
    """测试RAG配置"""
    print("\n" + "=" * 50)
    print("测试7: RAG配置")
    print("=" * 50)

    config_path = Path("config/rag_config.yaml")
    original = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    try:
        # 获取配置
        response = _admin_session.get(f"{BASE_URL}/api/admin/config/rag")
        print(f"获取配置状态码: {response.status_code}")
        print(f"当前配置: {json.dumps(response.json(), indent=2, ensure_ascii=False)[:200]}...")

        # 更新配置（仅提交部分段落，验证未提交的 embedding/parent_child 不被清空）
        new_config = {
            "chunk_strategy": {
                "chunk_size": 800,
                "chunk_overlap": 150
            },
            "retrieval_strategy": {
                "search_type": "hybrid",
                "vector_top_k": 15,
                "rerank_top_k": 5
            }
        }

        response = _admin_session.put(
            f"{BASE_URL}/api/admin/config/rag",
            json=new_config
        )
        print(f"更新配置状态码: {response.status_code}")
        print(f"响应: {response.json()}")

        # 验证更新
        response = _admin_session.get(f"{BASE_URL}/api/admin/config/rag")
        config = response.json()
        assert config['chunk_strategy']['chunk_size'] == 800
        # 局部合并：未提交的段落应保留
        assert config.get('embedding', {}).get('model_name', '') != ''
        assert config.get('parent_child', {}).get('enabled', False) is True

        print("RAG配置测试通过!")
    finally:
        # 测试后恢复原配置文件，避免污染仓库配置
        if original is not None:
            config_path.write_text(original, encoding="utf-8")


def test_memory_config():
    """测试记忆配置"""
    print("\n" + "=" * 50)
    print("测试8: 记忆配置")
    print("=" * 50)

    config_path = Path("config/memory_config.yaml")
    original = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    try:
        # 获取配置
        response = _admin_session.get(f"{BASE_URL}/api/admin/config/memory")
        print(f"获取配置状态码: {response.status_code}")
        print(f"当前配置: {json.dumps(response.json(), indent=2, ensure_ascii=False)[:200]}...")

        # 更新配置
        new_config = {
            "short_term_memory": {
                "max_history_length": 10,
                "enable_summary": True
            },
            "long_term_memory": {
                "enabled": True,
                "max_topics_per_user": 20
            }
        }

        response = _admin_session.put(
            f"{BASE_URL}/api/admin/config/memory",
            json=new_config
        )
        print(f"更新配置状态码: {response.status_code}")
        print(f"响应: {response.json()}")

        print("记忆配置测试通过!")
    finally:
        if original is not None:
            config_path.write_text(original, encoding="utf-8")


def test_stats():
    """测试统计API"""
    print("\n" + "=" * 50)
    print("测试9: 统计API")
    print("=" * 50)

    # 概览统计
    response = _admin_session.get(f"{BASE_URL}/api/admin/stats/overview")
    print(f"概览统计状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

    # 对话统计
    response = _admin_session.get(f"{BASE_URL}/api/admin/stats/conversations?days=7")
    print(f"\n对话统计状态码: {response.status_code}")
    result = response.json()
    print(f"天数: {len(result.get('stats', []))}")

    # 查询类型分布
    response = _admin_session.get(f"{BASE_URL}/api/admin/stats/queries")
    print(f"\n查询类型分布状态码: {response.status_code}")
    result = response.json()
    print(f"分布: {result.get('distribution')}")

    print("统计API测试通过!")


def test_document_delete(doc_id):
    """测试文档删除"""
    print("\n" + "=" * 50)
    print("测试10: 文档删除")
    print("=" * 50)

    response = _admin_session.delete(f"{BASE_URL}/api/admin/documents/{doc_id}")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")

    assert response.status_code == 200

    # 验证删除
    response = _admin_session.get(f"{BASE_URL}/api/admin/documents/{doc_id}")
    assert response.status_code == 404

    print("文档删除测试通过!")


if __name__ == "__main__":
    print("KBAgent - Smart Knowledge Base Q&A System - 管理后台API测试")
    print("=" * 50)

    try:
        _ensure_admin_token()

        # 基础测试
        test_health()

        # 文档管理测试
        doc_id = _upload_test_document()
        test_document_list()
        test_document_detail(doc_id)
        test_document_content(doc_id)
        test_document_chunks(doc_id)

        # 配置管理测试
        test_rag_config()
        test_memory_config()

        # 统计测试
        test_stats()

        # 清理
        test_document_delete(doc_id)

        print("\n" + "=" * 50)
        print("所有管理后台API测试通过!")
        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到服务器。请先启动服务:")
        print("  uvicorn app.api.chat_api:app --reload")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
