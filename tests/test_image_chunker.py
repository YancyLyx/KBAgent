# -*- coding: utf-8 -*-
"""图片分块器单元测试：mock VLM 描述，验证 chunk 结构与失败占位"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.image_chunker import ImageChunker


def _fake_png(tmp_path):
    # 1x1 像素最小 PNG
    import base64
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    path = tmp_path / "chart.png"
    path.write_bytes(png)
    return path


def test_image_chunk_described(tmp_path, monkeypatch):
    path = _fake_png(tmp_path)
    chunker = ImageChunker()
    monkeypatch.setattr(chunker, "_describe", lambda p: "柱状图：2023 Q3 销售额 120 万，同比 +15%")
    chunks = chunker.chunk_image(str(path), metadata={"source": "chart.png", "tag": "图表"})
    assert chunks
    c = chunks[0]
    assert "柱状图" in c["content"]
    assert "image" in c.get("content_types", [])
    assert c.get("source") == "chart.png"
    assert c.get("tag") == "图表"


def test_image_chunk_failure_placeholder(tmp_path, monkeypatch):
    path = _fake_png(tmp_path)
    chunker = ImageChunker()

    def boom(p):
        raise RuntimeError("vision unavailable")

    monkeypatch.setattr(chunker, "_describe", boom)
    chunks = chunker.chunk_image(str(path), metadata={"source": "chart.png"})
    assert chunks
    assert "描述失败" in chunks[0]["content"], "失败应落可读占位而非静默"
