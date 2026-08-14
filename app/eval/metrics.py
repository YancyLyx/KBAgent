# -*- coding: utf-8 -*-
"""请求级监控埋点

记录每次对话请求的耗时、缓存命中、错误与 token 估算，
以 JSONL 追加写入 data/request_metrics.jsonl，供可观测性与成本分析使用。

埋点不依赖任何外部服务，纯本地文件追加；生产环境可替换为
Prometheus + Grafana 或 SkyWalking 等链路观测平台。
"""

import json
import os
from datetime import datetime
from typing import Optional


METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "request_metrics.jsonl",
)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文约 1 token/字符，英文约 4 字符/token"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 4)


def record_request(
    query: str,
    latency_ms: float,
    cached: bool = False,
    error: Optional[str] = None,
    input_chars: int = 0,
    output_chars: int = 0,
    alert: bool = False,
) -> None:
    """记录一条请求埋点"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:200],
        "latency_ms": round(latency_ms, 1),
        "cached": cached,
        "error": error,
        "input_tokens_est": _estimate_tokens(query) + input_chars // 3,
        "output_tokens_est": output_chars // 3,
        "alert": alert,
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_metrics(limit: int = 1000) -> list:
    """读取最近 N 条埋点（用于管理面板展示）"""
    if not os.path.exists(METRICS_PATH):
        return []
    lines = []
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(lines) >= limit:
                break
    return lines


def summary() -> dict:
    """聚合最近埋点，输出简易指标（QPS、P50/P95 延迟、缓存命中率、错误率）"""
    import numpy as np

    metrics = load_metrics(limit=2000)
    if not metrics:
        return {"total": 0}

    latencies = [m["latency_ms"] for m in metrics]
    cached = sum(1 for m in metrics if m.get("cached"))
    errors = sum(1 for m in metrics if m.get("error"))

    return {
        "total": len(metrics),
        "qps_est": round(len(metrics) / 3600, 4),  # 按最近一小时估算，演示用
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "cache_hit_rate": round(cached / len(metrics), 3),
        "error_rate": round(errors / len(metrics), 3),
        "total_input_tokens_est": sum(m.get("input_tokens_est", 0) for m in metrics),
        "total_output_tokens_est": sum(m.get("output_tokens_est", 0) for m in metrics),
    }
