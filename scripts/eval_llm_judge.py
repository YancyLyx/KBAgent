#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-as-Judge 评测脚本

使用已有检索结果（step4_*），回填完整父块上下文，
调用回复模型（deepseek-v4-pro）生成回答，评测模型（deepseek-v4-flash）打 3 维分数。
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

# ── 路径 ──────────────────────────────────────────────
project_root = Path(__file__).parent.parent
eval_dir = project_root / "data" / "LLM-as-Judge_eval"
eval_dir.mkdir(parents=True, exist_ok=True)

# ── 读取 .env 配置 ──────────────────────────────────
def load_env():
    env_path = project_root / ".env"
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

env = load_env()
GEN_API_KEY = env.get("DEEPSEEK_API_KEY")
GEN_BASE_URL = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GEN_MODEL = env.get("LLM_MODEL", "deepseek-v4-pro")
EVAL_API_KEY = env.get("EVAL_API_KEY") or GEN_API_KEY
EVAL_BASE_URL = env.get("EVAL_BASE_URL") or GEN_BASE_URL
EVAL_MODEL = env.get("EVAL_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = """你是一个专业的企业智能客服助手。请基于提供的参考文档回答用户问题。
如果参考文档中没有相关信息，请明确告知用户无法回答。
请用中文回答，保持简洁专业。"""


# ── API 调用 ──────────────────────────────────────────
def call_llm(messages: List[dict], model: str, api_key: str, base_url: str,
             temperature: float = 0.3, max_tokens: int = 1000) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                print(f"  [API ERROR] {resp.status_code}: {resp.text[:200]}")
                if attempt < 2:
                    time.sleep(2)
                continue
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
    return None


# ── 数据加载 ──────────────────────────────────────
def load_data():
    with open(project_root / "data" / "testcorpus_eval" / "step4_vector_search.json") as f:
        vector_raw = json.load(f)
    with open(project_root / "data" / "testcorpus_eval" / "step4_hybrid_search.json") as f:
        hybrid_raw = json.load(f)
    with open(project_root / "data" / "testcorpus_eval" / "step4_reranked_results.json") as f:
        reranked_raw = json.load(f)

    with open(project_root / "data" / "testcorpus_eval" / "step1_parent_chunks.json") as f:
        parents = json.load(f)
    with open(project_root / "data" / "testcorpus_eval" / "step2_child_chunks.json") as f:
        children = json.load(f)

    parent_map = {}
    for p in parents:
        pid = hashlib.md5(p["content"].encode()).hexdigest()[:12]
        parent_map[pid] = p["content"]

    child_to_parent = {}
    for c in children:
        child_to_parent[c["chunk_id"]] = parent_map.get(c["parent_id"], "")

    index = {}
    for v, h, r in zip(vector_raw, hybrid_raw, reranked_raw):
        qi = v["query_index"]
        index[qi] = {
            "query": v["query"],
            "answer_gt": v["answer_gt"],
            "vector_topk": [x["chunk_id"] for x in v["top_k"]],
            "hybrid_topk": [x["chunk_id"] for x in h["rrf_fused"]],
            "reranked_topk": [x["chunk_id"] for x in r["reranked"]],
        }
    return index, child_to_parent


def build_context(chunk_ids: List[str], child_to_parent: Dict[str, str], top_k: int = 3):
    seen = set()
    contexts = []
    for cid in chunk_ids:
        parent_text = child_to_parent.get(cid, "")
        if parent_text and parent_text not in seen:
            seen.add(parent_text)
            contexts.append(parent_text)
        if len(contexts) >= top_k:
            break
    return "\n\n---\n\n".join(contexts)


def generate_answer(query: str, context: str) -> Optional[str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考文档：\n{context}\n\n用户问题：{query}\n\n请基于以上参考文档回答问题。"},
    ]
    return call_llm(messages, GEN_MODEL, GEN_API_KEY, GEN_BASE_URL,
                    temperature=0.3, max_tokens=1000)


# ── 更鲁棒的 JSON 提取 ────────────────────────────
def extract_json(raw: str) -> Optional[dict]:
    """从 LLM 输出中提取并解析 JSON，处理各种边缘情况（含 explanation 被截断）"""
    if not raw:
        return None

    # 尝试提取 ```json ... ``` 块
    jm = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if jm:
        raw = jm.group(1).strip()

    # 找到最外层的 {}
    brace_start = raw.find("{")
    if brace_start < 0:
        return None
    brace_end = raw.rfind("}")
    # 截断场景下可能没有右花括号：此时从起点取到末尾，交给逐字段解析
    json_str = raw[brace_start:] if brace_end <= brace_start else raw[brace_start:brace_end + 1]

    # 尝试用标准 json 解析
    try:
        result = json.loads(json_str)
        if all(k in result for k in ["relevance", "completeness", "usefulness"]):
            return result
    except json.JSONDecodeError:
        pass

    # 鲁棒解析：逐字段提取（即使 explanation 被截断，三个维度分数仍可用）
    result = {}
    for key in ["relevance", "completeness", "usefulness"]:
        m = re.search(rf'"{key}"\s*:\s*(\d)', json_str)
        if m:
            result[key] = int(m.group(1))
    if not all(k in result for k in ["relevance", "completeness", "usefulness"]):
        return None

    # 提取 explanation：找到 "explanation": "..." 中引号内的内容
    exp_match = re.search(r'"explanation"\s*:\s*"(.+?)"(?:\s*[,\}])', json_str, re.DOTALL)
    if exp_match:
        result["explanation"] = exp_match.group(1).strip()
    else:
        # 截断/异常引号时退化为取冒号后全部内容
        exp_match = re.search(r'"explanation"\s*:\s*"?([\s\S]*)', json_str)
        if exp_match:
            result["explanation"] = exp_match.group(1).strip().rstrip('"}').strip()
            if not result["explanation"]:
                result["explanation"] = "(说明被截断)"
    return result


def evaluate_answer(query: str, answer: str) -> Optional[Dict]:
    eval_prompt = f"""用户问题：{query}
AI回答：{answer}

评分维度（1-5分）：
1. 相关性（Relevance）：回答是否直接针对用户问题？
2. 完整性（Completeness）：回答是否覆盖了问题需要的所有关键信息？
3. 有用性（Usefulness）：回答是否切实解决了用户的问题？

严格按照以下 JSON 格式输出，不要加任何其他文字，explanation 控制在 50 字以内：
{{"relevance": 1-5, "completeness": 1-5, "usefulness": 1-5, "explanation": "简洁评分理由"}}"""

    messages = [
        {"role": "system", "content": "你是一个回答质量评测助手。请严格基于评分标准打分，仅输出 JSON，不要加其他文字。"},
        {"role": "user", "content": eval_prompt},
    ]
    raw = call_llm(messages, EVAL_MODEL, EVAL_API_KEY, EVAL_BASE_URL,
                   temperature=0.1, max_tokens=500)
    if not raw:
        return None

    result = extract_json(raw)
    if result:
        return {
            "relevance": int(result.get("relevance", 3)),
            "completeness": int(result.get("completeness", 3)),
            "usefulness": int(result.get("usefulness", 3)),
            "explanation": result.get("explanation", ""),
        }
    print(f"  [PARSE] raw: {raw[:200]}")
    return None


# ── 主流程 ────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LLM-as-Judge 评测")
    print("=" * 60)
    print(f"回复模型: {GEN_MODEL}")
    print(f"评测模型: {EVAL_MODEL}")
    print()

    index, child_to_parent = load_data()
    qi_list = sorted(index.keys())
    print(f"共 {len(qi_list)} 条 query")

    strategies = {
        "纯向量检索 (vector)": "vector_topk",
        "混合检索 (BM25+向量+RRF)": "hybrid_topk",
        "混合+RRF+Cross-Encoder精排": "reranked_topk",
    }

    all_results = {}
    all_raw = {}
    answer_cache = {}

    for sname, skey in strategies.items():
        print(f"\n{'─'*50}")
        print(f"[{sname}]")
        print(f"{'─'*50}")

        per_query_results = []
        per_query_raw = []

        for qi in qi_list:
            entry = index[qi]
            query = entry["query"]
            answer_gt = entry["answer_gt"]
            chunk_ids = entry[skey]
            context = build_context(chunk_ids, child_to_parent, top_k=3)

            if not context:
                continue

            # 缓存：相同 context 的回答复用
            cache_key = query
            if cache_key in answer_cache:
                answer = answer_cache[cache_key]
            else:
                answer = generate_answer(query, context)
                if answer:
                    answer_cache[cache_key] = answer
                else:
                    print(f"  Q#{qi}: 生成失败 ❌")
                    continue

            score = evaluate_answer(query, answer)
            if not score:
                print(f"  Q#{qi}: 评测失败 ❌ (answer: {answer[:80]}...)")
                continue

            per_query_results.append({
                "query_index": qi,
                "query": query,
                "answer_gt": answer_gt[:150],
                "generated_answer": answer[:300],
                "relevance": score["relevance"],
                "completeness": score["completeness"],
                "usefulness": score["usefulness"],
                "explanation": score["explanation"],
            })

            per_query_raw.append({
                "query_index": qi,
                "query": query,
                "answer_gt": answer_gt,
                "context_text": context,
                "generated_answer": answer,
                "scores": score,
            })

            if (qi + 1) % 10 == 0:
                print(f"  >> Q#0-{qi}: {len(per_query_results)}/{qi+1}")

        all_results[sname] = per_query_results
        all_raw[sname] = per_query_raw

    # ── 输出 ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {}
    for sname, results in all_results.items():
        if not results:
            summary[sname] = {"error": "无有效结果"}
            continue
        n = len(results)
        rel = round(np.mean([r["relevance"] for r in results]), 2)
        com = round(np.mean([r["completeness"] for r in results]), 2)
        use = round(np.mean([r["usefulness"] for r in results]), 2)
        total = round(np.mean([(r["relevance"]+r["completeness"]+r["usefulness"])/3 for r in results]), 2)
        low_rel = sum(1 for r in results if r["relevance"] <= 2)
        low_com = sum(1 for r in results if r["completeness"] <= 2)
        low_use = sum(1 for r in results if r["usefulness"] <= 2)
        summary[sname] = {
            "样本数": n,
            "相关性(avg)": rel,
            "完整性(avg)": com,
            "有用性(avg)": use,
            "总分(avg)": total,
            "低分告警_相关性": f"{low_rel}/{n} = {low_rel/n:.1%}",
            "低分告警_完整性": f"{low_com}/{n} = {low_com/n:.1%}",
            "低分告警_有用性": f"{low_use}/{n} = {low_use/n:.1%}",
        }
        print(f"\n{sname}: n={n}, 相关性={rel}, 完整性={com}, 有用性={use}, 总分={total}")

    report_md = f"""# LLM-as-Judge 评测报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
回复模型: {GEN_MODEL}
评测模型: {EVAL_MODEL}

## 数据概览
| 项目 | 数值 |
|---|---|
| Query 总数 | {len(qi_list)} 条 |
| 上下文策略 | top-3 父块（去重） |

## 各策略评分对比
| 策略 | 样本数 | 相关性 | 完整性 | 有用性 | 总分 |
|---|---|---|---|---|---|
"""
    for sname, s in summary.items():
        if "error" in s:
            continue
        report_md += f"| {sname} | {s['样本数']} | {s['相关性(avg)']} | {s['完整性(avg)']} | {s['有用性(avg)']} | {s['总分(avg)']} |\n"

    report_md += "\n## 低分告警统计（评分 ≤ 2）\n| 策略 | 相关性 | 完整性 | 有用性 |\n|---|---|---|---|\n"
    for sname, s in summary.items():
        if "error" in s:
            continue
        report_md += f"| {sname} | {s['低分告警_相关性']} | {s['低分告警_完整性']} | {s['低分告警_有用性']} |\n"

    report_md += "\n## 逐条详细分数\n| # | Query | V-相关 | V-完整 | V-有用 | H-相关 | H-完整 | H-有用 | R-相关 | R-完整 | R-有用 |\n|---|---|---|---|---|---|---|---|---|---|---|\n"

    v_dict = {r["query_index"]: r for r in all_results.get("纯向量检索 (vector)", [])}
    h_dict = {r["query_index"]: r for r in all_results.get("混合检索 (BM25+向量+RRF)", [])}
    r_dict = {r["query_index"]: r for r in all_results.get("混合+RRF+Cross-Encoder精排", [])}

    for qi in qi_list:
        v = v_dict.get(qi, {})
        h = h_dict.get(qi, {})
        r = r_dict.get(qi, {})
        q_text = index[qi]["query"][:30]
        report_md += f"| {qi} | {q_text}... | {v.get('relevance','-')} | {v.get('completeness','-')} | {v.get('usefulness','-')} | {h.get('relevance','-')} | {h.get('completeness','-')} | {h.get('usefulness','-')} | {r.get('relevance','-')} | {r.get('completeness','-')} | {r.get('usefulness','-')} |\n"

    report_md += "\n---\n低分告警阈值：评分 ≤ 2\n"

    report_path = eval_dir / f"report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n报告: {report_path}")

    for sname in all_raw:
        fname = sname.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_plus_")
        path = eval_dir / f"raw_{fname}_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_raw[sname], f, ensure_ascii=False, indent=2)
        print(f"原始: {path}")

    summary_path = eval_dir / f"summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总: {summary_path}")

    print(f"\n{'='*60}")
    print("评测完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
