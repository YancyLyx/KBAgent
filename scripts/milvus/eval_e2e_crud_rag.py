# -*- coding: utf-8 -*-
"""Phase 2 任务3：CRUD-RAG 端到端评测（Create/Read/Update，Delete 缺数据跳过）。

- 后端由 VECTOR_STORE_BACKEND 环境变量切换（create_vector_store 支持），产品路径默认 chroma 不变。
  评测用 VECTOR_STORE_BACKEND=milvus + E2E_COLLECTION=crud_rag。
- 检索复用 RAGPipeline.retrieve()，与 A1 同语料（crud_rag 80k + Task2 注入 2218 known_item），不重复注入。
- 抽样：split_merged.json，seed=42，每任务 150。Read 用 questanswer_1doc（与 Task2 注入样本对齐，已知项在库=真RAG）。
- TASK 环境变量控制跑哪些任务（read/create/update），默认全跑。
- Update：先贴 prompt+判定口径确认再跑（本脚本 update 需显式 TASK=update 触发）。
用 miniforge3 python。语料/产物不提交。
"""
import sys
import os
import json
import time
import asyncio
import random
import re
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.rag.rag_pipeline import RAGPipeline

SPLIT = project_root / "data/crud_rag/repo/data/crud_split/split_merged.json"
OUT_DIR = project_root / "data/crud_rag/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = Path(os.getenv("E2E_OUT", str(OUT_DIR / "task3_e2e_results.json")))

N_READ = int(os.getenv("N_READ", "150"))
N_CREATE = int(os.getenv("N_CREATE", "150"))
N_UPDATE = int(os.getenv("N_UPDATE", "150"))
TOP_K = 3
SEED = 42
TASKS = [t.strip() for t in os.getenv("TASK", "read,create,update").split(",") if t.strip()]


def build_read_prompt(q, contexts):
    ctx = "\n\n".join("[文档%d] %s" % (i + 1, c.get("content", "")) for i, c in enumerate(contexts))
    sys_msg = "你是一个严谨的阅读理解助手。请仅基于参考文档回答问题；若文档不足，请说明。中文回答，简洁准确。"
    user_msg = "参考文档：\n%s\n\n问题：%s\n\n请基于参考文档回答。" % (ctx, q)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_create_prompt(beginning, contexts):
    # 语义：检索背景资料，不注入续写源（避免泄露 golden），属"背景增强续写"
    ctx = "\n\n".join("[文档%d] %s" % (i + 1, c.get("content", "")) for i, c in enumerate(contexts))
    sys_msg = "你是一个新闻续写助手。请参考检索到的相关背景资料，为给定新闻开头续写一段连贯、事实合理的后续内容。中文，不要重复开头。"
    user_msg = "背景资料：\n%s\n\n新闻开头：%s\n\n请续写后续内容（约150-300字）。" % (ctx, beginning)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_judge_prompt(task, question, golden, generated):
    crit = {
        "read": "答案与标准答案在事实上一致且完整（5=完全一致，0=完全错误/无关）",
        "create": "续写与标准续写在事实和语义上吻合、连贯合理（5=高度吻合，0=完全偏离）",
    }[task]
    sys_msg = "你是严格的评测裁判。按评分标准打 0-5 整数分，并给一句话理由。输出JSON：{\"score\":int,\"reason\":str}。"
    user_msg = "评分标准：%s\n\n问题/开头：%s\n标准答案/续写：%s\n模型输出：%s\n\n请打分并输出JSON。" % (crit, question, golden, generated)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


async def llm_call(client, model, messages, temperature=0.3, max_tokens=800):
    t = time.time()
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    usage = resp.usage
    return resp.choices[0].message.content, {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }, time.time() - t


def parse_judge(text):
    m = re.search(r"\{[^}]*\}", text, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            return int(j.get("score", 0)), str(j.get("reason", ""))
        except Exception:
            pass
    sm = re.search(r'"score"\s*:\s*(\d+)', text)
    score = int(sm.group(1)) if sm else 0
    rm = re.search(r'"reason"\s*:\s*"([^"]*)', text)
    reason = rm.group(1) if rm else text[:80]
    return score, reason


async def run_read(pipe, samples):
    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for i, s in enumerate(samples):
        q = s.get("questions", "")
        golden = s.get("answers", "")
        if not q or not golden:
            continue
        try:
            contexts = pipe.retrieve(q, top_k=TOP_K, use_rerank=True)
        except Exception:
            contexts = []
        if len(contexts) == 0:
            print("  [read] 中止：样本 %s n_contexts==0" % s.get("ID"))
            return results, total_tokens, "ABORT: 样本 %s n_contexts==0" % s.get("ID")
        gen_msgs = build_read_prompt(q, contexts)
        try:
            gen, gen_u, gen_t = await llm_call(pipe.llm_client, pipe.llm_model, gen_msgs)
        except Exception as e:
            results.append({"ID": s.get("ID"), "error": str(e)[:120]})
            continue
        judge_msgs = build_judge_prompt("read", q, golden, gen)
        try:
            jtext, j_u, j_t = await llm_call(pipe.llm_client, pipe.llm_model, judge_msgs, temperature=0.0, max_tokens=300)
            score, reason = parse_judge(jtext)
            for k in total_tokens:
                total_tokens[k] += j_u[k]
        except Exception as e:
            score, reason = 0, "judge_err:%s" % e
        for k in total_tokens:
            total_tokens[k] += gen_u[k]
        results.append({"ID": s.get("ID"), "task": "read", "score": score, "reason": reason,
                         "generated": gen[:300], "golden": golden[:200], "n_contexts": len(contexts),
                         "gen_time": gen_t, "tokens": gen_u})
        if (i + 1) % 20 == 0:
            sc = [r.get("score", 0) for r in results if "score" in r]
            print("  [read] %d/%d done, avg_score=%.2f" % (i + 1, len(samples), sum(sc) / len(sc)))
    return results, total_tokens, None


async def run_create(pipe, samples):
    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for i, s in enumerate(samples):
        beginning = s.get("beginning", "")
        golden = s.get("continuing", "")
        if not beginning or not golden:
            continue
        try:
            contexts = pipe.retrieve(beginning, top_k=TOP_K, use_rerank=True)
        except Exception:
            contexts = []
        gen_msgs = build_create_prompt(beginning, contexts)
        try:
            gen, gen_u, gen_t = await llm_call(pipe.llm_client, pipe.llm_model, gen_msgs)
        except Exception as e:
            results.append({"ID": s.get("ID"), "error": str(e)[:120]})
            continue
        judge_msgs = build_judge_prompt("create", beginning, golden, gen)
        try:
            jtext, j_u, j_t = await llm_call(pipe.llm_client, pipe.llm_model, judge_msgs, temperature=0.0, max_tokens=300)
            score, reason = parse_judge(jtext)
            for k in total_tokens:
                total_tokens[k] += j_u[k]
        except Exception as e:
            score, reason = 0, "judge_err:%s" % e
        for k in total_tokens:
            total_tokens[k] += gen_u[k]
        results.append({"ID": s.get("ID"), "task": "create", "score": score, "reason": reason,
                         "generated": gen[:300], "golden": golden[:200], "n_contexts": len(contexts),
                         "gen_time": gen_t, "tokens": gen_u})
        if (i + 1) % 20 == 0:
            sc = [r.get("score", 0) for r in results if "score" in r]
            print("  [create] %d/%d done, avg_score=%.2f" % (i + 1, len(samples), sum(sc) / len(sc)))
    return results, total_tokens


# ---------------- Update（重设计：识别 is_correct + 修正 0-5，均不传检索背景） ----------------
def build_update_identify_prompt(headline, news_beginning, news_remainder, hallucinated_cont):
    """Step1 识别：输入 headLine + newsBeginning + newsRemainder + hallucinatedContinuation（无检索背景），
    让 LLM 指出续写中被修改/幻觉的内容（逐条原文）。"""
    sys_msg = ("你是事实核查助手。给定新闻标题、新闻真实开头、新闻真实剩余正文，以及一段可能含幻觉的续写。"
               "请逐句判断续写中哪些内容与真实新闻不符（即幻觉/被篡改内容），"
               "仅输出幻觉句子的原文（逐条列出），不要修正，不要输出无关内容。若无幻觉，输出：无。")
    user_msg = ("新闻标题：%s\n\n新闻真实开头：%s\n\n新闻真实剩余正文：%s\n\n"
                "待核查续写：%s\n\n请列出续写中与真实新闻不符的幻觉句子（逐条原文）。"
                % (headline, news_beginning, news_remainder, hallucinated_cont))
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_update_identify_judge_prompt(hallucinated_mod, identified):
    """Step1 判定：二元 is_correct——模型识别的幻觉 vs 标注 hallucinatedMod 是否指向同一处幻觉（语义一致即 1）。"""
    sys_msg = ("你是严格的评测裁判。判断模型识别出的幻觉内容与标准幻觉标注是否指向同一处幻觉（语义一致即可，"
               "不要求字面完全相同）。输出JSON：{\"is_correct\":bool,\"reason\":str}。"
               "is_correct=true 当且仅当二者指向同一处幻觉；模型指出任意可疑句不算对，必须与标准标注对应。")
    user_msg = "标准幻觉标注（hallucinatedMod）：%s\n\n模型识别的幻觉句子：%s\n\n请判断是否指向同一处幻觉并输出JSON。" % (hallucinated_mod, identified)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_update_correct_prompt(headline, news_beginning, news_remainder):
    """Step2 修正：基于 newsBeginning + newsRemainder（无检索背景）输出修正后的续写。"""
    sys_msg = "你是新闻续写助手。请仅基于新闻真实开头与真实剩余正文，输出一段与真实新闻一致的续写（约150-300字）。中文。"
    user_msg = "新闻标题：%s\n\n新闻真实开头：%s\n\n新闻真实剩余正文：%s\n\n请输出与真实新闻一致的续写。" % (headline, news_beginning, news_remainder)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_update_correct_judge_prompt(real_continuation, corrected):
    """Step2 判定：修正后续写 vs realContinuation，0-5 分。"""
    sys_msg = ("你是严格的评测裁判。比较模型修正后的续写与真实续写，按事实和语义一致度打 0-5 整数分并给一句话理由。"
               "输出JSON：{\"score\":int,\"reason\":str}。5=与真实续写高度一致；3=部分一致；0=完全偏离。")
    user_msg = "真实续写（realContinuation）：%s\n\n模型修正后续写：%s\n\n请打分并输出JSON。" % (real_continuation, corrected)
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def parse_is_correct(text):
    m = re.search(r"\{[^}]*\}", text, re.S)
    if m:
        try:
            j = json.loads(m.group(0))
            return bool(j.get("is_correct", False)), str(j.get("reason", ""))
        except Exception:
            pass
    sm = re.search(r'"is_correct"\s*:\s*(true|false)', text, re.I)
    return (sm.group(1).lower() == "true") if sm else False, text[:80]


async def run_update(pipe, samples):
    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for i, s in enumerate(samples):
        headline = s.get("headLine", "")
        news_beginning = s.get("newsBeginning", "")
        news_remainder = s.get("newsRemainder", "")
        hallucinated_cont = s.get("hallucinatedContinuation", "")
        hallucinated_mod = s.get("hallucinatedMod", "")
        real_continuation = s.get("realContinuation", "")
        if not (news_beginning and news_remainder and hallucinated_cont and hallucinated_mod and real_continuation):
            continue
        # 检索背景仅记录，不传入识别/修正（任务自包含）
        try:
            contexts = pipe.retrieve(news_beginning, top_k=TOP_K, use_rerank=True)
        except Exception:
            contexts = []
        # Step1 识别
        id_msgs = build_update_identify_prompt(headline, news_beginning, news_remainder, hallucinated_cont)
        try:
            identified, id_u, id_t = await llm_call(pipe.llm_client, pipe.llm_model, id_msgs, temperature=0.0, max_tokens=400)
        except Exception as e:
            results.append({"ID": s.get("ID"), "error": str(e)[:120]})
            continue
        for k in total_tokens:
            total_tokens[k] += id_u[k]
        # Step1 判定 is_correct
        idj_msgs = build_update_identify_judge_prompt(hallucinated_mod, identified)
        try:
            idj_text, idj_u, _ = await llm_call(pipe.llm_client, pipe.llm_model, idj_msgs, temperature=0.0, max_tokens=200)
            is_correct, idj_reason = parse_is_correct(idj_text)
            for k in total_tokens:
                total_tokens[k] += idj_u[k]
        except Exception as e:
            is_correct, idj_reason = False, "judge_err:%s" % e
        # Step2 修正
        corr_msgs = build_update_correct_prompt(headline, news_beginning, news_remainder)
        try:
            corrected, corr_u, _ = await llm_call(pipe.llm_client, pipe.llm_model, corr_msgs, temperature=0.3, max_tokens=600)
            for k in total_tokens:
                total_tokens[k] += corr_u[k]
        except Exception as e:
            results.append({"ID": s.get("ID"), "error": str(e)[:120], "is_correct": is_correct})
            continue
        # Step2 判定 0-5
        corrj_msgs = build_update_correct_judge_prompt(real_continuation, corrected)
        try:
            corrj_text, corrj_u, _ = await llm_call(pipe.llm_client, pipe.llm_model, corrj_msgs, temperature=0.0, max_tokens=300)
            corr_score, corr_reason = parse_judge(corrj_text)
            for k in total_tokens:
                total_tokens[k] += corrj_u[k]
        except Exception as e:
            corr_score, corr_reason = 0, "judge_err:%s" % e
        results.append({"ID": s.get("ID"), "task": "update",
                         "is_correct": is_correct, "idj_reason": idj_reason,
                         "correct_score": corr_score, "corr_reason": corr_reason,
                         "identified": identified[:300], "corrected": corrected[:300],
                         "hallucinatedMod": hallucinated_mod[:200], "realContinuation": real_continuation[:200],
                         "n_contexts": len(contexts), "tokens": id_u})
        if (i + 1) % 20 == 0:
            n = len(results)
            icr = sum(1 for r in results if r.get("is_correct")) / n
            cs = [r.get("correct_score", 0) for r in results if "correct_score" in r]
            cavg = sum(cs) / len(cs) if cs else 0
            print("  [update] %d/%d done, is_correct_rate=%.2f correct_avg=%.2f" % (i + 1, len(samples), icr, cavg))
    return results, total_tokens


def main():
    random.seed(SEED)
    t0 = time.time()
    data = json.load(open(SPLIT, encoding="utf-8"))
    collection = os.getenv("E2E_COLLECTION", "crud_rag")
    backend = os.getenv("VECTOR_STORE_BACKEND", "(未设置,默认chroma)")
    print("=" * 60)
    print("Task 3 端到端评测")
    print("=" * 60)
    print("VECTOR_STORE_BACKEND=%s  E2E_COLLECTION=%s" % (backend, collection))
    print("TASKS=%s  seed=%d" % (TASKS, SEED))
    print("初始化 RAGPipeline...")
    pipe = RAGPipeline(config_path="config/rag_config.yaml", collection_name=collection)
    print("后端确认: backend=%s" % getattr(pipe.vector_store, "backend", "?"))

    all_results = {}
    prev_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # 合并已有结果（支持分任务跑合并到同一份报告）
    if OUT.exists():
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            all_results = prev.get("results", {}) or {}
            prev_cost = prev.get("total_cost", prev_cost)
            print("（合并已有结果，已有任务: %s）" % list(all_results.keys()))
        except Exception:
            pass
    total_cost = dict(prev_cost)
    abort_msg = None

    if "read" in TASKS:
        read_samples = random.sample(data["questanswer_1doc"], min(N_READ, len(data["questanswer_1doc"])))
        print("\n=== Read (questanswer_1doc) 评测 %d 条 ===" % len(read_samples))
        r, tok, abort = asyncio.run(run_read(pipe, read_samples))
        all_results["read"] = r
        for k in total_cost:
            total_cost[k] += tok[k]
        if abort:
            abort_msg = abort
            print("  ABORT: %s" % abort)
        else:
            sc = [x.get("score", 0) for x in r if "score" in x]
            print("  Read avg_score=%.2f n=%d" % (sum(sc) / len(sc), len(sc)))

    if "create" in TASKS and not abort_msg:
        create_samples = random.sample(data["continuing_writing"], min(N_CREATE, len(data["continuing_writing"])))
        print("\n=== Create (continuing_writing) 评测 %d 条 ===" % len(create_samples))
        r, tok = asyncio.run(run_create(pipe, create_samples))
        all_results["create"] = r
        for k in total_cost:
            total_cost[k] += tok[k]
        sc = [x.get("score", 0) for x in r if "score" in x]
        print("  Create avg_score=%.2f n=%d" % (sum(sc) / len(sc), len(sc)))

    if "update" in TASKS and not abort_msg:
        update_samples = random.sample(data["hallu_modified"], min(N_UPDATE, len(data["hallu_modified"])))
        print("\n=== Update (hallu_modified) 评测 %d 条 ===" % len(update_samples))
        r, tok = asyncio.run(run_update(pipe, update_samples))
        all_results["update"] = r
        for k in total_cost:
            total_cost[k] += tok[k]
        n = len(r)
        icr = sum(1 for x in r if x.get("is_correct")) / n if n else 0
        cs = [x.get("correct_score", 0) for x in r if "correct_score" in x]
        cavg = sum(cs) / len(cs) if cs else 0
        print("  Update is_correct_rate=%.2f correct_avg=%.2f n=%d" % (icr, cavg, n))

    print("\n" + "=" * 60)
    print("Task 3 端到端评测汇总")
    print("=" * 60)
    print("backend=%s collection=%s" % (backend, collection))
    for task in ["read", "create", "update"]:
        r = all_results.get(task, [])
        if task == "update":
            n = len(r)
            icr = sum(1 for x in r if x.get("is_correct")) / n if n else 0
            cs = [x.get("correct_score", 0) for x in r if "correct_score" in x]
            cavg = sum(cs) / len(cs) if cs else 0
            toks = sum(x.get("tokens", {}).get("total_tokens", 0) for x in r if "tokens" in x)
            nctx = [x.get("n_contexts", 0) for x in r if "n_contexts" in x]
            nctx_str = ("min=%d max=%d" % (min(nctx), max(nctx))) if nctx else "n/a"
            print("%-8s n=%d is_correct_rate=%.2f correct_avg=%.2f tokens=%d n_contexts[%s]"
                  % (task, n, icr, cavg, toks, nctx_str))
        else:
            sc = [x.get("score", 0) for x in r if "score" in x]
            toks = sum(x.get("tokens", {}).get("total_tokens", 0) for x in r if "tokens" in x)
            nctx = [x.get("n_contexts", 0) for x in r if "n_contexts" in x]
            nctx_str = ("min=%d max=%d 全>0=%s" % (min(nctx), max(nctx), all(c > 0 for c in nctx))) if nctx else "n/a"
            avg = sum(sc) / len(sc) if sc else 0
            print("%-8s n=%d avg_score=%.2f tokens=%d n_contexts[%s]" % (task, len(sc), avg, toks, nctx_str))
    print("\n总 token 成本: %s" % total_cost)
    print("总耗时: %.1fs" % (time.time() - t0))
    if abort_msg:
        print("ABORT: %s" % abort_msg)

    json.dump({"backend": backend, "collection": collection, "results": all_results,
               "total_cost": total_cost, "abort": abort_msg},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("结果已存: %s" % OUT)


if __name__ == "__main__":
    main()
