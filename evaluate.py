#!/usr/bin/env python
"""Research Assistant Agent — 离线评测脚本。

用法:
    python evaluate.py                     # 跑全部 25 题
    python evaluate.py --ids Q001 Q003 Q005  # 只跑指定题
    python evaluate.py --category multi_tool # 只跑多工具题
    python evaluate.py --report             # 生成评测报告到 artifacts/results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent_graph import ResearchAgent, extract_answer, extract_tool_trace

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    question_id: str
    category: str
    question: str
    answer: str = ""
    tools_used: List[str] = field(default_factory=list)
    tool_count: int = 0
    expected_tools_hit: int = 0
    content_match: int = 0
    elapsed_sec: float = 0.0
    passed: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# 评测逻辑
# ---------------------------------------------------------------------------


def _load_questions(path: str = "data/eval_questions.json") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"评测集不存在: {p}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def evaluate_one(agent: ResearchAgent, q: Dict[str, Any]) -> EvalResult:
    """评测单道题。"""
    t0 = time.perf_counter()
    state = agent.run(q["question"])
    elapsed = time.perf_counter() - t0

    answer = extract_answer(state)
    trace = extract_tool_trace(state)

    tools_used = [t["tool_name"] for t in trace if t["success"]]
    tool_count = len(tools_used)

    # 工具选择得分
    expected = set(q.get("expected_tools", []))
    actual = set(tools_used)
    expected_hit = len(expected & actual) if expected else -1  # -1 = 无期望工具

    # 内容匹配
    keywords = q.get("expected_answer_contains", [])
    content_hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    content_match = content_hits / len(keywords) if keywords else -1.0

    # 是否通过
    min_calls = q.get("min_tool_calls", 0)
    passed = (
        (expected_hit >= 1 if expected else True)
        and tool_count >= min_calls
        and (content_match >= 0.3 if keywords else True)
    )

    return EvalResult(
        question_id=q["id"],
        category=q.get("category", "unknown"),
        question=q["question"],
        answer=answer[:300],  # 截断存储
        tools_used=tools_used,
        tool_count=tool_count,
        expected_tools_hit=expected_hit,
        content_match=round(content_match, 3) if isinstance(content_match, float) else content_match,
        elapsed_sec=round(elapsed, 1),
        passed=passed,
        note=q.get("note", ""),
    )


def print_results(results: List[EvalResult]) -> None:
    """打印评测结果表。"""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*80}")
    print(f"  Research Assistant Agent — 评测结果")
    print(f"  通过: {passed}/{total} ({passed/total*100:.1f}%)" if total else "")
    print(f"{'='*80}\n")

    header = f"{'ID':<6} {'分类':<20} {'通过':<6} {'工具数':<7} {'工具命中':<9} {'内容匹配':<9} {'耗时':<7}"
    print(header)
    print("-" * 80)

    for r in results:
        status = "✅" if r.passed else "❌"
        hit_str = str(r.expected_tools_hit) if r.expected_tools_hit >= 0 else "N/A"
        match_str = f"{r.content_match:.2f}" if isinstance(r.content_match, float) else "N/A"
        print(
            f"{r.question_id:<6} {r.category:<20} {status:<6} {r.tool_count:<7} "
            f"{hit_str:<9} {match_str:<9} {r.elapsed_sec:<6.1f}s"
        )

    print("-" * 80)
    print(f"总结: {passed}/{total} 通过")

    # 分category统计
    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    print("\n按分类:")
    for cat, items in sorted(cats.items()):
        cat_pass = sum(1 for x in items if x.passed)
        print(f"  {cat}: {cat_pass}/{len(items)} ({cat_pass/len(items)*100:.0f}%)")


def save_report(results: List[EvalResult], output_dir: str = "artifacts/results") -> None:
    """保存评测报告。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out / f"eval_{timestamp}.json"
    data = {
        "timestamp": timestamp,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "results": [
            {
                "id": r.question_id,
                "category": r.category,
                "question": r.question,
                "answer": r.answer,
                "tools_used": r.tools_used,
                "tool_count": r.tool_count,
                "passed": r.passed,
                "note": r.note,
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Markdown
    md_path = out / f"eval_{timestamp}.md"
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    lines = [
        f"# Agent 评测报告",
        f"",
        f"- 时间: {timestamp}",
        f"- 通过率: {passed}/{total} ({passed/total*100:.1f}%)" if total else "- 通过率: N/A",
        f"",
        f"## 详细结果",
        f"",
        f"| ID | 分类 | 通过 | 工具数 | 工具命中 | 工具列表 |",
        f"|----|------|------|--------|----------|----------|",
    ]
    for r in results:
        status = "✅" if r.passed else "❌"
        hit_str = str(r.expected_tools_hit) if r.expected_tools_hit >= 0 else "N/A"
        tools_str = ", ".join(r.tools_used) if r.tools_used else "(无)"
        lines.append(
            f"| {r.question_id} | {r.category} | {status} | {r.tool_count} | {hit_str} | {tools_str} |"
        )
    lines.append("")
    lines.append(f"## 按分类统计")
    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    for cat, items in sorted(cats.items()):
        cat_pass = sum(1 for x in items if x.passed)
        lines.append(f"- {cat}: {cat_pass}/{len(items)} ({cat_pass/len(items)*100:.0f}%)")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n评测报告已保存:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Research Assistant Agent 离线评测")
    parser.add_argument("--ids", nargs="*", help="只跑指定 ID 的题目")
    parser.add_argument("--category", "-c", type=str, help="只跑指定分类")
    parser.add_argument("--limit", "-n", type=int, help="最多跑 N 道题")
    parser.add_argument("--report", "-r", action="store_true", help="保存评测报告")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印每题答案摘要")
    args = parser.parse_args()

    all_qs = _load_questions()
    if not all_qs:
        print("未找到评测题目。")
        return

    # 筛选
    if args.ids:
        id_set = set(args.ids)
        qs = [q for q in all_qs if q["id"] in id_set]
    elif args.category:
        qs = [q for q in all_qs if q.get("category") == args.category]
    else:
        qs = list(all_qs)

    if args.limit:
        qs = qs[: args.limit]

    print(f"准备评测 {len(qs)} 道题...\n")

    agent = ResearchAgent()
    results: List[EvalResult] = []

    for i, q in enumerate(qs, 1):
        qid = q["id"]
        cat = q.get("category", "?")
        print(f"[{i}/{len(qs)}] {qid} ({cat}): {q['question'][:60]}...", end=" ", flush=True)

        r = evaluate_one(agent, q)
        results.append(r)

        status = "✅" if r.passed else "❌"
        print(f"{status} | 工具: {r.tools_used or '(无)'} | {r.elapsed_sec:.1f}s")

        if args.verbose and r.answer:
            print(f"     答案: {r.answer[:200]}")

        # 间隔一下，避免 LLM 过载
        if i < len(qs):
            time.sleep(0.3)

    print_results(results)

    if args.report:
        save_report(results)


if __name__ == "__main__":
    main()
