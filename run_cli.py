#!/usr/bin/env python
"""Research Assistant Agent — 命令行入口。

用法:
    python run_cli.py                      # 交互式对话
    python run_cli.py --query "什么是LoRA"  # 单次问答
    python run_cli.py --verbose            # 显示工具调用轨迹
"""

from __future__ import annotations

import argparse
import sys

from src.agent_graph import ResearchAgent, extract_answer, extract_tool_trace


def format_tool_trace(trace):
    """格式化工具调用轨迹。"""
    lines = []
    for entry in trace:
        status = "✓" if entry["success"] else "✗"
        lines.append(
            f"  [{entry['iteration']}] {status} {entry['tool_name']}"
            f"({', '.join(f'{k}={v}' for k, v in entry['args'].items())})"
        )
        preview = entry["output_preview"][:120]
        if len(entry["output_preview"]) > 120:
            preview += "..."
        lines.append(f"      → {preview}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Research Assistant Agent (CLI)")
    parser.add_argument("--query", "-q", type=str, help="单次问答（不进入交互模式）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示工具调用轨迹")
    parser.add_argument("--max-iters", type=int, default=4, help="最大工具调用轮次")
    args = parser.parse_args()

    print("=" * 56)
    print("  Research Assistant Agent — 研究助理智能体")
    print("  工具: web_search | python_repl | file_reader")
    print("=" * 56)
    print()

    agent = ResearchAgent()
    agent.max_iterations = args.max_iters

    # --- 单次问答 ---
    if args.query:
        print(f"❓ {args.query}\n")
        state = agent.run(args.query)

        if args.verbose:
            trace = extract_tool_trace(state)
            if trace:
                print("🔧 工具调用轨迹:")
                print(format_tool_trace(trace))
                print()

        answer = extract_answer(state)
        print(f"🤖 {answer}")
        return

    # --- 交互模式 ---
    print("输入问题开始对话，输入 /quit 退出，/verbose 切换轨迹显示。\n")
    verbose = False

    while True:
        try:
            query = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("/quit", "/exit", "/q"):
            print("再见！")
            break
        if query.lower() == "/verbose":
            verbose = not verbose
            print(f"工具轨迹显示: {'开启' if verbose else '关闭'}")
            continue

        print()
        state = agent.run(query)

        if verbose:
            trace = extract_tool_trace(state)
            if trace:
                print("🔧 工具调用轨迹:")
                print(format_tool_trace(trace))
                print()

        answer = extract_answer(state)
        print(f"🤖 {answer}\n")
        print("-" * 56)


if __name__ == "__main__":
    main()
