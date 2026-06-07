#!/usr/bin/env python
"""Research Assistant Agent — Gradio 界面。

用法:
    python app.py
    # 浏览器打开 http://localhost:7862
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.agent_graph import ResearchAgent, extract_answer, extract_tool_trace
from src.config import load_config


# ---------------------------------------------------------------------------
# 全局 agent 实例（进程内复用，避免每次对话都初始化 LLM 客户端）
# ---------------------------------------------------------------------------
_agent: ResearchAgent | None = None


def get_agent() -> ResearchAgent:
    global _agent
    if _agent is None:
        _agent = ResearchAgent()
    return _agent


# ---------------------------------------------------------------------------
# 对话函数
# ---------------------------------------------------------------------------


def chat_fn(message: str, history: list) -> str:
    """Gradio ChatInterface 回调。"""
    if not message.strip():
        return "请输入问题。"

    agent = get_agent()
    state = agent.run(message)

    # 收集工具调用轨迹
    trace = extract_tool_trace(state)
    tool_info = ""
    if trace:
        tool_info = "\n\n---\n**🔧 工具调用轨迹**\n"
        for entry in trace:
            status = "✅" if entry["success"] else "❌"
            tool_info += (
                f"\n{status} **{entry['tool_name']}** "
                f"(`{', '.join(f'{k}={v}' for k, v in entry['args'].items())}`)"
            )

    answer = extract_answer(state)
    return answer + tool_info


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    ui_cfg = cfg.get("ui", {}).get("gradio", {})

    demo = gr.ChatInterface(
        fn=chat_fn,
        title=ui_cfg.get("title", "Research Assistant Agent"),
        description=(
            "**研究助理智能体** — 基于 LangGraph + Ollama (Qwen2.5) 的多工具 Agent。\n\n"
            "支持工具：🔍 网页搜索 | 🐍 Python 代码执行 | 📄 文件读取 | 🔎 向量检索\n\n"
            "试试问：*2024 年诺贝尔物理学奖得主是谁？* 或 *帮我算一下 2 的 20 次方*"
        ),
        theme=ui_cfg.get("theme", "soft"),
        examples=[
            "2024 年诺贝尔物理学奖得主是谁？",
            "用 Python 计算斐波那契数列的前 20 项",
            "什么是 LoRA 微调？它和全参数微调有什么区别？",
            "Transformer 的自注意力机制是如何工作的？",
            "帮我搜索一下最新的 GPT-5 相关信息",
        ],
        fill_height=True,
    )

    port = int(ui_cfg.get("port", 7862))
    share = bool(ui_cfg.get("share", False))

    print(f"启动 Gradio 界面 → http://localhost:{port}")
    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    main()
