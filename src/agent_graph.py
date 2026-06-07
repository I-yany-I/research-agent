"""LangGraph Agent 状态机。

实现 ReAct 风格的 Agent 循环：
agent（推理+决策）→ tools（执行工具）→ agent（观察+再推理）→ ... → final_answer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langgraph.graph import StateGraph, END

from .config import load_config
from .llm_client import LLMClient
from .tools import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Agent 状态
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Agent 状态，在 LangGraph 节点间传递。"""

    query: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 4
    final_answer: str = ""
    tool_call_history: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "messages": list(self.messages),
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "final_answer": self.final_answer,
            "tool_call_history": list(self.tool_call_history),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentState":
        return cls(
            query=d.get("query", ""),
            messages=list(d.get("messages", [])),
            iteration=int(d.get("iteration", 0)),
            max_iterations=int(d.get("max_iterations", 4)),
            final_answer=d.get("final_answer", ""),
            tool_call_history=list(d.get("tool_call_history", [])),
            error=d.get("error", ""),
        )


# ---------------------------------------------------------------------------
# System Prompt 模板
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一个研究助理 Agent，可以通过调用工具来获取信息并回答用户问题。

## 行为规则

1. 分析用户问题，判断需要调用哪些工具。
2. 每次只调用 **一个** 最需要的工具。工具返回结果后，判断信息是否足够。
3. 如果信息不足，继续调用工具（可能需要不同的搜索词或不同的工具）。
4. 已有足够信息时，给出最终答案。
5. 如果经过多次尝试仍无法获取所需信息，诚实告知用户。

{}

## 输出格式

调用工具时，严格按以下格式输出（只输出这个，不要加解释）:

<tool_call>
<name>工具名称</name>
<args>
{{"参数名": "参数值"}}
</args>
</tool_call>

给出最终答案时，严格按以下格式输出:

<final_answer>
你的完整回答（引用工具返回的具体信息）
</final_answer>

现在开始。"""


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """从 LLM 输出中解析工具调用。

    期望格式:
    <tool_call>
    <name>web_search</name>
    <args>{"query": "..."}</args>
    </tool_call>

    Returns:
        (tool_name, args_dict) 或 None
    """
    m = re.search(r"<tool_call>\s*<name>(.*?)</name>\s*<args>(.*?)</args>\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None

    tool_name = m.group(1).strip()
    args_str = m.group(2).strip()

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        # 尝试修复常见错误：单引号替换、尾部逗号
        cleaned = args_str.replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            args = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    if not isinstance(args, dict):
        return None

    return tool_name, args


def _parse_final_answer(text: str) -> Optional[str]:
    """从 LLM 输出中解析最终答案。

    期望格式:
    <final_answer>
    回答内容...
    </final_answer>
    """
    m = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# LangGraph 节点
# ---------------------------------------------------------------------------

class ResearchAgent:
    """研究助理 Agent。

    使用 LangGraph 编排 ReAct 风格的 Agent 循环。
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = load_config(config_path)
        self.llm = LLMClient.from_config(self.config)
        self.tools = ToolRegistry(self.config)

        agent_cfg = self.config.get("agent", {})
        self.max_iterations = int(agent_cfg.get("max_iterations", 4))

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentState:
        """执行 Agent，返回最终状态。"""
        state = AgentState(
            query=query,
            max_iterations=self.max_iterations,
        )
        result = self._graph.invoke(state.to_dict())
        return AgentState.from_dict(result)

    def stream(self, query: str):
        """流式执行，逐步产出中间状态。"""
        state = AgentState(
            query=query,
            max_iterations=self.max_iterations,
        )
        for event in self._graph.stream(state.to_dict()):
            yield event

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(state_schema=Dict[str, Any])

        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tool_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._router,
            {
                "tools": "tools",
                "end": END,
            },
        )
        graph.add_edge("tools", "agent")

        return graph.compile()

    # ------------------------------------------------------------------
    # 节点实现
    # ------------------------------------------------------------------

    def _agent_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 推理节点：调用 LLM 决定下一步动作。"""
        state = AgentState.from_dict(state_dict)

        # 构建消息列表
        if state.iteration == 0:
            # 首轮：system prompt + user query
            tools_desc = self.tools.system_prompt_tools()
            system = _SYSTEM_PROMPT.format(tools_desc)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": state.query},
            ]
        else:
            # 后续轮次：在已有 messages 基础上追加工具结果
            messages = list(state.messages)

        # 调用 LLM
        response = self.llm.generate_with_retry(messages)

        # 更新 message 历史
        messages.append({"role": "assistant", "content": response})

        new_state = {
            "messages": messages,
            "iteration": state.iteration,
        }

        return new_state

    def _tool_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行节点：解析 LLM 输出中的工具调用并执行。"""
        state = AgentState.from_dict(state_dict)

        # 获取最后一条 assistant 消息
        last_msg = ""
        for m in reversed(state.messages):
            if m.get("role") == "assistant":
                last_msg = m.get("content", "")
                break

        parsed = _parse_tool_call(last_msg)
        if parsed is None:
            # 无法解析工具调用 → 记录错误，继续循环让 LLM 修正
            err_msg = (
                "上一条回复格式不正确。请严格按照以下格式输出工具调用：\n\n"
                "<tool_call>\n<name>工具名</name>\n<args>\n{...}\n</args>\n</tool_call>\n\n"
                "或者如果已有足够信息，请用 <final_answer> 标签给出最终答案。"
            )
            messages = list(state.messages)
            messages.append({"role": "user", "content": err_msg})
            return {"messages": messages, "iteration": state.iteration}

        tool_name, args = parsed
        result: ToolResult = self.tools.call(tool_name, **args)

        # 构建工具结果反馈
        if result.success:
            feedback = (
                f"工具 [{tool_name}] 返回结果:\n\n"
                f"{result.output}\n\n"
                f"请判断信息是否足够回答问题。如果足够，用 <final_answer> 标签回答；"
                f"如果不够，继续调用工具。"
            )
        else:
            feedback = (
                f"工具 [{tool_name}] 调用失败: {result.error}\n\n"
                f"请尝试换一个工具、换一个参数、或换一个搜索词。"
            )

        messages = list(state.messages)
        messages.append({"role": "user", "content": feedback})

        history_entry = {
            "iteration": state.iteration + 1,
            "tool_name": tool_name,
            "args": args,
            "success": result.success,
            "output_preview": result.output[:500] if result.success else result.error,
        }

        return {
            "messages": messages,
            "iteration": state.iteration + 1,
            "tool_call_history": state.tool_call_history + [history_entry],
        }

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def _router(self, state_dict: Dict[str, Any]) -> str:
        """决定下一步：继续调用工具 or 结束。"""
        state = AgentState.from_dict(state_dict)

        # 检查最后一条 assistant 消息是否包含 final_answer
        for m in reversed(state.messages):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if _parse_final_answer(content):
                    return "end"
                break

        # 超过最大迭代次数 → 强制结束
        if state.iteration >= state.max_iterations:
            return "end"

        # 检查是否刚刚追加了格式纠正提示（避免死循环）
        # 如果最近两条 user 消息都是格式纠正 → 强制结束
        format_corrections = 0
        for m in state.messages:
            if m.get("role") == "user" and "格式不正确" in m.get("content", ""):
                format_corrections += 1
        if format_corrections >= 2:
            return "end"

        return "tools"


# ---------------------------------------------------------------------------
# 便捷函数：从最终 state 提取答案
# ---------------------------------------------------------------------------

def extract_answer(state: AgentState) -> str:
    """从 Agent 状态中提取最终答案文本。"""
    # 优先从 final_answer 标签获取
    for m in reversed(state.messages):
        if m.get("role") == "assistant":
            ans = _parse_final_answer(m.get("content", ""))
            if ans:
                return ans

    # 没有 final_answer → 返回最后一条 assistant 消息
    for m in reversed(state.messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            # 去掉工具调用部分
            cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
            cleaned = cleaned.strip()
            if cleaned:
                return cleaned

    return "（Agent 未能生成回答）"


def extract_tool_trace(state: AgentState) -> List[Dict[str, Any]]:
    """提取工具调用轨迹（用于调试/评测）。"""
    return list(state.tool_call_history)
