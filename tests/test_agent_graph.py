"""Tests for the Agent graph module: tool call parser, final answer parser, and router logic.

Run from project root:
    python -m pytest tests/test_agent_graph.py -v
"""

import sys
import pytest
from pathlib import Path

# Ensure src/ is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_graph import (
    _parse_tool_call,
    _parse_final_answer,
    AgentState,
    ResearchAgent,
    extract_answer,
    extract_tool_trace,
)


# ---------------------------------------------------------------------------
# _parse_tool_call
# ---------------------------------------------------------------------------

class TestParseToolCall:
    def test_normal_call(self):
        text = """<tool_call>
<name>web_search</name>
<args>{"query": "Nobel Prize 2024"}</args>
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "Nobel Prize 2024"}

    def test_single_quotes_in_args(self):
        """Parser should tolerate single quotes by converting to double."""
        text = """<tool_call>
<name>web_search</name>
<args>{'query': 'Nobel Prize'}</args>
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "Nobel Prize"

    def test_trailing_comma_in_args(self):
        """Parser should remove trailing commas before closing brace."""
        text = """<tool_call>
<name>python_repl</name>
<args>{"code": "print(1+1)", }</args>
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "python_repl"
        assert args["code"] == "print(1+1)"

    def test_extra_whitespace(self):
        text = """<tool_call>
        <name>   web_search   </name>
        <args>
        {"query": "hello world"}
        </args>
        </tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"

    def test_no_tool_call_returns_none(self):
        assert _parse_tool_call("just some random text") is None

    def test_final_answer_not_tool_call(self):
        text = """<final_answer>
The answer is 42.
</final_answer>"""
        assert _parse_tool_call(text) is None

    def test_malformed_json_returns_none(self):
        text = """<tool_call>
<name>web_search</name>
<args>not valid json at all {{{</args>
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is None

    def test_direct_tool_tag_style(self):
        """Parser should support <web_search>{...}</web_search> style output."""
        text = """<web_search>
{"query": "南京 天气"}
</web_search>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "南京 天气"}

    def test_tool_name_in_text_with_args_tag(self):
        """Parser should support natural language tool intent + <args> JSON."""
        text = """调用web_search工具来查找南京今天的天气情况。
<args>
{"query": "南京 天气状况"}
</args>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "南京 天气状况"}

    def test_tool_call_simple_body_style(self):
        """Parser should support <tool_call>tool_name + json</tool_call> body."""
        text = """<tool_call>
web_search
{"query": "2024 Nobel Prize in Physics winner"}
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "2024 Nobel Prize in Physics winner"}

    def test_tool_call_simple_body_with_empty_json(self):
        """Parser should support simple body with empty args object."""
        text = """<tool_call>
python_repl
{}
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "python_repl"
        assert args == {}

    def test_unknown_tool_tag_with_args_infers_web_search(self):
        """Parser should infer web_search from unknown tag when args has query."""
        text = """<RigidbodySearchTool>
<args>
{"query": "2024 Nobel Prize in Physics winner"}
</args>
</RigidbodySearchTool>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "2024 Nobel Prize in Physics winner"}

    def test_plain_tool_and_json_without_tags(self):
        """Parser should support plain 'tool_name + JSON' output."""
        text = """web_search
{"query": "2024 Nobel Prize in Physics winner"}"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "2024 Nobel Prize in Physics winner"}

    def test_empty_name(self):
        text = """<tool_call>
<name></name>
<args>{"q": "test"}</args>
</tool_call>"""
        result = _parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == ""


# ---------------------------------------------------------------------------
# _parse_final_answer
# ---------------------------------------------------------------------------

class TestParseFinalAnswer:
    def test_normal_answer(self):
        text = """<final_answer>
The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton.
</final_answer>"""
        result = _parse_final_answer(text)
        assert result is not None
        assert "John Hopfield" in result

    def test_multiline_answer(self):
        text = """<final_answer>
Here is a multi-line answer:

- Point 1
- Point 2
- Point 3
</final_answer>"""
        result = _parse_final_answer(text)
        assert result is not None
        assert "Point 1" in result
        assert "Point 3" in result

    def test_no_final_answer_returns_none(self):
        assert _parse_final_answer("just some text") is None

    def test_tool_call_not_final_answer(self):
        text = """<tool_call>
<name>web_search</name>
<args>{"query": "test"}</args>
</tool_call>"""
        assert _parse_final_answer(text) is None

    def test_answer_with_whitespace(self):
        text = """<final_answer>

        Answer with surrounding whitespace.

        </final_answer>"""
        result = _parse_final_answer(text)
        assert result is not None
        assert "Answer with surrounding whitespace" in result


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_default_values(self):
        state = AgentState()
        assert state.query == ""
        assert state.messages == []
        assert state.iteration == 0
        assert state.max_iterations == 4
        assert state.final_answer == ""

    def test_to_dict_and_from_dict_roundtrip(self):
        original = AgentState(
            query="test query",
            messages=[{"role": "user", "content": "hello"}],
            iteration=2,
            max_iterations=4,
            final_answer="done",
            tool_call_history=[{"iteration": 1, "tool_name": "web_search"}],
            error="some error",
        )
        d = original.to_dict()
        restored = AgentState.from_dict(d)
        assert restored.query == original.query
        assert restored.iteration == original.iteration
        assert restored.max_iterations == original.max_iterations
        assert restored.final_answer == original.final_answer
        assert restored.error == original.error
        assert len(restored.tool_call_history) == 1

    def test_from_dict_missing_keys(self):
        restored = AgentState.from_dict({})
        assert restored.query == ""
        assert restored.iteration == 0


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    def test_extracts_final_answer_tag(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "what is AI?"},
                {"role": "assistant", "content": "<final_answer>\nAI is artificial intelligence.\n</final_answer>"},
            ]
        )
        ans = extract_answer(state)
        assert "artificial intelligence" in ans

    def test_falls_back_to_last_assistant_message(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
            ]
        )
        ans = extract_answer(state)
        assert "Hi!" in ans

    def test_strips_tool_calls_from_fallback(self):
        state = AgentState(
            messages=[
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "Let me search.\n<tool_call>\n<name>web_search</name>\n<args>{}</args>\n</tool_call>\n\nBased on results: the answer is yes."},
            ]
        )
        ans = extract_answer(state)
        assert "<tool_call>" not in ans
        assert "the answer is yes" in ans

    def test_empty_state_returns_placeholder(self):
        state = AgentState()
        ans = extract_answer(state)
        assert "未能生成回答" in ans


# ---------------------------------------------------------------------------
# extract_tool_trace
# ---------------------------------------------------------------------------

class TestExtractToolTrace:
    def test_extracts_history(self):
        state = AgentState(
            tool_call_history=[
                {"iteration": 1, "tool_name": "web_search", "args": {"query": "test"}},
                {"iteration": 2, "tool_name": "python_repl", "args": {"code": "1+1"}},
            ]
        )
        trace = extract_tool_trace(state)
        assert len(trace) == 2
        assert trace[0]["tool_name"] == "web_search"
        assert trace[1]["tool_name"] == "python_repl"

    def test_empty_history(self):
        state = AgentState()
        trace = extract_tool_trace(state)
        assert trace == []


# ---------------------------------------------------------------------------
# ResearchAgent internals
# ---------------------------------------------------------------------------

class TestResearchAgentInternals:
    def test_agent_node_preserves_existing_state_fields(self):
        class DummyLLM:
            def generate_with_retry(self, messages):
                return "<final_answer>ok</final_answer>"

        class DummyTools:
            def system_prompt_tools(self):
                return "dummy tools"

        agent = ResearchAgent.__new__(ResearchAgent)
        agent.llm = DummyLLM()
        agent.tools = DummyTools()

        state = AgentState(
            query="test",
            messages=[{"role": "user", "content": "hello"}],
            iteration=1,
            max_iterations=4,
            tool_call_history=[{"iteration": 1, "tool_name": "web_search", "success": True}],
        )

        out = agent._agent_node(state.to_dict())
        assert out.get("tool_call_history") == state.tool_call_history
        assert out.get("query") == "test"

    def test_router_respects_max_iterations_before_tool_call(self):
        agent = ResearchAgent.__new__(ResearchAgent)
        state = AgentState(
            query="test",
            iteration=4,
            max_iterations=4,
            messages=[
                {"role": "assistant", "content": "web_search\n{\"query\": \"x\"}"},
            ],
        )
        decision = agent._router(state.to_dict())
        assert decision == "end"
