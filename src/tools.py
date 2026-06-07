"""工具定义模块。

以独立函数 + ToolSpec 的方式定义 4 个工具，
每个工具返回 ToolResult，包含成功/失败状态和结构化输出。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """工具执行结果。"""

    tool_name: str
    success: bool
    output: str           # 给 LLM 看的文本摘要
    raw: Any = None       # 原始结构化数据（评测用）
    error: str = ""


@dataclass
class ToolSpec:
    """工具规格描述（用于生成 System Prompt）。"""

    name: str
    description: str
    parameters: str       # 参数说明（人类可读）
    fn: Any = field(repr=False)  # 实际可调用对象


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

def _web_search(query: str, max_results: int = 5, timeout: int = 10) -> ToolResult:
    """DuckDuckGo 网页搜索。"""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return ToolResult("web_search", False, "", error="duckduckgo-search 未安装")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return ToolResult("web_search", False, "", error=str(e))

    if not results:
        return ToolResult("web_search", True, f"未找到与 \"{query}\" 相关的结果。", raw=[])

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("body", "")
        url = r.get("href", "")
        formatted.append(f"[{i}] {title}\n    {snippet}\n    URL: {url}")

    output = "\n\n".join(formatted)
    return ToolResult("web_search", True, output, raw=results)


def _python_repl(code: str, timeout: int = 5, max_output_chars: int = 2000) -> ToolResult:
    """在子进程中执行 Python 代码。"""
    # 写临时文件
    try:
        fd, tmp = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        return ToolResult("python_repl", False, "", error=str(e))

    try:
        result = subprocess.run(
            ["python", tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parent.parent),  # 项目根目录
        )
        stdout = result.stdout
        stderr = result.stderr

        output_parts = []
        if stdout:
            output_parts.append(stdout.rstrip())
        if stderr:
            output_parts.append(f"[stderr]\n{stderr.rstrip()}")

        output = "\n".join(output_parts) if output_parts else "(无输出)"

        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n... (输出已截断)"

        return ToolResult(
            "python_repl",
            True,
            output,
            raw={"stdout": stdout, "stderr": stderr, "returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult("python_repl", False, f"执行超时（>{timeout}s）", error="timeout")
    except Exception as e:
        return ToolResult("python_repl", False, traceback.format_exc(), error=str(e))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _file_reader(
    path: str,
    allowed_dirs: Optional[List[str]] = None,
    max_size_kb: int = 512,
    allowed_exts: Optional[List[str]] = None,
) -> ToolResult:
    """安全读取本地文件。"""
    if allowed_dirs is None:
        allowed_dirs = ["./data", "./artifacts"]
    if allowed_exts is None:
        allowed_exts = [".txt", ".md", ".json", ".csv", ".py", ".yaml", ".log"]

    repo_root = Path(__file__).resolve().parent.parent
    resolved = (repo_root / path).resolve()

    # 安全检查：必须在白名单目录内
    in_allowed = False
    for ad in allowed_dirs:
        allowed_path = (repo_root / ad).resolve()
        try:
            resolved.relative_to(allowed_path)
            in_allowed = True
            break
        except ValueError:
            continue

    if not in_allowed:
        return ToolResult(
            "file_reader",
            False,
            "",
            error=f"路径不在允许范围内。允许的目录: {allowed_dirs}",
        )

    # 扩展名检查
    suffix = resolved.suffix.lower()
    if allowed_exts and suffix not in allowed_exts:
        return ToolResult(
            "file_reader",
            False,
            "",
            error=f"不支持的文件类型 '{suffix}'。允许: {allowed_exts}",
        )

    if not resolved.exists():
        return ToolResult("file_reader", False, "", error=f"文件不存在: {path}")

    if not resolved.is_file():
        return ToolResult("file_reader", False, "", error=f"不是文件: {path}")

    # 大小检查
    size_kb = resolved.stat().st_size / 1024
    if size_kb > max_size_kb:
        return ToolResult(
            "file_reader",
            False,
            "",
            error=f"文件过大 ({size_kb:.0f}KB > {max_size_kb}KB)",
        )

    try:
        content = resolved.read_text(encoding="utf-8")
        # 截断
        max_chars = max_size_kb * 1024
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... (文件过大，已截断至 {max_size_kb}KB)"
        return ToolResult("file_reader", True, content, raw={"path": str(resolved), "size_kb": size_kb})
    except UnicodeDecodeError:
        return ToolResult("file_reader", False, "", error="无法以 UTF-8 解码（可能是二进制文件）")
    except Exception as e:
        return ToolResult("file_reader", False, "", error=str(e))


# ---------------------------------------------------------------------------
# 工具注册表
# ---------------------------------------------------------------------------

class ToolRegistry:
    """管理所有可用工具，提供查询、调用、生成 System Prompt 的功能。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._config = config or {}

        # ----- 注册 web_search -----
        ws_cfg = self._config.get("tools", {}).get("web_search", {})
        if ws_cfg.get("enabled", True):
            max_results = int(ws_cfg.get("max_results", 5))
            timeout = int(ws_cfg.get("timeout", 10))

            def ws(query: str, _mr=max_results, _to=timeout) -> ToolResult:
                return _web_search(query, max_results=_mr, timeout=_to)

            self._tools["web_search"] = ToolSpec(
                name="web_search",
                description="搜索互联网，获取最新信息。适用于：事实查询、新闻、技术文档、概念解释。",
                parameters="query: 搜索关键词（中英文均可，建议精简为 3-8 个词）",
                fn=ws,
            )

        # ----- 注册 python_repl -----
        pr_cfg = self._config.get("tools", {}).get("python_repl", {})
        if pr_cfg.get("enabled", True):
            to = int(pr_cfg.get("timeout", 5))
            mc = int(pr_cfg.get("max_output_chars", 2000))

            def pr(code: str, _to=to, _mc=mc) -> ToolResult:
                return _python_repl(code, timeout=_to, max_output_chars=_mc)

            self._tools["python_repl"] = ToolSpec(
                name="python_repl",
                description="执行 Python 代码（沙箱子进程，5 秒超时）。适用于：数学计算、数据分析、字符串处理、文件格式转换。",
                parameters="code: 完整的 Python 代码字符串（可使用 math、json、re、collections 等标准库）",
                fn=pr,
            )

        # ----- 注册 vector_search -----
        vs_cfg = self._config.get("tools", {}).get("vector_search", {})
        if vs_cfg.get("enabled", True):
            index_dir = vs_cfg.get("index_dir", "./data/vector_index")
            embed_model = vs_cfg.get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            top_k = int(vs_cfg.get("top_k", 5))
            min_score = float(vs_cfg.get("min_score", 0.3))

            def vs(query: str, _id=index_dir, _em=embed_model, _tk=top_k, _ms=min_score) -> ToolResult:
                try:
                    from .retriever import VectorRetriever
                    retriever = VectorRetriever(
                        index_dir=_id,
                        embed_model=_em,
                        top_k=_tk,
                        min_score=_ms,
                    )
                    retriever.ensure_index()
                    chunks = retriever.search(query, top_k=_tk)
                    if not chunks:
                        return ToolResult("vector_search", True, f"未找到与 \"{query}\" 相关的文档。", raw=[])
                    formatted = []
                    for i, c in enumerate(chunks, 1):
                        score = c.get("score", 0)
                        title = c.get("title", "")
                        text = c.get("text", "")[:300]
                        formatted.append(f"[{i}] {title} (score={score:.3f})\n    {text}")
                    output = "\n\n".join(formatted)
                    return ToolResult("vector_search", True, output, raw=chunks)
                except Exception as e:
                    return ToolResult("vector_search", False, "", error=str(e))

            self._tools["vector_search"] = ToolSpec(
                name="vector_search",
                description="搜索内置知识库中的 AI/ML 相关文档。适用于：技术概念解释、架构原理查询。",
                parameters="query: 搜索查询（中英文均可，建议精简为 3-10 个词）",
                fn=vs,
            )

        # ----- 注册 file_reader -----
        fr_cfg = self._config.get("tools", {}).get("file_reader", {})
        if fr_cfg.get("enabled", True):
            allowed_dirs = fr_cfg.get("allowed_dirs", ["./data", "./artifacts"])
            max_kb = int(fr_cfg.get("max_file_size_kb", 512))
            allowed_exts = fr_cfg.get("allowed_extensions", [".txt", ".md", ".json", ".csv", ".py", ".yaml", ".log"])

            def fr(path: str, _ad=allowed_dirs, _mk=max_kb, _ae=allowed_exts) -> ToolResult:
                return _file_reader(path, allowed_dirs=_ad, max_size_kb=_mk, allowed_exts=_ae)

            self._tools["file_reader"] = ToolSpec(
                name="file_reader",
                description="读取本地文本文件内容。适用于：查看已下载的文档、数据文件、代码文件。",
                parameters="path: 相对于项目根目录的文件路径（如 ./data/sample.txt）",
                fn=fr,
            )

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """按名称调用工具。"""
        spec = self._tools.get(tool_name)
        if spec is None:
            return ToolResult(tool_name, False, "", error=f"未知工具: {tool_name}。可用: {list(self._tools)}")
        try:
            return spec.fn(**kwargs)
        except TypeError as e:
            return ToolResult(tool_name, False, "", error=f"参数错误: {e}")
        except Exception as e:
            return ToolResult(tool_name, False, "", error=f"{type(e).__name__}: {e}")

    def system_prompt_tools(self) -> str:
        """生成嵌入 System Prompt 的工具说明。"""
        lines = ["## 可用工具\n"]
        for t in self._tools.values():
            lines.append(f"### {t.name}")
            lines.append(f"功能: {t.description}")
            lines.append(f"参数: {t.parameters}")
            lines.append("")
        return "\n".join(lines)
