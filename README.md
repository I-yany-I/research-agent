# Research Assistant Agent — 基于 LangGraph 的多工具智能体

> 一个真正能调用外部工具（网页搜索、Python 执行、文件读取、向量检索）的研究助理 Agent。
> 基于 LangGraph 状态机编排，Ollama 本地部署 Qwen2.5 驱动推理。

---

## 系统架构

```
用户 Query
    │
    ▼
┌─────────────────────────────────────────┐
│              LangGraph 状态机             │
│                                          │
│   ┌──────────┐    tool_call    ┌───────┐ │
│   │  agent   │ ──────────────→ │ tools │ │
│   │ (推理)   │ ←────────────── │(执行) │ │
│   └────┬─────┘   tool_result  └───────┘ │
│        │                                 │
│        │ final_answer                    │
│        ▼                                 │
│   ┌──────────┐                           │
│   │   END    │                           │
│   └──────────┘                           │
└─────────────────────────────────────────┘
    │
    ▼
  最终回答 + 工具调用轨迹
```

### Agent 循环

1. **agent 节点**：LLM 分析当前信息，决定「调用工具」或「给出最终答案」
2. **tools 节点**：解析 LLM 输出中的 `<tool_call>` 标签，执行工具，将结果反馈给 LLM
3. **循环终止条件**：LLM 输出 `<final_answer>` 或达到最大迭代次数（默认 4 轮）

### 工具调用协议

小模型（1.5B-3B）的 Function Calling 不稳定，本项目使用**结构化文本标签**：

```xml
<!-- 调用工具 -->
<tool_call>
<name>web_search</name>
<args>
{"query": "2024 Nobel Prize Physics"}
</args>
</tool_call>

<!-- 给出最终答案 -->
<final_answer>
2024 年诺贝尔物理学奖授予 John J. Hopfield 和 Geoffrey E. Hinton...
</final_answer>
```

**为什么不用 Function Calling？** Qwen2.5-1.5B/3B 级别的模型在输出 JSON function call 时容易产生语法错误（漏引号、多逗号、嵌套错误）。文本标签协议的错误率显著更低，且解析器带有容错处理（单引号替换、尾部逗号修复）。

---

## 工具清单

| 工具 | 实现方式 | 用途 | 特色 |
|------|---------|------|------|
| **web_search** | DuckDuckGo Search（免费，无需 API key） | 事实查询、新闻、技术文档 | 返回标题+摘要+URL |
| **python_repl** | subprocess 子进程执行临时 .py 文件 | 数学计算、数据分析、算法验证 | 5s 超时、输出截断、异常捕获 |
| **file_reader** | 路径白名单 + 扩展名白名单 + 大小限制 | 读取本地文档、数据文件 | 多层安全检查，防止路径遍历攻击 |
| **vector_search** | FAISS FlatIP + Sentence-Transformers | 检索内置 AI/ML 知识文档 | 自动建索引、结果过滤、持久化 |

---

## 快速开始

### 1. 前置条件

- Python 3.10+
- [Ollama](https://ollama.com) 已安装并拉取模型
- 8GB+ RAM（CPU 模式可跑）或 8GB+ VRAM（GPU 模式）

```bash
# 安装 Ollama 并拉取模型（推荐 3B，推理更强）
ollama pull qwen2.5:3b-instruct
# 或使用更小的模型
ollama pull qwen2.5:1.5b-instruct
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 命令行交互

```bash
python run_cli.py
```

示例对话：
```
❓ 2024年诺贝尔物理学奖得主是谁？

🤖 2024 年诺贝尔物理学奖授予了 John J. Hopfield 和 Geoffrey E. Hinton，
以表彰他们在人工神经网络和机器学习方面的基础性发现与发明...
```

### 4. 启动 Gradio 界面

```bash
python app.py
# 浏览器打开 http://localhost:7862
```

### 5. 离线评测

```bash
# 跑全部 25 道评测题
python evaluate.py --report

# 只跑多工具协作题
python evaluate.py --category multi_tool --verbose

# 只跑指定题目
python evaluate.py --ids Q001 Q003 Q007
```

---

## 评测体系

### 评测集设计

自建 **25 道评测题**，覆盖 4 种场景：

| 分类 | 数量 | 示例 |
|------|------|------|
| `single_tool_web` | 12 题 | "2024 年诺贝尔物理学奖得主是谁？" |
| `single_tool_python` | 7 题 | "用 Python 算 1 到 100 的质数之和" |
| `multi_tool` | 4 题 | "搜索比特币价格 + 用 Python 换算人民币" |
| `boundary` | 2 题 | 无意义输入、信息不足场景 |

### 评测指标

| 指标 | 含义 | 为什么重要 |
|------|------|-----------|
| **Tool Selection Accuracy** | Agent 是否调用了正确的工具 | 反映规划能力 |
| **Task Completion Rate** | 答案是否包含期望的关键信息 | 反映端到端效果 |
| **Tool Efficiency** | 实际调用次数 / 最小所需次数 | 反映效率，避免无效循环 |
| **Recovery Rate** | 工具失败后能否换方案成功 | 反映鲁棒性 |

---

## 目录结构

```
research-agent/
├── README.md           # 本文档
├── requirements.txt    # Python 依赖
├── config.yaml         # 全参数中枢（LLM / Agent / Tools / UI）
├── app.py              # Gradio 界面入口
├── run_cli.py          # 命令行入口
├── evaluate.py         # 离线评测脚本
├── data/
│   ├── eval_questions.json   # 评测问题集（25 题）
│   └── vector_index/         # 向量索引（自动生成）
├── src/
│   ├── config.py        # 配置加载
│   ├── llm_client.py    # LLM 客户端（Ollama / OpenAI 兼容）
│   ├── tools.py         # 工具定义 + ToolRegistry
│   ├── retriever.py     # FAISS 向量检索
│   └── agent_graph.py   # LangGraph 状态机（核心）
└── artifacts/
    └── results/         # 评测报告（.json / .md）
```

---

*技术栈：LangGraph · LangChain · Ollama · Qwen2.5 · DuckDuckGo Search · FAISS · Sentence-Transformers · Gradio*
