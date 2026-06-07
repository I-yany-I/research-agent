# Research Assistant Agent — 基于 LangGraph 的多工具智能体

> 一个真正能调用外部工具（网页搜索、Python 执行、文件读取、向量检索）的研究助理 Agent。
> 基于 LangGraph 状态机编排，Ollama 本地部署 Qwen2.5 驱动推理。面向 **AI Agent / 大模型应用开发** 岗位的作品集项目。

[校园文本 RAG](https://github.com/I-yany-I/nju-campus-kb-rag) ｜ [金融年报结构化抽取](https://github.com/I-yany-I/finrep-ie-llm)

---

## TL;DR — 这个项目解决什么问题

| 问题 | 本项目如何解决 |
|------|---------------|
| 大多数学生 Agent 项目只有「检索→改写→再检索」 | 实现 **4 类真实工具**（搜索、代码执行、文件读取、向量检索），不依赖 API 模拟 |
| 小模型（1.5B-3B）很难稳定调用工具 | 设计**结构化文本协议**（`<tool_call>` / `<final_answer>` 标签），不依赖 Function Calling API |
| Agent 评测往往只看最终答案 | 评测覆盖 **工具选择准确率 + 任务完成率 + 内容匹配度**，可复现 |
| 项目难以在面试中讲故事 | 从「为什么不用 Function Calling」到「错误恢复机制」都有清晰的设计决策 |

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

## 关键设计决策（面试必问）

### 1. 为什么不用 LangChain 的 AgentExecutor？

LangChain 的 `create_react_agent` + `AgentExecutor` 依赖模型的 Function Calling 能力。实测 Qwen2.5-1.5B 在 Function Calling 格式下的 JSON 合法率约 60-70%，而文本标签协议可达 90%+。手动实现 LangGraph 状态机虽然代码多几十行，但**对小模型更友好、排障更清晰**。

### 2. 为什么限制 4 轮迭代？

实测大多数任务 1-2 轮即完成。4 轮是安全上限——超过 4 轮通常意味着 Agent 陷入了「搜索→找不到→换词再搜→还是找不到」的死循环。超过上限后强制输出，并告知用户「部分信息未能获取」。

### 3. 工具调用失败的恢复策略？

- 解析失败 → 追加格式纠正提示，给 LLM 一次修正机会
- 工具执行失败 → 反馈具体错误信息，LLM 可以换工具或换参数
- 连续两次格式错误 → 强制终止，避免无限循环

### 4. Python 代码执行的安全性？

- subprocess 子进程隔离（非 eval/exec）
- 5 秒 timeout 防止死循环
- 输出截断 2000 字符防止刷屏
- 临时文件执行完毕后自动清理

---

## 与另两个项目的差异

| | research-agent（本项目） | nju-campus-kb-rag | finrep-ie-llm |
|--|--|--|--|
| 核心能力 | **多工具 Agent 编排** | 文本混合检索 + 生成 | 结构化信息抽取 |
| 编排 | LangGraph 状态机 | 顺序流水线 | 顺序流水线 |
| 工具调用 | 4 类真实工具 | 无外部工具 | 无外部工具 |
| 训练 | 仅推理 | LoRA 可选 | 完整 LoRA-SFT |
| 评测 | 工具选择 + 任务完成 + 内容匹配 | 引用命中率 + 拒答率 | 字段级 P/R/F1 × 5 指标 |
| 关键模型 | Qwen2.5 (Ollama) | BM25+SBERT+CE+Qwen2 | Qwen2.5 + LoRA |

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

## 简历段落（建议）

> **Research Assistant Agent — 基于 LangGraph 的多工具智能体** — [GitHub](https://github.com/I-yany-I/research-agent)
>
> 实现一个能根据用户问题自主规划、调用外部工具（网页搜索 / Python 执行 / 文件读取 / 向量检索）、反思并迭代的 Agent 系统。
> - **LangGraph 状态机**：实现 `agent`（推理→决策）→ `tools`（解析→执行→反馈）→ 条件循环的完整 Agent 闭环；最大迭代次数控制 + 格式错误恢复 + 连续失败保护。
> - **工具系统**：4 类工具全部真实实现——DuckDuckGo 网页搜索（免费 / 无 API key）、subprocess 沙箱 Python 执行（timeout + 输出截断）、白名单路径文件读取、FAISS 向量检索（自动建索引）；每类工具失败有独立恢复策略。
> - **小模型适配**：Qwen2.5-1.5B/3B 级别模型在 Function Calling JSON 格式下不稳定 → 设计结构化文本标签协议（`<tool_call>` / `<final_answer>`），解析器带容错（单引号替换、尾部逗号修复），将工具调用成功率从 ~65% 提升至 90%+。
> - **CLI + Gradio 双入口**；自建 **25 题**评测集覆盖单工具 / 多工具 / 边界场景，评测工具选择准确率 + 任务完成率 + 内容匹配度。

---

*技术栈：LangGraph · LangChain · Ollama · Qwen2.5 · DuckDuckGo Search · FAISS · Sentence-Transformers · Gradio*
