"""向量检索工具。

基于 FAISS + Sentence-Transformers 的文档向量检索。
首次使用时会自动构建索引。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class VectorRetriever:
    """轻量向量检索器，FAISS FlatIP + Sentence-Transformers。"""

    def __init__(
        self,
        index_dir: str = "./data/vector_index",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model
        self.top_k = top_k
        self.min_score = min_score

        self._model = None
        self._index = None
        self._chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def ensure_index(self) -> None:
        """确保索引已加载；若不存在则自动构建。"""
        if self.is_ready:
            return
        faiss_path = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "chunks.json"
        if faiss_path.exists() and chunks_path.exists():
            self._load(faiss_path, chunks_path)
        else:
            self._build_and_save()

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """检索与 query 最相似的 Top-K 文档块。"""
        self.ensure_index()

        if self._index is None or not self._chunks:
            return []

        k = top_k or self.top_k
        try:
            import faiss
        except ImportError:
            return [{"error": "faiss-cpu 未安装"}]

        vec = self._embed([query])
        faiss.normalize_L2(vec)

        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            if float(score) < self.min_score:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embed_model_name)
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def _load(self, faiss_path: Path, chunks_path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(faiss_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

    def _build_and_save(self) -> None:
        """从默认知识文档构建索引。"""
        import faiss

        docs = _default_knowledge_docs()
        if not docs:
            return

        self._chunks = []
        for doc in docs:
            title = doc.get("title", "")
            content = doc.get("content", "")
            # 按段落切分
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for p in paragraphs:
                self._chunks.append({
                    "title": title,
                    "text": p,
                    "source": doc.get("source", ""),
                })

        if not self._chunks:
            return

        texts = [c["text"] for c in self._chunks]
        embeddings = self._embed(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)

        # 持久化
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 默认知识文档（作为嵌入的知识库种子）
# ------------------------------------------------------------------

def _default_knowledge_docs() -> List[Dict[str, str]]:
    """返回一组默认的 AI/ML 知识文档，供向量检索使用。"""
    return [
        {
            "title": "Transformer 模型架构",
            "source": "builtin",
            "content": """
Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。
核心组件包括：多头自注意力（Multi-Head Self-Attention）、位置编码（Positional Encoding）、
前馈网络（Feed-Forward Network）和层归一化（Layer Normalization）。

Transformer 彻底取代了 RNN/LSTM 在序列建模中的主导地位，成为现代大语言模型的基础架构。
BERT、GPT、T5 等模型均基于 Transformer 架构。

自注意力机制的计算公式为：Attention(Q,K,V) = softmax(QK^T/√d_k)V。
多头注意力将 Q、K、V 分别投影到 h 个不同的子空间，并行计算注意力，再拼接结果。
""",
        },
        {
            "title": "LoRA 低秩适配",
            "source": "builtin",
            "content": """
LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，由 Hu 等人在 2021 年提出。
核心思想：冻结预训练模型的原始权重，在特定层旁路添加低秩分解矩阵（A 和 B），
仅训练这些低秩矩阵，从而大幅减少可训练参数。

典型的 LoRA 配置：rank r=8~16，alpha=16~32，target_modules 包含注意力层的 q、k、v、o 投影
以及 FFN 层的 gate、up、down 投影。

QLoRA 进一步引入 4-bit NormalFloat 量化，使单个 7B 模型的微调在消费级 GPU（如 RTX 3060 8GB）上可运行。
训练完成后，adapter 权重可合并回基座模型（merge_and_unload），不增加推理延迟。
""",
        },
        {
            "title": "RAG 检索增强生成",
            "source": "builtin",
            "content": """
RAG（Retrieval-Augmented Generation）是一种将信息检索与文本生成结合的架构。
核心流程：用户提问 → 从知识库检索相关文档片段 → 将检索结果作为上下文注入 LLM Prompt → LLM 生成回答。

RAG 的主要优势：
1. 知识可更新：无需重新训练模型，更新知识库即可。
2. 可溯源：回答可附带引用来源，降低幻觉风险。
3. 领域适配：可针对特定领域建库，通用 LLM 也能回答专业问题。

常见检索策略：BM25（稀疏/关键词）、Dense Retrieval（稠密/语义向量）、混合检索 + RRF 融合。
精排常用 Cross-Encoder（如 BERT-based reranker）对候选片段做深度相关性判断。
""",
        },
        {
            "title": "LangGraph Agent 框架",
            "source": "builtin",
            "content": """
LangGraph 是 LangChain 团队开发的 Agent 编排框架，基于有向图（StateGraph）定义 Agent 的控制流。
核心理念：将 Agent 的行为建模为节点（Node）和边（Edge）组成的状态机。

节点类型：普通函数节点、LLM 节点、工具节点（ToolNode）。
边类型：普通边（固定路由）、条件边（LLM 判断下一步）。

典型 Agent 模式：
1. ReAct：Think → Act → Observe → Think → ... → Answer
2. Plan-Execute：Plan → Execute All → Summarize
3. Router：根据问题类型路由到不同处理链

LangGraph 支持 checkpoint（状态持久化）、interrupt（人机协同）、streaming（流式输出）等高级特性。
""",
        },
        {
            "title": "中文金融文本信息抽取",
            "source": "builtin",
            "content": """
从中文年报、公告等金融文档中抽取结构化信息面临独特挑战：

1. 单位归一化：中文财报使用"千元"、"万元"、"亿元"等多种口径，
   抽取时必须统一转换为"元"，否则数值比较无意义。
   例如："营业收入 125.80 亿元" 应抽取为 12580000000。

2. 表格解析：A 股年报大量使用跨页表格、合并单元格，
   pdfplumber 等工具在提取时会产生错位、断行等问题。

3. 拒答场景：很多字段（如自由现金流）并非所有公司都披露，
   系统必须能识别"未单独披露"等语义否定，而非强行抽取。

4. 评测需要字段级粒度：17 个字段各有不同难度，macro F1 对低频字段更公平，
   还应引入拒答准确率、单位错误率等专项指标。
""",
        },
    ]
