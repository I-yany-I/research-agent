"""LLM 客户端抽象。

支持 Ollama（OpenAI 兼容 API）和通用 OpenAI 兼容后端。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """轻量 LLM 客户端，封装 Ollama / OpenAI 兼容 API 的 chat 接口。"""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b-instruct",
        base_url: str = "http://127.0.0.1:11434/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        api_key: str = "ollama",  # Ollama 不需要真实 key，但 client 需要非空值
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout),
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
    ) -> str:
        """发送聊天请求，返回模型回复文本。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            stop: 可选的停止词列表。

        Returns:
            模型回复文本（去除首尾空白）。
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
        except Exception as e:
            return f"[LLM 请求失败] {e}"

        choice = resp.choices[0]
        content = choice.message.content
        return (content or "").strip()

    def generate_with_retry(
        self,
        messages: List[Dict[str, str]],
        retries: int = 2,
        stop: Optional[List[str]] = None,
    ) -> str:
        """带重试的生成，重试时 temperature 逐步提高以增加多样性。"""
        last_err = ""
        for attempt in range(retries + 1):
            try:
                # 重试时略微提高温度
                saved_temp = self.temperature
                if attempt > 0:
                    self.temperature = min(saved_temp + 0.3 * attempt, 0.8)

                result = self.chat(messages, stop=stop)

                self.temperature = saved_temp
                if result and not result.startswith("[LLM 请求失败]"):
                    return result

                last_err = result
            except Exception as e:
                last_err = str(e)
                self.temperature = saved_temp

        return f"[生成失败，已重试 {retries} 次] {last_err}"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """从配置字典创建客户端。"""
        llm_cfg = config.get("llm", {})
        return cls(
            model_name=llm_cfg.get("model_name", "qwen2.5:3b-instruct"),
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:11434/v1"),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
            timeout=int(llm_cfg.get("timeout", 60)),
        )
