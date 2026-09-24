"""LLM 接入接口（可选依赖）。

提供统一 LLMProvider 抽象；内置：
- OpenAIProvider  ：兼容 OpenAI / 任意 OpenAI 兼容网关（需 pip install openai）
- LlamaCppProvider：本地 GGUF 推理（需 pip install llama-cpp-python）
这些为「生成式」增强入口；离线默认链路使用 ExtractiveGenerator，无需任何模型权重。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.errors import CXError, CODES
from ..core.types import Answer, Chunk, Query


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """给定 prompt，返回生成文本。"""


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - 依赖可选
            raise CXError("E_DEP", "未安装 openai，请 pip install -r requirements-llm.txt", cause=e)
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.2
        )
        return resp.choices[0].message.content or ""


class LlamaCppProvider(LLMProvider):
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover - 依赖可选
            raise CXError("E_DEP", "未安装 llama-cpp-python", cause=e)
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)

    def complete(self, prompt: str) -> str:
        out = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])
        return out["choices"][0]["message"]["content"] or ""


def build_llm_prompt(query: str, context: List[Chunk], lang: str = "zh") -> str:
    ctx = "\n\n".join(f"[文档片段 {i+1}]\n{c.text}" for i, c in enumerate(context))
    return (
        f"你是一个严谨的问答助手。仅基于下方「参考文档」作答，"
        f"若资料不足请明说不知道。\n\n参考文档：\n{ctx}\n\n问题：{query}\n\n回答："
    )


class LLMGenerator:
    """把 LLMProvider 适配为 BaseGenerator 风格（generate 接口）。"""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, query: Query, context: List[Chunk]) -> Answer:
        prompt = build_llm_prompt(query.text, context)
        text = self.provider.complete(prompt)
        return Answer(text=text, query=query.text, context=context, generator="llm")
