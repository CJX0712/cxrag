"""Generation 模块：抽取式生成（TextRank）+ 可选 LLM 接口。"""
from .extractive import ExtractiveGenerator
from .llm import LlamaCppProvider, LLMProvider, OpenAIProvider

__all__ = ["ExtractiveGenerator", "LLMProvider", "OpenAIProvider", "LlamaCppProvider"]
