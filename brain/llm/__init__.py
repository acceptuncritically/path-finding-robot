"""大模型层：客户端抽象 + Mock + OpenAI 兼容实现 + 输出解析。"""

from __future__ import annotations

from .base import LLMClient
from .mock import MockLLM
from .openai_compat import OpenAICompatConfig, OpenAICompatLLM
from .parser import parse_target_from_llm_output

__all__ = [
    "LLMClient",
    "MockLLM",
    "OpenAICompatConfig",
    "OpenAICompatLLM",
    "parse_target_from_llm_output",
]