"""大模型客户端抽象。

业务层（``brain.planner``）只依赖这个抽象，因此：
- 前期用 ``MockLLM`` 离线开发 / 跑单测；
- 联调时换 ``OpenAICompatLLM``（通义千问 / LLaMA / Ollama 都兼容）；
- 将来换成本地 SDK、加缓存、加日志装饰器，都不影响业务层。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """大模型客户端接口：输入系统提示词 + 用户消息，返回**原始**输出文本。"""

    name: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """调用大模型并返回原始文本。

        Raises:
            brain.errors.LLMError: 网络 / 鉴权 / 服务端 / 响应结构异常。
        """
        raise NotImplementedError

    def describe(self) -> str:
        """给日志/启动横幅用的描述。"""
        return self.name