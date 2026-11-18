"""OpenAI 兼容接口的大模型客户端。

通义千问（DashScope 兼容模式）、Ollama、llama.cpp server、vLLM 等都支持
``POST {base_url}/chat/completions``，所以一个实现就能覆盖“云端 + 本地”两种部署。
只依赖 Python 标准库（urllib），无需安装 openai SDK。

如果以后想换官方 SDK / 加缓存 / 加流式输出，只需新增一个 ``LLMClient`` 实现，
业务层代码不用动。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..errors import LLMError
from .base import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAICompatConfig:
    """OpenAI 兼容接口配置。"""

    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 128

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url 不能为空")
        if not self.model.strip():
            raise ValueError("model 不能为空")
        if self.timeout <= 0:
            raise ValueError(f"timeout 必须大于 0：{self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries 不能为负数：{self.max_retries}")


class _NonRetryableLLMError(LLMError):
    """鉴权/参数错误：重试没有意义。"""


class OpenAICompatLLM(LLMClient):
    """通过 OpenAI 兼容的 ``/chat/completions`` 接口调用大模型。"""

    name = "openai-compat"

    def __init__(self, config: OpenAICompatConfig) -> None:
        self.config = config
        self._chat_url = config.base_url.rstrip("/") + "/chat/completions"

    def describe(self) -> str:
        key_state = "已配置" if self.config.api_key else "未配置"
        return (
            f"OpenAI 兼容接口（{self.config.base_url}，模型 {self.config.model}，"
            f"API Key {key_state}）"
        )

    def complete(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error: LLMError | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                content = self._request_once(data, headers)
                logger.debug("大模型原始输出：%s", content)
                return content
            except _NonRetryableLLMError as exc:
                raise LLMError(str(exc)) from exc
            except LLMError as exc:
                last_error = exc
                if attempt <= self.config.max_retries:
                    wait_seconds = min(2 ** (attempt - 1), 4.0)
                    logger.warning(
                        "大模型调用失败（第 %d 次）：%s，%.1f 秒后重试",
                        attempt,
                        exc,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
        raise last_error or LLMError("调用大模型失败：未知错误")

    def _request_once(self, data: bytes, headers: dict[str, str]) -> str:
        request = urllib.request.Request(
            self._chat_url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001 - 读错误响应体失败不影响主流程
                detail = ""
            message = f"大模型接口返回 HTTP {exc.code}：{detail or exc.reason}"
            if exc.code in (400, 401, 403, 404, 422):
                raise _NonRetryableLLMError(message) from exc
            raise LLMError(message) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"调用大模型失败：{exc}") from exc
        return self._extract_content(body)

    def _extract_content(self, body: str) -> str:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(f"大模型响应不是合法 JSON：{body[:200]!r}") from exc
        try:
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"大模型响应结构异常：{body[:200]!r}") from exc
        if not isinstance(content, str):
            raise LLMError(f"大模型响应 content 不是字符串：{content!r}")
        return content