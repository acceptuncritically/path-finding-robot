"""配置加载：环境变量 + 可选 ``.env`` 文件。

所有配置集中在这里读取，业务代码不直接碰 ``os.environ``。
优先级：**系统环境变量 > .env 文件 > 代码默认值**。

支持的环境变量
--------------
大模型：
- ``BRAIN_LLM_PROVIDER``    ``mock``（默认，离线）或 ``openai``（任何 OpenAI 兼容接口）
- ``BRAIN_LLM_BASE_URL``    例如通义千问 ``https://dashscope.aliyuncs.com/compatible-mode/v1``
- ``BRAIN_LLM_API_KEY``     未设置时会回退读取 ``DASHSCOPE_API_KEY`` / ``OPENAI_API_KEY``
- ``BRAIN_LLM_MODEL``       例如 ``qwen-plus``、``qwen2.5:7b``
- ``BRAIN_LLM_TIMEOUT``     请求超时（秒）
- ``BRAIN_LLM_MAX_RETRIES`` 网络失败重试次数
- ``BRAIN_LLM_TEMPERATURE`` / ``BRAIN_LLM_MAX_TOKENS``

通信：
- ``BRAIN_TRANSPORT``    ``mock``（默认）/ ``tcp`` / ``serial``
- ``BRAIN_TCP_HOST`` / ``BRAIN_TCP_PORT``
- ``BRAIN_SERIAL_PORT`` / ``BRAIN_BAUDRATE``
- ``BRAIN_CONNECT_TIMEOUT``
- ``BRAIN_REPORT_TIMEOUT``  等待 C 端回包超时（秒）

日志：
- ``BRAIN_LOG_FILE``（可选）/ ``BRAIN_VERBOSE``（1/true 开启调试日志）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 9000
DEFAULT_SERIAL_PORT = "COM3"
DEFAULT_BAUDRATE = 115200

_LLM_PROVIDERS = ("mock", "openai")
_TRANSPORT_KINDS = ("mock", "tcp", "serial")


# --------------------------------------------------------------------------- #
# 读取 .env / 环境变量
# --------------------------------------------------------------------------- #
def read_env_file(path: Path) -> dict[str, str]:
    """读取简单的 ``KEY=VALUE`` 文件；文件不存在返回空 dict。"""
    if not path.exists():
        return {}
    try:
        # utf-8-sig：兼容带 BOM 的 .env 文件（Windows 记事本常见）
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"读取环境变量文件失败：{path}（{exc}）") from exc

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _merged_env(
    env: Mapping[str, str] | None,
    env_file: Path | None,
) -> dict[str, str]:
    file_values = read_env_file(env_file or DEFAULT_ENV_FILE)
    os_values = dict(os.environ) if env is None else dict(env)
    merged = dict(file_values)
    merged.update(os_values)  # 系统环境变量优先
    return merged


def _get_str(env: Mapping[str, str], key: str, default: str = "") -> str:
    value = env.get(key)
    if value is None:
        return default
    stripped = str(value).strip()
    return stripped if stripped else default


def _get_optional_str(env: Mapping[str, str], key: str) -> str | None:
    value = _get_str(env, key)
    return value or None


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"{key} 必须是数字，实际为 {raw!r}") from exc


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"{key} 必须是整数，实际为 {raw!r}") from exc


def _get_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")


# --------------------------------------------------------------------------- #
# 配置数据类（不可变；需要修改时用 dataclasses.replace）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LLMConfig:
    """大模型客户端配置。"""

    provider: str = "mock"
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_LLM_MODEL
    timeout: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 128

    def __post_init__(self) -> None:
        if self.provider not in _LLM_PROVIDERS:
            raise ConfigError(
                f"BRAIN_LLM_PROVIDER 必须是 {_LLM_PROVIDERS} 之一，实际为 {self.provider!r}"
            )
        if self.timeout <= 0:
            raise ConfigError(f"BRAIN_LLM_TIMEOUT 必须大于 0，实际为 {self.timeout}")
        if self.max_retries < 0:
            raise ConfigError(f"BRAIN_LLM_MAX_RETRIES 不能为负数：{self.max_retries}")
        if self.max_tokens <= 0:
            raise ConfigError(f"BRAIN_LLM_MAX_TOKENS 必须大于 0：{self.max_tokens}")

    def describe(self) -> str:
        if self.provider == "mock":
            return "MockLLM（离线规则解析，无需 API Key）"
        key_state = "已配置" if self.api_key else "未配置"
        return f"OpenAI 兼容接口：{self.base_url} / 模型 {self.model} / API Key {key_state}"


@dataclass(frozen=True)
class TransportConfig:
    """通信方式配置。"""

    kind: str = "mock"
    host: str = DEFAULT_TCP_HOST
    port: int = DEFAULT_TCP_PORT
    serial_port: str = DEFAULT_SERIAL_PORT
    baudrate: int = DEFAULT_BAUDRATE
    connect_timeout: float = 3.0

    def __post_init__(self) -> None:
        if self.kind not in _TRANSPORT_KINDS:
            raise ConfigError(
                f"BRAIN_TRANSPORT 必须是 {_TRANSPORT_KINDS} 之一，实际为 {self.kind!r}"
            )
        if not (1 <= self.port <= 65535):
            raise ConfigError(f"TCP 端口必须在 1~65535 之间，实际为 {self.port}")
        if self.baudrate <= 0:
            raise ConfigError(f"串口波特率必须大于 0，实际为 {self.baudrate}")
        if self.connect_timeout <= 0:
            raise ConfigError(f"连接超时必须大于 0，实际为 {self.connect_timeout}")


@dataclass(frozen=True)
class AppConfig:
    """应用总配置。"""

    llm: LLMConfig
    transport: TransportConfig
    report_timeout: float = 5.0
    log_file: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.report_timeout <= 0:
            raise ConfigError(f"BRAIN_REPORT_TIMEOUT 必须大于 0，实际为 {self.report_timeout}")

    # -- 便捷的“带覆盖”方法，CLI 参数覆盖时使用 ---------------------------- #
    def with_llm(self, **changes: object) -> "AppConfig":
        return replace(self, llm=replace(self.llm, **changes))

    def with_transport(self, **changes: object) -> "AppConfig":
        return replace(self, transport=replace(self.transport, **changes))

    def with_app(self, **changes: object) -> "AppConfig":
        return replace(self, **changes)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        env_file: str | Path | None = None,
    ) -> "AppConfig":
        """从环境变量 / ``.env`` 构建配置。"""
        file_path = Path(env_file) if env_file is not None else None
        merged = _merged_env(env, file_path)

        api_key = (
            _get_str(merged, "BRAIN_LLM_API_KEY")
            or _get_str(merged, "DASHSCOPE_API_KEY")
            or _get_str(merged, "OPENAI_API_KEY")
        )
        provider = _get_str(merged, "BRAIN_LLM_PROVIDER").lower()
        if not provider:
            provider = "openai" if api_key else "mock"

        llm = LLMConfig(
            provider=provider,
            base_url=_get_str(merged, "BRAIN_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            api_key=api_key,
            model=_get_str(merged, "BRAIN_LLM_MODEL", DEFAULT_LLM_MODEL),
            timeout=_get_float(merged, "BRAIN_LLM_TIMEOUT", 30.0),
            max_retries=_get_int(merged, "BRAIN_LLM_MAX_RETRIES", 2),
            temperature=_get_float(merged, "BRAIN_LLM_TEMPERATURE", 0.0),
            max_tokens=_get_int(merged, "BRAIN_LLM_MAX_TOKENS", 128),
        )
        transport = TransportConfig(
            kind=_get_str(merged, "BRAIN_TRANSPORT", "mock").lower(),
            host=_get_str(merged, "BRAIN_TCP_HOST", DEFAULT_TCP_HOST),
            port=_get_int(merged, "BRAIN_TCP_PORT", DEFAULT_TCP_PORT),
            serial_port=_get_str(merged, "BRAIN_SERIAL_PORT", DEFAULT_SERIAL_PORT),
            baudrate=_get_int(merged, "BRAIN_BAUDRATE", DEFAULT_BAUDRATE),
            connect_timeout=_get_float(merged, "BRAIN_CONNECT_TIMEOUT", 3.0),
        )
        return cls(
            llm=llm,
            transport=transport,
            report_timeout=_get_float(merged, "BRAIN_REPORT_TIMEOUT", 5.0),
            log_file=_get_optional_str(merged, "BRAIN_LOG_FILE"),
            verbose=_get_bool(merged, "BRAIN_VERBOSE", False),
        )