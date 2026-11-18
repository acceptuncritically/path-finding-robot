"""大脑端统一异常体系。

分门别类的原因是：上层（会话 / 命令行 / 未来的 Web 接口）需要按异常类型
给出不同的人话反馈，并决定“能不能重试”。所有业务异常都继承 ``BrainError``。
"""

from __future__ import annotations


class BrainError(Exception):
    """大脑端所有业务异常的基类。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
class ConfigError(BrainError):
    """配置缺失或非法（环境变量写错、数值无法解析等）。"""


# --------------------------------------------------------------------------- #
# 大模型调用与输出解析
# --------------------------------------------------------------------------- #
class LLMError(BrainError):
    """大模型调用失败（网络、鉴权、服务端错误、响应结构异常等）。"""


class LLMOutputError(BrainError):
    """大模型输出无法按要求解析成目标坐标。"""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class TargetFormatError(LLMOutputError):
    """大模型输出不是合法的目标 JSON / 缺少字段 / 类型错误（可重试）。"""


class TargetOutOfRangeError(LLMOutputError):
    """大模型输出的坐标超出 0~5 栅格范围（可重试）。"""


class TargetUncertainError(LLMOutputError):
    """大模型明确表示无法确定目标（返回 -1,-1 约定标记，不重试，直接问用户）。"""


class PlanningFailure(BrainError):
    """一次“自然语言 -> 目标坐标”的规划最终失败（已用尽重试）。

    Attributes:
        kind: 失败类别，取值 ``format`` / ``range`` / ``uncertain`` / ``llm``。
        attempts: 实际尝试次数（含第一次）。
        last_raw: 最后一次大模型原始输出。
    """

    def __init__(
        self,
        message: str,
        kind: str,
        attempts: int = 0,
        last_raw: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts
        self.last_raw = last_raw


# --------------------------------------------------------------------------- #
# 通信协议（与 C 端之间的报文）
# --------------------------------------------------------------------------- #
class ProtocolError(BrainError):
    """通信协议错误（报文不符合约定）。"""

    def __init__(self, reason: str, raw: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


class MalformedReportError(ProtocolError):
    """C 端回包不是合法 JSON / 顶层结构错误 / 缺少必填字段。"""


class UnknownStatusError(ProtocolError):
    """C 端回包的 status 不是约定的 ``running`` / ``fail``。"""

    def __init__(self, status: object, raw: str = "") -> None:
        super().__init__(f"未知状态：{status!r}", raw)
        self.status = status


class InvalidReportPathError(ProtocolError):
    """C 端回包的 path 字段非法（类型错误、坐标越界、与 status 不匹配等）。"""


class InvalidTargetMessageError(ProtocolError):
    """下发的 target 指令结构非法（主要供模拟 C 端 / 测试使用）。"""


# --------------------------------------------------------------------------- #
# 传输层
# --------------------------------------------------------------------------- #
class TransportError(BrainError):
    """传输层错误（TCP/串口打开失败、发送失败、连接异常等）。"""


class TransportClosedError(TransportError):
    """连接已断开，且没有更多数据。"""


# --------------------------------------------------------------------------- #
# 会话（状态机）
# --------------------------------------------------------------------------- #
class SessionBusyError(BrainError):
    """上一次任务还没结束就再次提交指令（串行策略的并发保护）。"""


class SessionClosedError(BrainError):
    """会话已关闭后继续使用。"""