"""传输层抽象：业务层只认“发一条 / 收一条”，不关心底层是 Mock、TCP 还是串口。

约定
----
- 一行 = 一条 JSON 报文，以 ``\\n`` 结尾；
- ``send_line`` 负责编码（UTF-8）与发送；
- ``recv_line(timeout)`` 返回**一条完整报文**（不含换行），超时返回 ``None``；
- 打开/关闭失败抛 ``TransportError``，连接断开抛 ``TransportClosedError``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Transport(ABC):
    """通信传输接口。"""

    name: str = "base"

    @abstractmethod
    def open(self) -> None:
        """建立连接 / 打开端口。"""

    @abstractmethod
    def close(self) -> None:
        """关闭连接（可重复调用）。"""

    @abstractmethod
    def send_line(self, text: str) -> None:
        """发送一条报文（内部会确保以 ``\\n`` 结尾）。"""

    @abstractmethod
    def recv_line(self, timeout: float) -> str | None:
        """接收一条完整报文；超时返回 ``None``。"""

    @property
    def is_open(self) -> bool:
        """连接是否已打开。"""
        return False

    def drain(self, timeout: float = 0.05, max_messages: int = 20) -> list[str]:
        """清空接收缓冲区里的**残留 / 迟到报文**，返回被丢弃的非空报文。

        使用场景：上一条任务超时 / 协议错误后，C 端可能稍后才把回包送到；
        新任务下发前必须先清掉这些“无主回包”，否则会被误当成新任务的回包，
        造成“目标 A 却展示目标 B 的路径”的静默错配。

        默认实现基于 ``recv_line`` 循环；Mock 传输层会覆盖以保护脚本化回包。
        """
        discarded: list[str] = []
        for _ in range(max_messages):
            line = self.recv_line(timeout)
            if line is None:
                break
            if line.strip():
                discarded.append(line)
        return discarded

    def describe(self) -> str:
        """给启动横幅 / 日志用的描述。"""
        return self.name

    def __enter__(self) -> "Transport":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False