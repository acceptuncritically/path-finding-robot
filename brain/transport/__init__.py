"""传输层：Mock / TCP / 串口 三种实现，统一 ``Transport`` 接口。"""

from __future__ import annotations

from .base import Transport
from .mock import SCENARIOS, MockRobotTransport
from .serial import SerialTransport
from .tcp import TcpTransport

__all__ = [
    "Transport",
    "MockRobotTransport",
    "TcpTransport",
    "SerialTransport",
    "SCENARIOS",
]