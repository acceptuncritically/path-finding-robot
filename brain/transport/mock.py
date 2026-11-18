"""Mock 传输层：本地模拟 C 端（机器人躯体底层）。

不用接硬件就能跑通「下发目标 -> 收到 running/fail 路径」整条链路，
并且可以注入各种异常场景（超时 / 坏 JSON / 未知状态 / 发送失败），
用来开发主流程和写单元测试。
"""

from __future__ import annotations

import logging
from typing import Collection, Sequence

from ..errors import ProtocolError, TransportError
from ..grid import DEFAULT_OBSTACLES, DEFAULT_START, find_path
from ..protocol import (
    Target,
    build_fail_report,
    build_running_report,
    parse_target_message,
)
from .base import Transport

logger = logging.getLogger(__name__)

# 可注入的场景
SCENARIOS = (
    "normal",        # 正常：BFS 算路径，成功则 running，目标被挡则 fail
    "fail",          # 强制失败：总是返回 {"status":"fail","path":[]}
    "timeout",       # 不回包（模拟等待超时）
    "bad_json",      # 回一段非法 JSON
    "unknown_status",# 回未知 status（测试向前兼容）
    "empty_path",    # 回 running 但 path=[]（协议歧义，测试异常处理）
    "send_error",    # 发送时抛异常（测试通信异常处理）
)


class MockRobotTransport(Transport):
    """在内存里模拟一个 C 端：收目标 -> BFS 寻路 -> 回报路径。"""

    name = "mock"

    def __init__(
        self,
        obstacles: Collection[tuple[int, int]] | None = None,
        start: tuple[int, int] = DEFAULT_START,
        scenario: str = "normal",
        scripted_reports: Sequence[str | None] | None = None,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario 必须是 {SCENARIOS} 之一：{scenario!r}")
        self.obstacles: frozenset[tuple[int, int]] = (
            frozenset(obstacles) if obstacles is not None else DEFAULT_OBSTACLES
        )
        self.start = (int(start[0]), int(start[1]))
        self.scenario = scenario
        self._scripted_reports: list[str | None] = list(scripted_reports or [])
        self._opened = False
        self._pending_report: str | None = None

        #: 记录所有下发报文（测试断言用）
        self.sent_lines: list[str] = []
        #: 记录成功解析出的目标（测试断言用）
        self.received_targets: list[Target] = []
        #: 记录结构非法的下发报文（正常流程应始终为空）
        self.invalid_commands: list[str] = []

    # ------------------------------------------------------------------ #
    # Transport 接口
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False
        self._pending_report = None

    @property
    def is_open(self) -> bool:
        return self._opened

    def describe(self) -> str:
        return (
            f"Mock 传输层（障碍物 {len(self.obstacles)} 个，起点 {self.start}，"
            f"场景 {self.scenario}）"
        )

    def send_line(self, text: str) -> None:
        self._ensure_open()
        self.sent_lines.append(text)
        logger.info("Mock 传输层收到下发报文：%s", text.strip())

        if self.scenario == "send_error":
            raise TransportError("模拟发送失败（send_error 场景）")

        try:
            target = parse_target_message(text)
        except ProtocolError as exc:
            self.invalid_commands.append(text)
            logger.warning("Mock 传输层收到非法下发报文：%s（%s）", text.strip(), exc)
            self._pending_report = build_fail_report()
            return

        self.received_targets.append(target)
        if self._scripted_reports:
            self._pending_report = None
            return
        self._pending_report = self._make_report(target)

    def recv_line(self, timeout: float) -> str | None:
        self._ensure_open()
        if self._scripted_reports:
            report = self._scripted_reports.pop(0)
            logger.info("Mock 传输层按脚本返回回包：%r", report)
            return report
        report = self._pending_report
        self._pending_report = None
        logger.info("Mock 传输层返回回包：%r", report)
        return report

    def drain(self, timeout: float = 0.05, max_messages: int = 20) -> list[str]:
        # 脚本化回包是测试工具（按 send 后顺序消费），不能被当成“残留报文”清掉
        if self._scripted_reports:
            return []
        return super().drain(timeout=timeout, max_messages=max_messages)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _ensure_open(self) -> None:
        if not self._opened:
            raise TransportError("Mock 传输层尚未打开，请先调用 open()")

    def _make_report(self, target: Target) -> str | None:
        if self.scenario == "timeout":
            return None
        if self.scenario == "bad_json":
            return "这不是合法的 JSON 报文"
        if self.scenario == "unknown_status":
            return '{"status":"paused","path":[]}'
        if self.scenario == "fail":
            return build_fail_report()
        if self.scenario == "empty_path":
            return '{"status":"running","path":[]}'

        path = find_path(self.start, (target.x, target.y), self.obstacles)
        if path is None:
            logger.info("Mock 传输层：目标 %s 被障碍物占据或不可达，返回 fail", target)
            return build_fail_report()
        return build_running_report(path)