"""任务状态机：一条指令从输入到反馈的完整生命周期。

设计目标（解耦 / 可复用）：
- 不依赖 ``input()`` / ``print()``：交互方式由调用方决定（控制台、Web、语音都行）；
- 依赖注入：``TargetPlanner``（大模型）与 ``Transport``（C 端通信）都可替换；
- 串行策略：一次只处理一个任务（带并发保护），符合“一问一答”的通信协议。

状态流转::

    IDLE -> PLANNING -> SENDING -> WAITING_REPORT -> IDLE
              │            │              │
              └── 失败 ────┴── 失败 ──────┴── 超时/协议错 ──> IDLE（可继续下一条）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum

from .errors import (
    PlanningFailure,
    ProtocolError,
    SessionBusyError,
    SessionClosedError,
    TransportError,
)
from .feedback import FeedbackRenderer
from .planner import TargetPlanner
from .protocol import RobotReport, Target, build_target_message, parse_robot_report
from .transport.base import Transport

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """单条指令的处理结果类别。"""

    RUNNING = "running"                      # C 端寻路成功（已规划路径）
    FAIL = "fail"                            # C 端寻路失败（障碍物堵死）
    INVALID_INSTRUCTION = "invalid_instruction"  # 指令无法解析成合法目标
    LLM_ERROR = "llm_error"                  # 大模型调用失败
    TRANSPORT_ERROR = "transport_error"      # 串口/TCP 通信异常
    PROTOCOL_ERROR = "protocol_error"        # C 端回包不符合协议
    TIMEOUT = "timeout"                      # 等待 C 端回包超时

    @property
    def is_ok(self) -> bool:
        """是否完成了一次有效任务（无论 C 端寻路成功还是失败）。"""
        return self in (TaskStatus.RUNNING, TaskStatus.FAIL)


@dataclass(frozen=True)
class TaskOutcome:
    """一次 ``handle_instruction`` 的完整结果（可直接用于 UI 展示 / 测试断言）。"""

    status: TaskStatus
    user_message: str
    target: Target | None = None
    report: RobotReport | None = None
    raw_report: str | None = None
    attempts: int = 0

    @property
    def is_ok(self) -> bool:
        return self.status.is_ok


class BrainSession:
    """串行任务状态机（可被控制台 / Web 接口复用）。"""

    def __init__(
        self,
        planner: TargetPlanner,
        transport: Transport,
        feedback: FeedbackRenderer | None = None,
        report_timeout: float = 5.0,
    ) -> None:
        if report_timeout <= 0:
            raise ValueError(f"report_timeout 必须大于 0：{report_timeout}")
        self._planner = planner
        self._transport = transport
        self._feedback = feedback or FeedbackRenderer()
        self._report_timeout = report_timeout
        self._lock = threading.Lock()
        self._state = "IDLE"
        self._closed = False
        self._last_outcome: TaskOutcome | None = None
        self._task_count = 0

    # ------------------------------------------------------------------ #
    # 只读属性
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state

    @property
    def last_outcome(self) -> TaskOutcome | None:
        return self._last_outcome

    @property
    def task_count(self) -> int:
        return self._task_count

    @property
    def planner(self) -> TargetPlanner:
        return self._planner

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def feedback(self) -> FeedbackRenderer:
        return self._feedback

    @property
    def report_timeout(self) -> float:
        return self._report_timeout

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """打开通信连接（Mock 为空操作；TCP/串口会真正建立连接）。"""
        if self._closed:
            raise SessionClosedError("会话已关闭，不能再次打开")
        self._transport.open()

    def close(self) -> None:
        """关闭通信连接（可重复调用）。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._transport.close()
        except TransportError as exc:  # 关闭失败不向上抛，只记日志
            logger.warning("关闭传输层时出错：%s", exc)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def handle_instruction(self, instruction: str) -> TaskOutcome:
        """处理一条用户指令，返回可供展示的结果。

        Raises:
            SessionBusyError: 上一条指令尚未结束（并发保护）。
            SessionClosedError: 会话已关闭。
        """
        if self._closed:
            raise SessionClosedError("会话已关闭，不能处理新指令")
        if not self._lock.acquire(blocking=False):
            raise SessionBusyError("上一次任务还没有结束，请等收到反馈后再输入新指令")

        try:
            self._state = "PLANNING"
            try:
                plan = self._planner.plan(instruction)
            except PlanningFailure as failure:
                status = (
                    TaskStatus.LLM_ERROR
                    if failure.kind == "llm"
                    else TaskStatus.INVALID_INSTRUCTION
                )
                logger.warning(
                    "规划失败：kind=%s attempts=%d reason=%s",
                    failure.kind,
                    failure.attempts,
                    failure.message,
                )
                return self._finish(
                    TaskOutcome(
                        status=status,
                        user_message=self._feedback.planning_failure(failure),
                        attempts=failure.attempts,
                    )
                )

            target = plan.target
            logger.info("最终目标：%s", target)

            # 下发前清理上一任务可能遗留的迟到回包，避免「目标 A 却收到 B 的回包」的静默错配
            self._drain_stale_reports()
            self._state = "SENDING"
            try:
                message = build_target_message(target)
            except (TypeError, ValueError) as exc:
                # 理论上 Planner 已保证合法，这里是防御式兜底
                logger.error("组装下发报文失败：%s", exc)
                return self._finish(
                    TaskOutcome(
                        status=TaskStatus.INVALID_INSTRUCTION,
                        user_message=f"目标坐标非法，未发送给小车：{exc}",
                        target=target,
                        attempts=plan.attempts,
                    )
                )

            try:
                self._transport.send_line(message)
            except TransportError as exc:
                logger.error("下发报文失败：%s", exc)
                return self._finish(
                    TaskOutcome(
                        status=TaskStatus.TRANSPORT_ERROR,
                        user_message=self._feedback.transport_error(exc),
                        target=target,
                        attempts=plan.attempts,
                    )
                )
            logger.info("已下发 C 端：%s", message.strip())

            self._state = "WAITING_REPORT"
            try:
                raw_report = self._transport.recv_line(self._report_timeout)
            except TransportError as exc:
                logger.error("等待回包失败：%s", exc)
                return self._finish(
                    TaskOutcome(
                        status=TaskStatus.TRANSPORT_ERROR,
                        user_message=self._feedback.transport_error(exc),
                        target=target,
                        attempts=plan.attempts,
                    )
                )

            if raw_report is None:
                logger.warning("等待 C 端回包超时（%.1fs）", self._report_timeout)
                return self._finish(
                    TaskOutcome(
                        status=TaskStatus.TIMEOUT,
                        user_message=self._feedback.timeout(self._report_timeout),
                        target=target,
                        attempts=plan.attempts,
                    )
                )

            logger.info("收到 C 端回包：%s", raw_report)
            try:
                report = parse_robot_report(raw_report)
            except ProtocolError as exc:
                logger.warning("C 端回包协议错误：%s（原始报文：%r）", exc.reason, raw_report)
                return self._finish(
                    TaskOutcome(
                        status=TaskStatus.PROTOCOL_ERROR,
                        user_message=self._feedback.protocol_error(exc),
                        target=target,
                        raw_report=raw_report,
                        attempts=plan.attempts,
                    )
                )

            if report.status == "running":
                outcome = TaskOutcome(
                    status=TaskStatus.RUNNING,
                    user_message=self._feedback.running(report, target),
                    target=target,
                    report=report,
                    raw_report=raw_report,
                    attempts=plan.attempts,
                )
            else:
                outcome = TaskOutcome(
                    status=TaskStatus.FAIL,
                    user_message=self._feedback.fail(target),
                    target=target,
                    report=report,
                    raw_report=raw_report,
                    attempts=plan.attempts,
                )
            return self._finish(outcome)
        finally:
            self._state = "IDLE"
            self._lock.release()

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _drain_stale_reports(self) -> list[str]:
        """丢弃传输层中残留 / 迟到的回包（上一任务超时后到达的“无主回包”）。

        不做清理的话，这些回包会被下一条指令误当成自己的结果，造成
        “目标 A 却展示 B 的路径”的静默错配。清理动作只记日志，不影响主流程；
        若清理时通信已异常，交给后续 send/recv 统一报通信错误。
        """
        try:
            discarded = self._transport.drain()
        except TransportError as exc:
            logger.warning("清理传输层残留报文时通信异常：%s", exc)
            return []
        if discarded:
            logger.warning(
                "已丢弃 %d 条残留/迟到报文（避免与本次任务错配）：%r",
                len(discarded),
                discarded,
            )
        return discarded

    def _finish(self, outcome: TaskOutcome) -> TaskOutcome:
        self._state = "IDLE"
        self._last_outcome = outcome
        self._task_count += 1
        logger.info(
            "任务结束：status=%s，用户反馈：%s",
            outcome.status.value,
            outcome.user_message,
        )
        return outcome