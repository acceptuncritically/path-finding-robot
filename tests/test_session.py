"""会话状态机单元测试：成功/失败/超时/协议错/通信错/重试/并发保护/连续对话。"""

from __future__ import annotations

import threading
import unittest

from brain.errors import LLMError, SessionBusyError, SessionClosedError, TransportError
from brain.feedback import FeedbackRenderer
from brain.llm.base import LLMClient
from brain.llm.mock import MockLLM
from brain.planner import TargetPlanner
from brain.protocol import build_target_message, Point, Target
from brain.session import BrainSession, TaskStatus
from brain.transport.mock import MockRobotTransport


class _FailingLLM(LLMClient):
    name = "failing"

    def complete(self, system_prompt: str, user_message: str) -> str:
        raise LLMError("模拟大模型故障")


class _BlockingTransport(MockRobotTransport):
    """recv 时阻塞，用于测试并发保护。"""

    def __init__(self) -> None:
        super().__init__()
        self.recv_started = threading.Event()
        self.release_recv = threading.Event()

    def recv_line(self, timeout: float) -> str | None:
        self.recv_started.set()
        self.release_recv.wait(timeout=5.0)
        return super().recv_line(timeout)


def _make_session(
    llm: LLMClient | None = None,
    transport: MockRobotTransport | None = None,
    report_timeout: float = 1.0,
) -> BrainSession:
    return BrainSession(
        planner=TargetPlanner(llm or MockLLM(), max_retries=2),
        transport=transport or MockRobotTransport(),
        feedback=FeedbackRenderer(),
        report_timeout=report_timeout,
    )


class SessionSuccessTests(unittest.TestCase):
    def test_running_outcome(self) -> None:
        transport = MockRobotTransport()
        session = _make_session(transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去迷宫右下角")

        self.assertEqual(outcome.status, TaskStatus.RUNNING)
        self.assertTrue(outcome.is_ok)
        self.assertEqual(outcome.target, Target(5, 5))
        self.assertIsNotNone(outcome.report)
        self.assertEqual(outcome.report.path[-1], Point(5, 5))
        self.assertIn("已规划路线", outcome.user_message)
        self.assertEqual(
            transport.sent_lines, ['{"target_x":5,"target_y":5}\n']
        )
        self.assertEqual(session.state, "IDLE")
        self.assertEqual(session.task_count, 1)
        self.assertIs(session.last_outcome, outcome)

    def test_fail_outcome(self) -> None:
        session = _make_session()
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(1,1)")

        self.assertEqual(outcome.status, TaskStatus.FAIL)
        self.assertTrue(outcome.is_ok)
        self.assertIn("阻挡", outcome.user_message)

    def test_already_at_target(self) -> None:
        transport = MockRobotTransport(start=(3, 2))
        session = _make_session(transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(3,2)")

        self.assertEqual(outcome.status, TaskStatus.RUNNING)
        self.assertIn("无需移动", outcome.user_message)

    def test_retry_then_success(self) -> None:
        llm = MockLLM(
            scripted_outputs=[
                '{"target_x": 9, "target_y": 9}',
                '{"target_x": 4, "target_y": 5}',
            ]
        )
        transport = MockRobotTransport()
        session = _make_session(llm=llm, transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(9,9)")

        self.assertEqual(outcome.status, TaskStatus.RUNNING)
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(transport.sent_lines, ['{"target_x":4,"target_y":5}\n'])


class SessionFailureTests(unittest.TestCase):
    def test_invalid_instruction_never_sends(self) -> None:
        llm = MockLLM(scripted_outputs=["这是一段乱码"] * 3)
        transport = MockRobotTransport()
        session = _make_session(llm=llm, transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(3,2)")

        self.assertEqual(outcome.status, TaskStatus.INVALID_INSTRUCTION)
        self.assertFalse(outcome.is_ok)
        self.assertEqual(transport.sent_lines, [])
        self.assertIn("无法理解", outcome.user_message)

    def test_uncertain_instruction(self) -> None:
        session = _make_session()
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("随便走走吧")

        self.assertEqual(outcome.status, TaskStatus.INVALID_INSTRUCTION)
        self.assertIn("无法确定目标", outcome.user_message)

    def test_llm_error(self) -> None:
        session = _make_session(llm=_FailingLLM(), transport=MockRobotTransport())
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去右下角")

        self.assertEqual(outcome.status, TaskStatus.LLM_ERROR)
        self.assertIn("大模型调用失败", outcome.user_message)

    def test_timeout(self) -> None:
        transport = MockRobotTransport(scenario="timeout")
        session = _make_session(transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("回到(0,0)起点")

        self.assertEqual(outcome.status, TaskStatus.TIMEOUT)
        self.assertIn("响应", outcome.user_message)
        self.assertEqual(transport.sent_lines, ['{"target_x":0,"target_y":0}\n'])

    def test_bad_json_report(self) -> None:
        session = _make_session(transport=MockRobotTransport(scenario="bad_json"))
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(3,2)")

        self.assertEqual(outcome.status, TaskStatus.PROTOCOL_ERROR)
        self.assertIsNotNone(outcome.raw_report)
        self.assertIn("格式异常", outcome.user_message)

    def test_unknown_status_report(self) -> None:
        session = _make_session(transport=MockRobotTransport(scenario="unknown_status"))
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(3,2)")

        self.assertEqual(outcome.status, TaskStatus.PROTOCOL_ERROR)
        self.assertIn("未知状态", outcome.user_message)

    def test_running_with_empty_path_report(self) -> None:
        session = _make_session(transport=MockRobotTransport(scenario="empty_path"))
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(3,2)")

        self.assertEqual(outcome.status, TaskStatus.PROTOCOL_ERROR)
        self.assertIn("格式异常", outcome.user_message)

    def test_send_error(self) -> None:
        transport = MockRobotTransport(scenario="send_error")
        session = _make_session(transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcome = session.handle_instruction("去(2,2)")

        self.assertEqual(outcome.status, TaskStatus.TRANSPORT_ERROR)
        self.assertIn("通信异常", outcome.user_message)


class SessionLifecycleTests(unittest.TestCase):
    def test_continuous_dialogue(self) -> None:
        session = _make_session()
        session.open()

        first = session.handle_instruction("去(1,1)")   # 障碍物堵死
        second = session.handle_instruction("去(5,5)")  # 正常寻路

        self.assertEqual(first.status, TaskStatus.FAIL)
        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertEqual(session.task_count, 2)
        self.assertEqual(session.state, "IDLE")
        self.assertEqual(len(session.transport.sent_lines), 2)
        session.close()

    def test_closed_session_rejects_instruction(self) -> None:
        session = _make_session()
        session.open()
        session.close()
        session.close()  # 幂等
        with self.assertRaises(SessionClosedError):
            session.handle_instruction("去(3,2)")

    def test_busy_guard(self) -> None:
        transport = _BlockingTransport()
        session = _make_session(transport=transport)
        session.open()
        self.addCleanup(session.close)

        outcomes: dict[str, TaskStatus] = {}

        def worker() -> None:
            outcome = session.handle_instruction("去(5,5)")
            outcomes["status"] = outcome.status

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertTrue(transport.recv_started.wait(2.0), "等待 recv 开始超时")
        with self.assertRaises(SessionBusyError):
            session.handle_instruction("去(2,2)")

        transport.release_recv.set()
        thread.join(5.0)
        self.assertFalse(thread.is_alive(), "worker 线程未结束")
        self.assertEqual(outcomes.get("status"), TaskStatus.RUNNING)
        self.assertEqual(len(transport.sent_lines), 1, "并发保护失效：第二条指令被下发")

    def test_report_timeout_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            _make_session(report_timeout=0)


class _LateReplyTransport(MockRobotTransport):
    """第一次任务“超时”，但迟到的回包仍留在缓冲区。

    用于验证：新任务下发前会 drain 掉上一任务的迟到回包，避免错配。
    """

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0
        self._held: str | None = None

    def recv_line(self, timeout: float) -> str | None:
        self._calls += 1
        if self._calls == 1:
            return None  # 新任务下发前的 drain：没有残留
        if self._calls == 2:
            # 偷偷收下 C 端回包但本任务表现为“超时”（模拟回包迟到）
            self._held = super().recv_line(timeout)
            return None
        if self._held is not None:
            line, self._held = self._held, None
            return line
        return super().recv_line(timeout)


class SessionStaleReplyTests(unittest.TestCase):
    def test_stale_reply_not_attributed_to_next_task(self) -> None:
        transport = _LateReplyTransport()
        session = _make_session(transport=transport)
        session.open()
        self.addCleanup(session.close)

        first = session.handle_instruction("去(5,5)")
        self.assertEqual(first.status, TaskStatus.TIMEOUT)

        # 第二条指令目标 (0,0)；若不做 drain，会把上一条迟到的 (5,5) 路径错当成本次结果
        second = session.handle_instruction("回到(0,0)起点")
        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertIsNotNone(second.report)
        self.assertEqual((second.report.path[-1].x, second.report.path[-1].y), (0, 0))
        self.assertIn("无需移动", second.user_message)
        self.assertEqual(transport.sent_lines[-1], '{"target_x":0,"target_y":0}\n')


class _OpenFailingTransport(MockRobotTransport):
    def open(self) -> None:
        raise TransportError("模拟连接失败")


class SessionOpenTests(unittest.TestCase):
    def test_open_failure_propagates(self) -> None:
        session = _make_session(transport=_OpenFailingTransport())
        with self.assertRaises(TransportError):
            session.open()


if __name__ == "__main__":
    unittest.main()