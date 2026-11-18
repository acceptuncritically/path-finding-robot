"""人话反馈单元测试。"""

from __future__ import annotations

import unittest

from brain.errors import (
    MalformedReportError,
    PlanningFailure,
    TransportError,
    UnknownStatusError,
)
from brain.feedback import FeedbackRenderer
from brain.protocol import Point, RobotReport, Target


def _running_report(*points: tuple[int, int]) -> RobotReport:
    return RobotReport(status="running", path=tuple(Point(x=x, y=y) for x, y in points))


class RunningFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feedback = FeedbackRenderer()

    def test_running_message_contains_path_and_steps(self) -> None:
        report = _running_report((0, 0), (1, 0), (1, 1))
        message = self.feedback.running(report, Target(1, 1))
        self.assertIn("已规划路线", message)
        self.assertIn("[(0,0),(1,0),(1,1)]", message)
        self.assertIn("共 2 步", message)
        self.assertIn("目标点", message)

    def test_running_without_target_argument(self) -> None:
        report = _running_report((0, 0), (1, 0))
        message = self.feedback.running(report)
        self.assertIn("已规划路线", message)
        self.assertNotIn("目标点", message)

    def test_single_point_means_already_there(self) -> None:
        report = _running_report((3, 2))
        message = self.feedback.running(report)
        self.assertIn("无需移动", message)


class FailureFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feedback = FeedbackRenderer()

    def test_fail_message(self) -> None:
        message = self.feedback.fail(Target(1, 1))
        self.assertIn("障碍物阻挡", message)
        self.assertIn("请更换目标", message)

    def test_planning_failure_uncertain(self) -> None:
        message = self.feedback.planning_failure(
            PlanningFailure("模型无法确定", kind="uncertain", attempts=1)
        )
        self.assertIn("无法确定目标", message)

    def test_planning_failure_range(self) -> None:
        message = self.feedback.planning_failure(
            PlanningFailure("坐标超出 0~5 范围：x=9, y=9", kind="range", attempts=3)
        )
        self.assertIn("超出迷宫范围", message)
        self.assertIn("请重新输入", message)

    def test_planning_failure_llm(self) -> None:
        message = self.feedback.planning_failure(
            PlanningFailure("网络不可达", kind="llm", attempts=1)
        )
        self.assertIn("大模型调用失败", message)

    def test_planning_failure_format(self) -> None:
        message = self.feedback.planning_failure(
            PlanningFailure("找不到合法 JSON", kind="format", attempts=3)
        )
        self.assertIn("无法理解", message)


class ErrorFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feedback = FeedbackRenderer()

    def test_protocol_error_unknown_status(self) -> None:
        message = self.feedback.protocol_error(UnknownStatusError("paused", "{}"))
        self.assertIn("未知状态", message)
        self.assertIn("paused", message)

    def test_protocol_error_generic(self) -> None:
        message = self.feedback.protocol_error(MalformedReportError("不是合法 JSON", "xxx"))
        self.assertIn("格式异常", message)
        self.assertIn("不是合法 JSON", message)

    def test_transport_error(self) -> None:
        message = self.feedback.transport_error(TransportError("连接被拒绝"))
        self.assertIn("通信异常", message)
        self.assertIn("连接被拒绝", message)

    def test_timeout_integer_and_fraction(self) -> None:
        self.assertIn("5 秒", self.feedback.timeout(5))
        self.assertIn("1.5 秒", self.feedback.timeout(1.5))

    def test_goodbye_and_help(self) -> None:
        self.assertIn("再见", self.feedback.goodbye())
        help_text = self.feedback.help_text()
        self.assertIn("右下角", help_text)
        self.assertIn("退出", help_text)


if __name__ == "__main__":
    unittest.main()