"""Mock 传输层单元测试：BFS 路径、失败场景、异常注入、命令记录。"""

from __future__ import annotations

import unittest

from brain.errors import (
    InvalidReportPathError,
    MalformedReportError,
    TransportError,
    UnknownStatusError,
)
from brain.protocol import Point, Target, build_target_message, parse_robot_report
from brain.transport.mock import MockRobotTransport


class MockRobotTransportTests(unittest.TestCase):
    def _open(self, **kwargs: object) -> MockRobotTransport:
        transport = MockRobotTransport(**kwargs)  # type: ignore[arg-type]
        transport.open()
        self.addCleanup(transport.close)
        return transport

    def test_normal_running_path(self) -> None:
        transport = self._open()
        transport.send_line(build_target_message(Target(5, 5)))

        raw = transport.recv_line(0.1)
        self.assertIsNotNone(raw)
        report = parse_robot_report(raw)
        self.assertEqual(report.status, "running")
        self.assertEqual(report.path[0], Point(0, 0))
        self.assertEqual(report.path[-1], Point(5, 5))
        self.assertEqual(report.step_count, 10)
        # 路径必须四连通（每一步只上下左右移动一格）
        for previous, current in zip(report.path, report.path[1:]):
            distance = abs(previous.x - current.x) + abs(previous.y - current.y)
            self.assertEqual(distance, 1, f"路径不连通：{previous} -> {current}")

    def test_obstacle_target_returns_fail(self) -> None:
        transport = self._open(obstacles={(1, 1)})
        transport.send_line(build_target_message(Target(1, 1)))
        raw = transport.recv_line(0.1)
        report = parse_robot_report(raw)
        self.assertEqual(report.status, "fail")
        self.assertEqual(report.path, ())

    def test_unreachable_target_returns_fail(self) -> None:
        # 目标 (1,1) 的左侧和上方都被堵住 -> 从 (0,0) 绕不过去
        transport = self._open(obstacles={(0, 1), (1, 0)})
        transport.send_line(build_target_message(Target(1, 1)))
        raw = transport.recv_line(0.1)
        report = parse_robot_report(raw)
        self.assertEqual(report.status, "fail")

    def test_already_at_target(self) -> None:
        transport = self._open(start=(3, 2))
        transport.send_line(build_target_message(Target(3, 2)))
        raw = transport.recv_line(0.1)
        report = parse_robot_report(raw)
        self.assertEqual(report.status, "running")
        self.assertEqual(report.path, (Point(3, 2),))
        self.assertEqual(report.step_count, 0)

    def test_timeout_scenario(self) -> None:
        transport = self._open(scenario="timeout")
        transport.send_line(build_target_message(Target(5, 5)))
        self.assertIsNone(transport.recv_line(0.1))

    def test_bad_json_scenario(self) -> None:
        transport = self._open(scenario="bad_json")
        transport.send_line(build_target_message(Target(5, 5)))
        raw = transport.recv_line(0.1)
        with self.assertRaises(MalformedReportError):
            parse_robot_report(raw)  # type: ignore[arg-type]

    def test_unknown_status_scenario(self) -> None:
        transport = self._open(scenario="unknown_status")
        transport.send_line(build_target_message(Target(5, 5)))
        raw = transport.recv_line(0.1)
        with self.assertRaises(UnknownStatusError):
            parse_robot_report(raw)  # type: ignore[arg-type]

    def test_empty_path_scenario(self) -> None:
        transport = self._open(scenario="empty_path")
        transport.send_line(build_target_message(Target(5, 5)))
        raw = transport.recv_line(0.1)
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report(raw)  # type: ignore[arg-type]

    def test_send_error_scenario(self) -> None:
        transport = self._open(scenario="send_error")
        with self.assertRaises(TransportError):
            transport.send_line(build_target_message(Target(5, 5)))

    def test_send_before_open(self) -> None:
        transport = MockRobotTransport()
        with self.assertRaises(TransportError):
            transport.send_line(build_target_message(Target(5, 5)))

    def test_invalid_command_recorded_and_answered_with_fail(self) -> None:
        transport = self._open()
        transport.send_line('{"target_x": 9, "target_y": 2}')
        self.assertEqual(len(transport.invalid_commands), 1)
        self.assertEqual(transport.received_targets, [])
        report = parse_robot_report(transport.recv_line(0.1))  # type: ignore[arg-type]
        self.assertEqual(report.status, "fail")

    def test_scripted_reports_take_precedence(self) -> None:
        transport = self._open(
            scripted_reports=[None, '{"status":"fail","path":[]}']
        )
        transport.send_line(build_target_message(Target(5, 5)))
        self.assertIsNone(transport.recv_line(0.1))

        transport.send_line(build_target_message(Target(4, 4)))
        raw = transport.recv_line(0.1)
        report = parse_robot_report(raw)  # type: ignore[arg-type]
        self.assertEqual(report.status, "fail")

    def test_describe_mentions_mock(self) -> None:
        transport = self._open()
        self.assertIn("Mock", transport.describe())

    def test_invalid_scenario_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MockRobotTransport(scenario="不存在的场景")

    def test_close_is_idempotent(self) -> None:
        transport = MockRobotTransport()
        transport.open()
        transport.close()
        transport.close()
        self.assertFalse(transport.is_open)


if __name__ == "__main__":
    unittest.main()