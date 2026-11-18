"""协议层单元测试：报文组装 / 解析 / 严格校验。"""

from __future__ import annotations

import json
import unittest

from brain.errors import (
    InvalidReportPathError,
    InvalidTargetMessageError,
    MalformedReportError,
    UnknownStatusError,
)
from brain.protocol import (
    Point,
    RobotReport,
    Target,
    build_fail_report,
    build_robot_report,
    build_running_report,
    build_target_message,
    format_path,
    parse_robot_report,
    parse_target_message,
)


class TargetTests(unittest.TestCase):
    def test_valid_target_fields(self) -> None:
        target = Target(5, 5)
        self.assertEqual(target.x, 5)
        self.assertEqual(target.y, 5)
        self.assertEqual(target.as_dict(), {"target_x": 5, "target_y": 5})
        self.assertEqual(str(target), "(5,5)")

    def test_human_text_is_1_based(self) -> None:
        self.assertIn("第 6 行第 6 列", Target(5, 5).human())
        self.assertIn("第 1 行第 1 列", Target(0, 0).human())

    def test_reject_bool_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Target(True, 1)  # bool 不是合法整数
        with self.assertRaises(ValueError):
            Target(1, False)

    def test_reject_non_int_coordinates(self) -> None:
        for bad in ("3", 3.0, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Target(bad, 1)  # type: ignore[arg-type]

    def test_reject_out_of_range(self) -> None:
        for bad in (-1, 6):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Target(bad, 0)


class BuildTargetMessageTests(unittest.TestCase):
    def test_exact_wire_format(self) -> None:
        self.assertEqual(
            build_target_message(Target(5, 5)),
            '{"target_x":5,"target_y":5}\n',
        )

    def test_single_line_and_json_parseable(self) -> None:
        line = build_target_message(Target(0, 0))
        self.assertEqual(line.count("\n"), 1)
        self.assertTrue(line.endswith("\n"))
        self.assertEqual(json.loads(line), {"target_x": 0, "target_y": 0})

    def test_requires_target_instance(self) -> None:
        with self.assertRaises(TypeError):
            build_target_message((5, 5))  # type: ignore[arg-type]

    def test_round_trip_with_parser(self) -> None:
        target = Target(3, 2)
        self.assertEqual(parse_target_message(build_target_message(target)), target)


class ParseTargetMessageTests(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(parse_target_message('{"target_x": 3, "target_y": 2}'), Target(3, 2))
        self.assertEqual(parse_target_message(' {"target_x":3,"target_y":2} '), Target(3, 2))

    def test_rejects_bad_json(self) -> None:
        with self.assertRaises(InvalidTargetMessageError):
            parse_target_message("去右下角")

    def test_rejects_missing_field(self) -> None:
        with self.assertRaises(InvalidTargetMessageError):
            parse_target_message('{"target_x": 3}')

    def test_rejects_string_number(self) -> None:
        with self.assertRaises(InvalidTargetMessageError):
            parse_target_message('{"target_x": "3", "target_y": 2}')

    def test_rejects_float_and_bool(self) -> None:
        for payload in (
            '{"target_x": 3.0, "target_y": 2}',
            '{"target_x": true, "target_y": 2}',
        ):
            with self.subTest(payload=payload), self.assertRaises(InvalidTargetMessageError):
                parse_target_message(payload)

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(InvalidTargetMessageError):
            parse_target_message('{"target_x": 6, "target_y": 2}')

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(InvalidTargetMessageError):
            parse_target_message("[1, 2]")

    def test_deeply_nested_json_rejected(self) -> None:
        with self.assertRaises(InvalidTargetMessageError):
            parse_target_message("[" * 3000 + "]" * 3000)


class ParseRobotReportTests(unittest.TestCase):
    def test_running_with_path(self) -> None:
        report = parse_robot_report(
            '{"status":"running","path":[{"x":0,"y":0},{"x":1,"y":0}]}'
        )
        self.assertIsInstance(report, RobotReport)
        self.assertEqual(report.status, "running")
        self.assertTrue(report.is_success)
        self.assertEqual(report.path, (Point(0, 0), Point(1, 0)))
        self.assertEqual(report.step_count, 1)
        self.assertEqual(report.path_text(), "[(0,0),(1,0)]")

    def test_fail_with_empty_path(self) -> None:
        report = parse_robot_report('{"status":"fail","path":[]}')
        self.assertEqual(report.status, "fail")
        self.assertFalse(report.is_success)
        self.assertEqual(report.path, ())
        self.assertEqual(report.step_count, 0)

    def test_whitespace_and_newline_tolerated(self) -> None:
        report = parse_robot_report(
            '  {"status":"running","path":[{"x":0,"y":0}]}\r\n'
        )
        self.assertEqual(report.path, (Point(0, 0),))

    def test_unknown_extra_fields_ignored(self) -> None:
        report = parse_robot_report(
            '{"status":"running","path":[{"x":0,"y":0}],"timestamp":123,"msg":"hi"}'
        )
        self.assertEqual(report.status, "running")

    def test_malformed_json(self) -> None:
        for raw in ("not json", "", "{"):
            with self.subTest(raw=raw), self.assertRaises(MalformedReportError):
                parse_robot_report(raw)

    def test_top_level_must_be_object(self) -> None:
        with self.assertRaises(MalformedReportError):
            parse_robot_report('[{"status":"running"}]')

    def test_missing_status(self) -> None:
        with self.assertRaises(MalformedReportError):
            parse_robot_report('{"path":[]}')

    def test_missing_path(self) -> None:
        with self.assertRaises(MalformedReportError):
            parse_robot_report('{"status":"running"}')

    def test_unknown_status(self) -> None:
        with self.assertRaises(UnknownStatusError) as ctx:
            parse_robot_report('{"status":"paused","path":[]}')
        self.assertEqual(ctx.exception.status, "paused")

    def test_non_string_status(self) -> None:
        with self.assertRaises(UnknownStatusError):
            parse_robot_report('{"status":123,"path":[]}')

    def test_path_must_be_list(self) -> None:
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report('{"status":"running","path":{}}')

    def test_path_item_must_be_object(self) -> None:
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report('{"status":"running","path":[[0,0]]}')

    def test_path_item_missing_field(self) -> None:
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report('{"status":"running","path":[{"x":0}]}')

    def test_path_item_wrong_type(self) -> None:
        for payload in (
            '{"status":"running","path":[{"x":true,"y":0}]}',
            '{"status":"running","path":[{"x":"0","y":0}]}',
            '{"status":"running","path":[{"x":0.0,"y":0}]}',
        ):
            with self.subTest(payload=payload), self.assertRaises(InvalidReportPathError):
                parse_robot_report(payload)

    def test_path_item_out_of_range(self) -> None:
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report('{"status":"running","path":[{"x":0,"y":6}]}')

    def test_running_with_empty_path_rejected(self) -> None:
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report('{"status":"running","path":[]}')

    def test_fail_with_non_empty_path_rejected(self) -> None:
        with self.assertRaises(InvalidReportPathError):
            parse_robot_report('{"status":"fail","path":[{"x":0,"y":0}]}')

    def test_deeply_nested_json_does_not_crash(self) -> None:
        with self.assertRaises(MalformedReportError):
            parse_robot_report("[" * 3000 + "]" * 3000)


class BuildRobotReportTests(unittest.TestCase):
    def test_build_running_exact_format(self) -> None:
        self.assertEqual(
            build_running_report([(0, 0), (1, 0)]),
            '{"status":"running","path":[{"x":0,"y":0},{"x":1,"y":0}]}',
        )

    def test_build_fail_exact_format(self) -> None:
        self.assertEqual(build_fail_report(), '{"status":"fail","path":[]}')

    def test_accepts_point_objects(self) -> None:
        self.assertEqual(
            build_running_report([Point(0, 0)]),
            '{"status":"running","path":[{"x":0,"y":0}]}',
        )

    def test_invalid_status(self) -> None:
        with self.assertRaises(ValueError):
            build_robot_report("paused", [])

    def test_running_requires_path(self) -> None:
        with self.assertRaises(ValueError):
            build_running_report([])

    def test_fail_requires_empty_path(self) -> None:
        with self.assertRaises(ValueError):
            build_robot_report("fail", [(0, 0)])


class FormatPathTests(unittest.TestCase):
    def test_format(self) -> None:
        self.assertEqual(
            format_path([Point(0, 0), Point(1, 0), Point(1, 1)]),
            "[(0,0),(1,0),(1,1)]",
        )

    def test_empty(self) -> None:
        self.assertEqual(format_path([]), "[]")


if __name__ == "__main__":
    unittest.main()