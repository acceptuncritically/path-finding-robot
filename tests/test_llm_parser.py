"""大模型输出解析单元测试：从宽到严的容错 + 严格校验。"""

from __future__ import annotations

import unittest

from brain.errors import TargetFormatError, TargetOutOfRangeError, TargetUncertainError
from brain.llm.parser import parse_target_from_llm_output
from brain.protocol import Target


class ParseTargetFromLLMOutputTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(
            parse_target_from_llm_output('{"target_x": 3, "target_y": 2}'),
            Target(3, 2),
        )

    def test_compact_json(self) -> None:
        self.assertEqual(
            parse_target_from_llm_output('{"target_x":3,"target_y":2}'), Target(3, 2)
        )

    def test_markdown_fence(self) -> None:
        raw = '```json\n{"target_x": 3, "target_y": 2}\n```'
        self.assertEqual(parse_target_from_llm_output(raw), Target(3, 2))

    def test_fence_without_language(self) -> None:
        raw = '```\n{"target_x": 0, "target_y": 5}\n```'
        self.assertEqual(parse_target_from_llm_output(raw), Target(0, 5))

    def test_explanation_around_json(self) -> None:
        raw = '好的，目标应该是：{"target_x": 5, "target_y": 5}，请确认。'
        self.assertEqual(parse_target_from_llm_output(raw), Target(5, 5))

    def test_extra_fields_ignored(self) -> None:
        raw = '{"target_x": 3, "target_y": 2, "reason": "用户要求"}'
        self.assertEqual(parse_target_from_llm_output(raw), Target(3, 2))

    def test_duplicate_same_target_deduplicated(self) -> None:
        raw = '{"target_x": 1, "target_y": 1} 或者 {"target_x": 1, "target_y": 1}'
        self.assertEqual(parse_target_from_llm_output(raw), Target(1, 1))

    def test_regex_fallback_without_braces(self) -> None:
        raw = "target_x: 4, target_y: 3"
        self.assertEqual(parse_target_from_llm_output(raw), Target(4, 3))

    def test_ambiguous_multiple_targets_rejected(self) -> None:
        raw = '{"target_x": 3, "target_y": 2} 或 {"target_x": 5, "target_y": 5}'
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output(raw)

    def test_example_coordinate_with_uncertain_text(self) -> None:
        # 模型只是举例说明格式、并没有给出真实目标 -> 不能猜坐标（应视为“无法确定”）
        raw = "我无法确定目标，示例格式：target_x: 3, target_y: 2"
        with self.assertRaises(TargetUncertainError):
            parse_target_from_llm_output(raw)

    def test_example_coordinate_with_prose_not_used(self) -> None:
        raw = "示例格式 target_x: 3, target_y: 2，请按此格式输出"
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output(raw)

    def test_float_negative_one_not_treated_as_uncertain(self) -> None:
        # 浮点 -1.0 必须按类型错误拒绝，而不是被当成 -1,-1“无法确定”标记
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output('{"target_x": -1.0, "target_y": -1.0}')

    def test_deeply_nested_json_does_not_crash(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output("[" * 3000 + "]" * 3000)

    def test_uncertain_sentinel(self) -> None:
        with self.assertRaises(TargetUncertainError):
            parse_target_from_llm_output('{"target_x": -1, "target_y": -1}')

    def test_out_of_range_positive(self) -> None:
        with self.assertRaises(TargetOutOfRangeError):
            parse_target_from_llm_output('{"target_x": 6, "target_y": 5}')

    def test_out_of_range_negative(self) -> None:
        with self.assertRaises(TargetOutOfRangeError):
            parse_target_from_llm_output('{"target_x": -2, "target_y": 3}')

    def test_string_number_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output('{"target_x": "3", "target_y": 2}')

    def test_float_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output('{"target_x": 3.0, "target_y": 2}')

    def test_bool_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output('{"target_x": true, "target_y": 2}')

    def test_missing_field_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output('{"target_x": 3}')

    def test_non_json_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output("我觉得应该去右下角")

    def test_empty_rejected(self) -> None:
        for raw in ("", "   "):
            with self.subTest(raw=raw), self.assertRaises(TargetFormatError):
                parse_target_from_llm_output(raw)

    def test_non_string_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output(None)  # type: ignore[arg-type]

    def test_non_object_json_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output("[1, 2, 3]")

    def test_broken_fence_rejected(self) -> None:
        with self.assertRaises(TargetFormatError):
            parse_target_from_llm_output("```\n这里什么都没有\n```")


if __name__ == "__main__":
    unittest.main()