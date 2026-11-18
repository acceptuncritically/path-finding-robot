"""规划器单元测试：重试策略、错误归类、失败语义。"""

from __future__ import annotations

import unittest

from brain.errors import LLMError, PlanningFailure
from brain.llm.base import LLMClient
from brain.llm.mock import MockLLM
from brain.planner import TargetPlanner
from brain.protocol import Target


class _FailingLLM(LLMClient):
    name = "failing"

    def __init__(self, message: str = "网络不可达") -> None:
        self.message = message
        self.calls = 0

    def complete(self, system_prompt: str, user_message: str) -> str:
        self.calls += 1
        raise LLMError(self.message)


class TargetPlannerTests(unittest.TestCase):
    def test_success_first_attempt(self) -> None:
        llm = MockLLM(scripted_outputs=['{"target_x": 3, "target_y": 2}'])
        planner = TargetPlanner(llm, max_retries=2)
        result = planner.plan("去(3,2)")
        self.assertEqual(result.target, Target(3, 2))
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(result.raw_outputs), 1)
        self.assertEqual(len(llm.calls), 1)

    def test_retry_after_bad_output(self) -> None:
        llm = MockLLM(scripted_outputs=["垃圾输出", '{"target_x": 4, "target_y": 4}'])
        planner = TargetPlanner(llm, max_retries=2)
        result = planner.plan("去(4,4)")
        self.assertEqual(result.target, Target(4, 4))
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("输出不可用", llm.calls[1])

    def test_retry_after_out_of_range_then_success(self) -> None:
        llm = MockLLM(
            scripted_outputs=[
                '{"target_x": 9, "target_y": 9}',
                '{"target_x": 2, "target_y": 1}',
            ]
        )
        planner = TargetPlanner(llm, max_retries=2)
        result = planner.plan("去(9,9)")
        self.assertEqual(result.target, Target(2, 1))
        self.assertEqual(result.attempts, 2)
        self.assertIn("超出", llm.calls[1])

    def test_exhausted_retries_with_format_error(self) -> None:
        llm = MockLLM(scripted_outputs=["垃圾"] * 3)
        planner = TargetPlanner(llm, max_retries=2)
        with self.assertRaises(PlanningFailure) as ctx:
            planner.plan("去(3,2)")
        failure = ctx.exception
        self.assertEqual(failure.kind, "format")
        self.assertEqual(failure.attempts, 3)
        self.assertEqual(failure.last_raw, "垃圾")

    def test_exhausted_retries_with_range_error(self) -> None:
        llm = MockLLM(scripted_outputs=['{"target_x": 9, "target_y": 9}'] * 3)
        planner = TargetPlanner(llm, max_retries=2)
        with self.assertRaises(PlanningFailure) as ctx:
            planner.plan("去(9,9)")
        self.assertEqual(ctx.exception.kind, "range")

    def test_uncertain_target_no_retry(self) -> None:
        llm = MockLLM(scripted_outputs=['{"target_x": -1, "target_y": -1}'])
        planner = TargetPlanner(llm, max_retries=2)
        with self.assertRaises(PlanningFailure) as ctx:
            planner.plan("随便走走吧")
        self.assertEqual(ctx.exception.kind, "uncertain")
        self.assertEqual(len(llm.calls), 1)

    def test_llm_error_maps_to_kind_llm(self) -> None:
        llm = _FailingLLM()
        planner = TargetPlanner(llm, max_retries=2)
        with self.assertRaises(PlanningFailure) as ctx:
            planner.plan("去右下角")
        self.assertEqual(ctx.exception.kind, "llm")
        self.assertIn("网络不可达", ctx.exception.message)

    def test_empty_instruction(self) -> None:
        llm = MockLLM()
        planner = TargetPlanner(llm, max_retries=2)
        with self.assertRaises(PlanningFailure) as ctx:
            planner.plan("   ")
        self.assertEqual(ctx.exception.kind, "format")
        self.assertEqual(ctx.exception.attempts, 0)
        self.assertEqual(llm.calls, [])

    def test_negative_max_retries_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TargetPlanner(MockLLM(), max_retries=-1)


if __name__ == "__main__":
    unittest.main()