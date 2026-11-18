"""离线 Mock 大模型：不联网，按规则解析中文指令，用于开发和单元测试。

它返回的是**原始文本**（模拟真实模型的输出），走和真实模型完全一样的
解析 / 校验 / 重试链路，因此可以放心用来本地验证整条业务逻辑。

两种用法：
1. 常规：``MockLLM()``，按内置规则理解中文指令；
2. 测试：``MockLLM(scripted_outputs=[...])``，按顺序返回预设的原始输出
   （塞非法 JSON / 越界坐标 / 空字符串，就能测重试与失败分支）。
"""

from __future__ import annotations

import re
from typing import Sequence

from .base import LLMClient

# 坐标写法："(5,5)"、"（3，2）"、"去5,5"
_COORD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[（(]\s*(\d+)\s*[，,]\s*(\d+)\s*[)）]"),
    re.compile(r"坐标\s*[（(]?\s*(\d+)\s*[，,]\s*(\d+)"),
    re.compile(r"(\d+)\s*[，,]\s*(\d+)"),
)

# “第 6 行第 3 列”写法
_ROW_COL_PATTERN = re.compile(r"第\s*(\d+)\s*行\s*第\s*(\d+)\s*列")

# 方位关键词 -> 目标坐标（顺序敏感：先匹配更具体的说法）
_KEYWORD_TARGETS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("右下角", (5, 5)),
    ("右上角", (0, 5)),
    ("左下角", (5, 0)),
    ("左上角", (0, 0)),
    ("起点对面", (5, 5)),
    ("回到起点", (0, 0)),
    ("回起点", (0, 0)),
    ("原点", (0, 0)),
    ("起点", (0, 0)),
    ("正中央", (2, 2)),
    ("中心", (2, 2)),
    ("中间", (2, 2)),
)

_UNCERTAIN_WORDS = ("随便", "不知道", "不确定", "无所谓")

_INSTRUCTION_PATTERN = re.compile(r"用户指令[:：]\s*(.+)")


class MockLLM(LLMClient):
    """离线规则版大模型（确定性输出，便于测试）。"""

    name = "mock"

    def __init__(self, scripted_outputs: Sequence[str] | None = None) -> None:
        self._scripted: list[str] = list(scripted_outputs or [])
        #: 记录每次收到的 user_message，测试里可断言重试次数/纠正提示
        self.calls: list[str] = []

    def complete(self, system_prompt: str, user_message: str) -> str:
        self.calls.append(user_message)
        if self._scripted:
            return self._scripted.pop(0)
        return self._rule_based(user_message)

    @staticmethod
    def _extract_instruction(user_message: str) -> str:
        match = _INSTRUCTION_PATTERN.search(user_message)
        if not match:
            return user_message.strip()
        instruction = match.group(1).strip()
        # 重试消息里“用户指令：xxx”之后还有纠正说明，只取第一行
        return instruction.splitlines()[0].strip() if instruction else ""

    def _rule_based(self, user_message: str) -> str:
        instruction = self._extract_instruction(user_message)

        row_col_match = _ROW_COL_PATTERN.search(instruction)
        if row_col_match:
            row = int(row_col_match.group(1))
            col = int(row_col_match.group(2))
            return self._to_json(row - 1, col - 1)

        for pattern in _COORD_PATTERNS:
            coord_match = pattern.search(instruction)
            if coord_match:
                return self._to_json(int(coord_match.group(1)), int(coord_match.group(2)))

        for keyword, (x, y) in _KEYWORD_TARGETS:
            if keyword in instruction:
                return self._to_json(x, y)

        if any(word in instruction for word in _UNCERTAIN_WORDS):
            return '{"target_x": -1, "target_y": -1}'

        # 模拟“模型没按要求输出 JSON”的情况，用于测试重试与最终失败分支
        return (
            "我不太确定您想去的目标位置，可以告诉我具体坐标"
            "（如：去(3,2)）或方位（如：右下角）吗？"
        )

    @staticmethod
    def _to_json(x: int, y: int) -> str:
        return f'{{"target_x": {x}, "target_y": {y}}}'